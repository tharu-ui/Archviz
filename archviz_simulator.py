
import argparse
import json
import random
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any

# ----------------------------
# Config dataclass
# ----------------------------
@dataclass
class Config:
    issue_width: int = 2
    instr_window: int = 32
    num_alus: int = 2
    num_fpus: int = 1
    branch_accuracy_pct: float = 95.0  # used only by 'random' predictor
    hw_threads: int = 2
    rob_size: int = 64
    rs_size: int = 48
    phys_regs: int = 128
    max_cycles: int = 10000
    predictor: str = "random"  # random|bimodal|gshare
    bimodal_size: int = 1024
    gshare_history: int = 8
    verbose: bool = False
    event_emit: bool = True

# ----------------------------
# Instruction / MicroOp
# ----------------------------
@dataclass
class Instr:
    id: int
    opcode: str
    srcs: List[str]
    dst: Optional[str]
    latency: int
    thread_id: int
    is_branch: bool = False
    is_mem: bool = False

@dataclass
class MicroOp:
    instr: Instr
    src_phys: List[Optional[int]]
    dst_phys: Optional[int]
    exec_remaining: int
    fu_type: str = "ALU"  # ALU/FPU/MEM
    dispatched: bool = False

# ----------------------------
# Predictors
# ----------------------------
class RandomPredictor:
    def __init__(self, accuracy_pct: float):
        self.accuracy = max(0.0, min(100.0, accuracy_pct)) / 100.0
    def predict(self, instr: Instr) -> bool:
        return random.random() < self.accuracy
    def update(self, instr: Instr, taken: bool):
        pass

class BimodalPredictor:
    def __init__(self, size: int = 1024):
        self.size = max(16, size)
        self.table = [2] * self.size  # 2-bit counters init weakly taken (2)
    def _idx(self, key: int) -> int:
        return key % self.size
    def predict(self, instr: Instr) -> bool:
        idx = self._idx(instr.id)
        return self.table[idx] >= 2
    def update(self, instr: Instr, taken: bool):
        idx = self._idx(instr.id)
        if taken:
            if self.table[idx] < 3: self.table[idx] += 1
        else:
            if self.table[idx] > 0: self.table[idx] -= 1

class GSharePredictor:
    def __init__(self, history_bits: int = 8, table_size: int = 2048):
        self.hb = max(2, min(16, history_bits))
        self.ghr = 0
        self.size = max(64, table_size)
        self.table = [2] * self.size
    def _idx(self, instr: Instr) -> int:
        pc = instr.id
        mask = (1 << self.hb) - 1
        idx = (pc ^ (self.ghr & mask)) % self.size
        return idx
    def predict(self, instr: Instr) -> bool:
        idx = self._idx(instr)
        return self.table[idx] >= 2
    def update(self, instr: Instr, taken: bool):
        idx = self._idx(instr)
        if taken:
            if self.table[idx] < 3: self.table[idx] += 1
        else:
            if self.table[idx] > 0: self.table[idx] -= 1
        # update global history
        self.ghr = ((self.ghr << 1) | (1 if taken else 0)) & ((1 << self.hb) - 1)

# ----------------------------
# Physical regs + rename + checkpoint
# ----------------------------
class PhysicalRegisterFile:
    def __init__(self, n):
        self.n = n
        self.free_list = deque(range(n))
        self.busy = [False] * n
    def alloc(self) -> Optional[int]:
        if not self.free_list:
            return None
        r = self.free_list.popleft()
        self.busy[r] = True
        return r
    def free(self, r:int):
        if r is None: return
        if not self.busy[r]:
            # avoid duplicates
            if r not in self.free_list:
                self.free_list.append(r)
        else:
            self.busy[r] = False
            self.free_list.append(r)
    def mark_ready(self, r:int):
        if r is None: return
        self.busy[r] = False
    def is_busy(self, r:int) -> bool:
        if r is None: return False
        return self.busy[r]
    def snapshot(self) -> Tuple[List[bool], List[int]]:
        # return busy array copy + free_list as list
        return (list(self.busy), list(self.free_list))
    def restore(self, snap: Tuple[List[bool], List[int]]):
        busy_copy, free_list_copy = snap
        self.busy = list(busy_copy)
        self.free_list = deque(free_list_copy)

class RenameTable:
    def __init__(self, n_threads:int, arch_regs:List[str]):
        self.map = {t: {r: None for r in arch_regs} for t in range(n_threads)}
    def get(self, t:int, arch:str) -> Optional[int]:
        return self.map[t].get(arch, None)
    def set(self, t:int, arch:str, phys:Optional[int]):
        self.map[t][arch] = phys
    def snapshot(self):
        # deep copy
        return {t: dict(self.map[t]) for t in self.map}
    def restore(self, snap):
        self.map = {t: dict(snap[t]) for t in snap}

# ----------------------------
# ROB & RS
# ----------------------------
@dataclass
class ROBEntry:
    tag: int
    mu: MicroOp
    ready: bool = False
    committed: bool = False
    is_branch: bool = False

class ROB:
    def __init__(self, size:int):
        self.size = size
        self.buf: deque[ROBEntry] = deque()
        self.next_tag = 1
    def can_push(self)->bool:
        return len(self.buf) < self.size
    def push(self, mu:MicroOp) -> int:
        tag = self.next_tag; self.next_tag += 1
        e = ROBEntry(tag=tag, mu=mu, is_branch=mu.instr.is_branch)
        self.buf.append(e)
        return tag
    def mark_ready(self, tag:int):
        for e in self.buf:
            if e.tag == tag:
                e.ready = True
                return True
        return False
    def commit_ready(self)->Optional[ROBEntry]:
        if self.buf and self.buf[0].ready:
            return self.buf.popleft()
        return None
    def flush_until(self, tag:int):
        # flush everything after tag (simple flush)
        self.buf = deque([e for e in self.buf if e.tag < tag])
    def flush_all(self):
        self.buf.clear()
    def occupancy(self)->int:
        return len(self.buf)
    def snapshot(self) -> Tuple[List[int], int]:
        # minimal snapshot: current tags and next_tag
        tags = [e.tag for e in self.buf]
        return (tags, self.next_tag)
    def restore(self, snap):
        # expensive to restore full ROB; easier approach: flush all and reset next_tag
        self.flush_all()
        _, next_tag = snap
        self.next_tag = next_tag

@dataclass
class RSEntry:
    tag: int
    mu: MicroOp
    src_tags: List[Optional[int]]  # which ROB tag produces source (None = ready)
    latency_left: int
    fu_type: str

class ReservationStations:
    def __init__(self, size:int):
        self.size = size
        self.entries: List[RSEntry] = []
    def can_accept(self)->bool:
        return len(self.entries) < self.size
    def add(self, e:RSEntry):
        if not self.can_accept():
            raise RuntimeError("RS full")
        self.entries.append(e)
    def find_ready(self, max_issue:int):
        ready = [e for e in self.entries if all(t is None for t in e.src_tags)]
        ready.sort(key=lambda x: x.tag)
        return ready[:max_issue]
    def update_wakeup(self, completed_tag:int):
        for e in self.entries:
            e.src_tags = [None if t==completed_tag else t for t in e.src_tags]
    def retire_finished(self):
        finished = [e for e in self.entries if e.latency_left <= 0]
        self.entries = [e for e in self.entries if e.latency_left > 0]
        return finished
    def remove_by_tag(self, tag:int):
        self.entries = [e for e in self.entries if e.tag != tag]
    def occupancy(self)->int:
        return len(self.entries)
    def snapshot(self):
        # simple: store tags of entries (not full microops). We'll flush RS on mispredict and restore nothing (checkpointing handles rename/free-lists)
        return [e.tag for e in self.entries]

# ----------------------------
# Functional units
# ----------------------------
class FU:
    def __init__(self, n:int):
        self.total = n
        self.busy = 0
    def acquire(self, n=1) -> bool:
        if self.busy + n > self.total:
            return False
        self.busy += n
        return True
    def release(self, n=1):
        self.busy = max(0, self.busy - n)
    def avail(self)->int:
        return self.total - self.busy

# ----------------------------
# Scheduler
# ----------------------------
class Scheduler:
    def __init__(self, hw_threads:int, policy:str="round_robin"):
        self.hw_threads = hw_threads
        self.policy = policy
        self.rr_next = 0
    def order(self, runnable:List[int]) -> List[int]:
        if not runnable:
            return []
        if self.policy == "round_robin":
            # rotate list using rr_next index if runnable provided as sorted
            ordered = []
            for i in range(len(runnable)):
                idx = (self.rr_next + i) % len(runnable)
                ordered.append(runnable[idx])
            self.rr_next = (self.rr_next + 1) % len(runnable)
            return ordered
        return runnable

# ----------------------------
# Metrics & Events
# ----------------------------
class Metrics:
    def __init__(self, hw_threads:int):
        self.cycles = 0
        self.total_committed = 0
        self.per_thread_committed = [0]*hw_threads
        self.branch_mispredicts = 0
        self.stall_counts = defaultdict(int)
        self.ipc_history = []
        self.rob_history = []
        self.rs_history = []
        self.issued_history = []

class EventEmitter:
    def __init__(self, enable:bool=True):
        self.enable = enable
        self.events = []
    def emit(self, cycle:int, data:dict):
        if not self.enable: return
        evt = {"cycle":cycle, **data}
        self.events.append(evt)
    def dump(self, path:str):
        if not self.enable: return
        with open(path,"w") as f:
            json.dump(self.events, f, indent=2)

# ----------------------------
# Simulator core
# ----------------------------
class Simulator:
    def __init__(self, cfg:Config, workloads:Dict[int, List[Instr]]):
        self.cfg = cfg
        # predictor selection
        if cfg.predictor == "random":
            self.predictor = RandomPredictor(cfg.branch_accuracy_pct)
        elif cfg.predictor == "bimodal":
            self.predictor = BimodalPredictor(cfg.bimodal_size)
        elif cfg.predictor == "gshare":
            self.predictor = GSharePredictor(cfg.gshare_history, max(256, cfg.bimodal_size))
        else:
            self.predictor = RandomPredictor(cfg.branch_accuracy_pct)
        # regs/rename
        self.phys = PhysicalRegisterFile(cfg.phys_regs)
        arch_regs = [f"r{i}" for i in range(32)]
        self.rename = RenameTable(cfg.hw_threads, arch_regs)
        self.rob = ROB(cfg.rob_size)
        self.rs = ReservationStations(cfg.rs_size)
        self.alus = FU(cfg.num_alus)
        self.fpus = FU(cfg.num_fpus)
        self.scheduler = Scheduler(cfg.hw_threads)
        self.metrics = Metrics(cfg.hw_threads)
        self.eventer = EventEmitter(cfg.event_emit)
        # thread instruction queues
        self.queues = {t: deque(workloads.get(t, [])) for t in range(cfg.hw_threads)}
        # bookkeeping: map phys reg -> producing ROB tag (to determine src_tags)
        # We'll maintain a dict phys_producer[phys] = tag when we allocate dests and push to ROB
        self.phys_producer: Dict[int, int] = {}
        # checkpoint stack for speculative branches (list of tuples: rename_snap, phys_snap, rob_snap)
        self.checkpoints: List[Tuple[Any,Any,Any]] = []
        self.instr_counter = 1

    def decode_rename(self, instr:Instr) -> Optional[MicroOp]:
        mu_srcs = []
        for s in instr.srcs:
            p = self.rename.get(instr.thread_id, s)
            mu_srcs.append(p)
        dst_p = None
        if instr.dst:
            dst_p = self.phys.alloc()
            if dst_p is None:
                return None  # stall, no phys regs
            self.rename.set(instr.thread_id, instr.dst, dst_p)
        mu = MicroOp(instr=instr, src_phys=mu_srcs, dst_phys=dst_p, exec_remaining=instr.latency)
        # decide FU type
        op = instr.opcode.upper()
        if op.startswith("F") or "MULF" in op:
            mu.fu_type = "FPU"
        elif instr.is_mem:
            mu.fu_type = "MEM"
        else:
            mu.fu_type = "ALU"
        return mu

    def issue_stage(self):
        issued = []
        cfg = self.cfg
        # in-flight limit: check ROB+RS occupancy vs instr_window
        inflight = self.rob.occupancy() + self.rs.occupancy()
        if inflight >= cfg.instr_window:
            self.metrics.stall_counts["instr_window_full"] += 1
            return issued
        runnable = [t for t,q in self.queues.items() if q]
        if not runnable: return issued
        order = self.scheduler.order(sorted(runnable))
        remaining_slots = cfg.issue_width
        for t in order:
            if remaining_slots <= 0: break
            while remaining_slots > 0 and self.queues[t]:
                if not self.rob.can_push():
                    self.metrics.stall_counts["rob_full"] += 1
                    return issued
                if not self.rs.can_accept():
                    self.metrics.stall_counts["rs_full"] += 1
                    return issued
                instr = self.queues[t][0]
                # predictor checkpoint if branch
                if instr.is_branch:
                    # create checkpoint: rename table snapshot + phys snapshot + rob snapshot
                    rename_snap = self.rename.snapshot()
                    phys_snap = self.phys.snapshot()
                    rob_snap = self.rob.snapshot()
                    self.checkpoints.append((rename_snap, phys_snap, rob_snap))
                    # predictor returns predicted taken/not; we don't model target PC but record for stats
                    _ = self.predictor.predict(instr)
                mu = self.decode_rename(instr)
                if mu is None:
                    self.metrics.stall_counts["no_free_phys"] += 1
                    return issued
                tag = self.rob.push(mu)
                # map dst phys -> producer tag for wakeups
                if mu.dst_phys is not None:
                    self.phys_producer[mu.dst_phys] = tag
                # build src_tags list: find producing tags for each src phys
                src_tags = []
                for p in mu.src_phys:
                    if p is None:
                        src_tags.append(None)
                    else:
                        src_tags.append(self.phys_producer.get(p, None))
                rs_entry = RSEntry(tag=tag, mu=mu, src_tags=src_tags, latency_left=mu.exec_remaining, fu_type=mu.fu_type)
                self.rs.add(rs_entry)
                self.queues[t].popleft()
                issued.append((t, instr.opcode, tag))
                remaining_slots -= 1
                if remaining_slots <= 0:
                    break
        return issued

    def execute_stage(self):
        # pick ready RS entries upto issue_width
        cfg = self.cfg
        candidates = self.rs.find_ready(cfg.issue_width)
        for e in candidates:
            pool = self.alus if e.fu_type == "ALU" else (self.fpus if e.fu_type=="FPU" else self.alus)
            if not pool.acquire(1):
                self.metrics.stall_counts["fu_busy"] += 1
                continue
            # occupy for one cycle: decrement latency_left
            e.latency_left -= 1
            pool.release(1)

    def writeback_wakeup(self):
        finished = self.rs.retire_finished()
        for e in finished:
            self.rob.mark_ready(e.tag)
            # mark dest phys ready & remove producer mapping
            dst = e.mu.dst_phys
            if dst is not None:
                self.phys.mark_ready(dst)
                self.phys_producer.pop(dst, None)
            # wakeup other RS entries depending on this tag
            self.rs.update_wakeup(e.tag)

    def commit_stage(self):
        committed = []
        while True:
            e = self.rob.commit_ready()
            if not e: break
            committed.append(e)
            mu = e.mu
            self.metrics.total_committed += 1
            tid = mu.instr.thread_id
            self.metrics.per_thread_committed[tid] += 1
            # on commit, free nothing immediate for simplicity (we already freed phys at writeback by marking ready)
        return committed

    def handle_branch_mispredict(self, committed_entries:List[ROBEntry]):
        # check for branch commits and evaluate prediction correctness
        for e in committed_entries:
            if e.is_branch:
                # decide actual correctness stochastically / by predictor type (predictor classes handle update)
                taken = random.choice([True, False])  # actual outcome random for now
                # predictor update
                self.predictor.update(e.mu.instr, taken)
                predicted = self.predictor.predict(e.mu.instr)  # what we predicted earlier (approx)
                if predicted != taken:
                    # mispredict => restore last checkpoint if available
                    if self.checkpoints:
                        rename_snap, phys_snap, rob_snap = self.checkpoints.pop()
                        self.rename.restore(rename_snap)
                        self.phys.restore(phys_snap)
                        self.rob.restore(rob_snap)
                        # flush RS (we had speculative RS entries)
                        self.rs.entries.clear()
                    else:
                        # fallback: full flush
                        self.rob.flush_all()
                        self.rs.entries.clear()
                    self.metrics.branch_mispredicts += 1
                    self.metrics.stall_counts["mispredict_recovery"] += 1
                    # clear producer map (safe)
                    self.phys_producer.clear()
                    break
                else:
                    # correct: drop the checkpoint corresponding to this branch (if any)
                    if self.checkpoints:
                        try:
                            self.checkpoints.pop()
                        except:
                            pass

    def emit_cycle(self, cycle:int, issued, committed_count):
        event = {
            "issued": issued,
            "committed_count": committed_count,
            "rob_occupancy": self.rob.occupancy(),
            "rs_occupancy": self.rs.occupancy(),
            "stall_counts": dict(self.metrics.stall_counts),
            "branch_mispredicts": self.metrics.branch_mispredicts
        }
        self.eventer.emit(cycle, event)

    def run(self):
        cfg = self.cfg
        cycle = 0
        while True:
            cycle += 1
            self.metrics.cycles = cycle
            issued = self.issue_stage()
            self.execute_stage()
            self.writeback_wakeup()
            committed_entries = self.commit_stage()
            self.handle_branch_mispredict(committed_entries)
            committed_this_cycle = len(committed_entries)
            # metrics bookkeeping
            self.metrics.ipc_history.append(committed_this_cycle)
            self.metrics.rob_history.append(self.rob.occupancy())
            self.metrics.rs_history.append(self.rs.occupancy())
            self.metrics.issued_history.append(len(issued))
            if cfg.event_emit:
                self.emit_cycle(cycle, issued, committed_this_cycle)
            # termination
            pending = any(len(q)>0 for q in self.queues.values())
            inflight = (self.rob.occupancy() > 0) or (self.rs.occupancy() > 0)
            if not pending and not inflight:
                break
            if cycle >= cfg.max_cycles:
                break
        if cfg.event_emit:
            self.eventer.dump("archviz_events.json")
        return self.metrics

# ----------------------------
# Workload helpers
# ----------------------------
_instr_id = 1
def mk_instr(op, srcs=None, dst=None, latency=1, tid=0, is_branch=False, is_mem=False):
    global _instr_id
    ins = Instr(id=_instr_id, opcode=op, srcs=srcs or [], dst=dst, latency=latency, thread_id=tid, is_branch=is_branch, is_mem=is_mem)
    _instr_id += 1
    return ins

def workload_compute(n, tid=0):
    return [mk_instr("ADD", srcs=["r1","r2"], dst="r3", latency=1, tid=tid) for _ in range(n)]

def workload_branchy(n, tid=0):
    return [mk_instr("BR", srcs=[], dst=None, latency=1, tid=tid, is_branch=True) for _ in range(n)]

def workload_mixed(n, tid=0):
    lst=[]
    for i in range(n):
        if i%5==0:
            lst.append(mk_instr("LOAD", srcs=["r4"], dst="r5", latency=2, tid=tid, is_mem=True))
        elif i%3==0:
            lst.append(mk_instr("MUL", srcs=["r6","r7"], dst="r8", latency=3, tid=tid))
        else:
            lst.append(mk_instr("ADD", srcs=["r1","r2"], dst="r3", latency=1, tid=tid))
    return lst

def build_sample(hw_threads:int):
    w={}
    for t in range(hw_threads):
        if t==0:
            w[t]=workload_mixed(30, tid=t)
        else:
            w[t]=workload_branchy(20, tid=t)
    return w

# ----------------------------
# CLI and main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--issue-width", type=int, default=2)
    p.add_argument("--instr-window", type=int, default=32)
    p.add_argument("--alus", type=int, default=2)
    p.add_argument("--fpu", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--rob", type=int, default=48)
    p.add_argument("--rs", type=int, default=32)
    p.add_argument("--phys", type=int, default=128)
    p.add_argument("--max-cycles", type=int, default=5000)
    p.add_argument("--predictor", choices=["random","bimodal","gshare"], default="random")
    p.add_argument("--branch-acc", type=float, default=90.0)
    p.add_argument("--bimodal-size", type=int, default=1024)
    p.add_argument("--gshare-history", type=int, default=8)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(issue_width=args.issue_width,
                 instr_window=args.instr_window,
                 num_alus=args.alus,
                 num_fpus=args.fpu,
                 branch_accuracy_pct=args.branch_acc,
                 hw_threads=args.threads,
                 rob_size=args.rob,
                 rs_size=args.rs,
                 phys_regs=args.phys,
                 max_cycles=args.max_cycles,
                 predictor=args.predictor,
                 bimodal_size=args.bimodal_size,
                 gshare_history=args.gshare_history,
                 verbose=args.verbose)
    workloads = build_sample(cfg.hw_threads)
    sim = Simulator(cfg, workloads)
    metrics = sim.run()

    # summary
    print("\n--- Simulation Summary ---")
    print(f"Cycles: {metrics.cycles}")
    print(f"Total committed: {metrics.total_committed}")
    print(f"Per-thread committed: {metrics.per_thread_committed}")
    print(f"Branch mispredicts: {metrics.branch_mispredicts}")
    print("Stall counts:", dict(metrics.stall_counts))
    ipc = metrics.total_committed / metrics.cycles if metrics.cycles>0 else 0.0
    print(f"Overall IPC: {ipc:.3f}")
    for t,c in enumerate(metrics.per_thread_committed):
        print(f"Thread {t} IPC approx: {c/metrics.cycles:.3f}")
    # write summary CSV
    try:
        import csv
        with open("archviz_summary.csv","w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["cycles","total_committed","branch_mispredicts","ipc"])
            w.writerow([metrics.cycles, metrics.total_committed, metrics.branch_mispredicts, ipc])
        print("Wrote archviz_summary.csv and archviz_events.json")
    except Exception as e:
        print("Could not write CSV:", e)

if __name__=="__main__":
    main()
