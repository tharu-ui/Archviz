import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ---- Sidebar: Config Inputs ----
st.sidebar.title("ARCHVIZ Configuration")

issue_width = st.sidebar.slider("Issue Width", 1, 8, 4)
threads = st.sidebar.selectbox("Number of Threads", [1, 2, 4])
rob_size = st.sidebar.slider("ROB Size", 16, 128, 64, step=16)
rs_size = st.sidebar.slider("Reservation Stations", 8, 64, 48, step=8)
instr_window = st.sidebar.slider("Instruction Window", 10, 100, 40, step=5)
branch_predictor = st.sidebar.selectbox("Branch Predictor", ["static", "gshare", "2-level"])
history_bits = st.sidebar.slider("Gshare History Bits", 2, 16, 10)

# ---- Run Simulation Button ----
if st.button("Run Simulation"):
    # Dummy simulation results (replace with your backend later)
    cycles = np.arange(1, 11)
    ipc = np.random.uniform(0.5, issue_width, 10)
    branch_acc = np.random.uniform(60, 95)
    cpi = 1 / np.mean(ipc)

    st.subheader("Simulation Results")
    st.write(f"**Average IPC:** {np.mean(ipc):.2f}")
    st.write(f"**CPI:** {cpi:.2f}")
    st.write(f"**Branch Prediction Accuracy:** {branch_acc:.1f}%")

    # ---- Graphs ----
    st.subheader("IPC Over Time")
    fig, ax = plt.subplots()
    ax.plot(cycles, ipc, marker='o')
    ax.set_xlabel("Cycle")
    ax.set_ylabel("IPC")
    ax.set_title("IPC vs Cycles")
    st.pyplot(fig)

    st.subheader("Pipeline Utilization Example")
    fig2, ax2 = plt.subplots()
    threads_labels = [f"Thread {i+1}" for i in range(threads)]
    utilization = np.random.uniform(40, 95, threads)
    ax2.bar(threads_labels, utilization)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Utilization (%)")
    st.pyplot(fig2)

else:
    st.info("Configure parameters on the left, then click **Run Simulation**")
