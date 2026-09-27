"""Step 10 demo app. Two deviations from the guide's reference:

1. Agents are discovered by scanning runs/<agent>/seed*_lam*/best.pt for
   whichever checkpoints actually exist, rather than a hardcoded
   ("dqn", "ppo", "a2c") tuple -- today that's DQN only; PPO and A2C
   appear in the sidebar automatically the moment their checkpoints land,
   no code change needed.
2. The action-score bar chart explicitly greys out illegal actions
   (matplotlib, not st.bar_chart, which just omits NaN bars rather than
   showing them as disabled) so an examiner sees *why* an action wasn't
   taken, not just that it wasn't.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from zsrl import get_device
from zsrl.corruptions import CORRUPTIONS, make
from zsrl.dataset import load_canvases
from zsrl.encoder import FrozenEncoder
from zsrl.env import ACTION_NAMES, N_ACTIONS, ZoomSearchEnv, node_box
from scripts.train import build_agent

st.set_page_config(page_title="Zoom Search Agent", layout="wide")

ORACLE_COST, HEURISTIC_COST, EXHAUSTIVE_COST = 3, 12, 64


def discover_checkpoints():
    """{agent_name: checkpoint_path} for every runs/<agent>/seed*_lam*/best.pt
    found on disk, one per agent (first match, sorted). Returns {} entries
    are skipped entirely, not shown as a broken option in the sidebar."""
    found = {}
    for path in sorted(glob.glob(os.path.join("runs", "*", "seed*_lam*", "best.pt"))):
        agent_name = path.split(os.sep)[-3]
        if agent_name not in found:
            found[agent_name] = path
    return found


@st.cache_resource
def load():
    device = get_device()
    encoder = FrozenEncoder(device)
    canvases = load_canvases(split="test")  # full test split if present, else data/demo/
    dim = ZoomSearchEnv(canvases, encoder).state_dim
    checkpoints = discover_checkpoints()
    agents = {}
    for name, path in checkpoints.items():
        try:
            a = build_agent(name, dim, device, 0)
            a.load(path)
            agents[name] = a
        except Exception as e:
            st.sidebar.warning(f"{name}: found {path} but failed to load ({e})")
    return device, encoder, canvases, agents


device, encoder, canvases, agents = load()

st.title("Compute-Aware Zoom Search")
st.caption("The agent chooses where to spend full resolution.")

if not agents:
    st.error(
        "No trained checkpoints found under runs/<agent>/seed*_lam*/best.pt. "
        "The demo has nothing to load -- train at least one agent first."
    )
    st.stop()

with st.sidebar:
    st.caption(f"Agents with a checkpoint on disk: {', '.join(agents)}")
    agent_name = st.selectbox("Agent", list(agents.keys()))
    idx = st.number_input("Test image index", 0, len(canvases) - 1, 0)
    corruption = st.selectbox("Condition", ["clean"] + list(CORRUPTIONS))
    severity = st.slider("Severity", 1, 3, 1, disabled=(corruption == "clean"))
    step_once = st.button("Step once")
    run_all = st.button("Run episode")
    reset = st.button("Reset")

fn, tag = make(corruption, severity) if corruption != "clean" else (None, "clean")
key = (idx, tag, agent_name)

if reset or "env" not in st.session_state or st.session_state.get("key") != key:
    env = ZoomSearchEnv(canvases, encoder, corruption=fn, corruption_tag=tag)
    st.session_state.state = env.reset(index=int(idx))
    st.session_state.env = env
    st.session_state.key = key
    st.session_state.total = 0.0
    st.session_state.log = []
    st.session_state.done = False

env = st.session_state.env
agent = agents[agent_name]


def advance():
    if st.session_state.done:
        return
    s = st.session_state.state
    mask = env.legal()
    q = agent.q_values(s, mask)
    a = int(np.argmax(np.where(np.isfinite(q), q, -1e9)))
    s2, r, done, info = env.step(a)
    st.session_state.state = s2
    st.session_state.total += r
    st.session_state.done = done
    st.session_state.log.append({
        "step": info["steps"], "action": ACTION_NAMES[a],
        "depth": info["depth"], "reward": round(r, 2),
        "return": round(st.session_state.total, 2)})


if step_once:
    advance()
if run_all:
    for _ in range(env.budget):
        advance()
        if st.session_state.done:
            break

left, right = st.columns([3, 2])

with left:
    img = env.image.copy()
    d = ImageDraw.Draw(img)
    for n in env.visited[:-1]:
        d.rectangle(node_box(n), outline=(120, 120, 255), width=2)
    d.rectangle(env.target, outline=(0, 255, 0), width=4)
    d.rectangle(node_box(env.node), outline=(255, 40, 40), width=5)
    st.image(img, caption="green = target, red = current region, "
                          "blue = regions already examined",
             use_container_width=True)

with right:
    x1, y1, x2, y2 = node_box(env.node)
    st.image(env.image.crop((x1, y1, x2, y2)).resize((280, 280)),
             caption=f"what the agent sees now (depth {env.node[0]})")

    c1, c2, c3 = st.columns(3)
    c1.metric("Depth", env.node[0])
    c2.metric("Evaluations used", env.t)
    c3.metric("Return", f"{st.session_state.total:.2f}")

    mask = env.legal()
    q = agent.q_values(st.session_state.state, mask)
    q = np.where(np.isfinite(q), q, 0.0)

    fig, ax = plt.subplots(figsize=(5, 2.6))
    colors = ["steelblue" if legal else "lightgrey" for legal in mask]
    bars = ax.bar(ACTION_NAMES, q, color=colors)
    for bar, legal in zip(bars, mask):
        if not legal:
            bar.set_hatch("//")
    ax.set_ylabel("score")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Action scores. Grey/hatched bars are illegal at this depth "
               "(masked out, never selectable).")

if st.session_state.log:
    st.subheader("Trace")
    st.dataframe(st.session_state.log, use_container_width=True)

st.divider()
st.subheader("Cost comparison for this image")
st.write({
    "agent so far": env.t,
    "oracle": ORACLE_COST,
    "heuristic (odd_one_out)": HEURISTIC_COST,
    "exhaustive to depth 3": EXHAUSTIVE_COST,
})
