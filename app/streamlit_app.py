"""Demo app, v2.

Covers the assessment guide's recommended live-demo sequence end to end:

  1 start the environment          -> sidebar: agent, image, condition
  2 show the current observation   -> left pane + "what the agent sees now"
  3 show available + chosen action -> masked action-score chart
  4 execute and show the response  -> Step once / Run episode
  5 show reward and next state     -> trace table, metrics row
  6 show the learning/update step  -> "Learning update" panel (new in v2)
  7 repeat / show the trained policy
  8 learning curve + final metrics -> "Training evidence" panel (new in v2)

Two deviations from the guide's reference, both deliberate:

1. Agents are discovered by scanning runs/<agent>/seed*_lam*/best.pt rather
   than a hardcoded tuple, so PPO and A2C appear the moment their checkpoints
   land without a code change.
2. The action-score chart greys out illegal actions explicitly (matplotlib,
   not st.bar_chart, which omits NaN bars rather than showing them disabled)
   so an examiner sees *why* an action was not taken, not just that it wasn't.

Compute is reported in 224-equivalent units, never raw action counts: the
depth-0 node is encoded at 448 and costs 4, deeper nodes cost 1. See
zsrl/env.py:units.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw

from zsrl import MIN_COMMIT_DEPTH, get_device
from zsrl.corruptions import CORRUPTIONS, make
from zsrl.dataset import load_canvases
from zsrl.env import ACTION_NAMES, ZoomSearchEnv, node_box, units
from scripts.train import build_agent

st.set_page_config(page_title="Zoom Search Agent", layout="wide")

# Costs in 224-equivalent units. The oracle still needs to OBSERVE each node
# it passes through: root(4) + depth1(1) + depth2(1) + depth3(1) = 7.
ORACLE_UNITS = units(0) + MIN_COMMIT_DEPTH
HEURISTIC_UNITS = 12
EXHAUSTIVE_UNITS = 64


def discover_checkpoints():
    found = {}
    for path in sorted(glob.glob(os.path.join("runs", "*", "seed*_lam*", "best.pt"))):
        agent_name = path.split(os.sep)[-3]
        if agent_name not in found:
            found[agent_name] = path
    return found


@st.cache_resource
def load():
    device = get_device()
    from zsrl.encoder import FrozenEncoder
    encoder = FrozenEncoder(device)
    canvases = load_canvases(split="test")      # full split if present, else data/demo/
    dim = ZoomSearchEnv(canvases, encoder).state_dim
    agents, broken = {}, {}
    for name, path in discover_checkpoints().items():
        try:
            a = build_agent(name, dim, device, 0)
            a.load(path)
            agents[name] = a
        except Exception as e:
            broken[name] = (path, str(e))
    return device, encoder, canvases, agents, broken, dim


device, encoder, canvases, agents, broken, STATE_DIM = load()

st.title("Compute-Aware Zoom Search")
st.caption("The agent chooses where to spend full resolution.")

for name, (path, err) in broken.items():
    st.sidebar.error(
        f"**{name}**: found `{path}` but could not load it.\n\n"
        f"This is usually a v1 checkpoint (6 actions, 2082-d state) against the "
        f"v2 environment (5 actions, {STATE_DIM}-d state). Move the old `runs/` "
        f"aside and use checkpoints trained at tag `env-frozen-v2`.\n\n`{err[:160]}`")

if not agents:
    st.error(
        f"No loadable checkpoints under `runs/<agent>/seed*_lam*/best.pt`.\n\n"
        f"This environment is v2: {len(ACTION_NAMES)} actions, {STATE_DIM}-d state. "
        f"Checkpoints trained before tag `env-frozen-v2` will not load.")
    st.stop()

with st.sidebar:
    st.caption(f"Loaded: {', '.join(agents)}")
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
    st.session_state.last = None

env = st.session_state.env
agent = agents[agent_name]


def advance():
    """One environment step, recording everything the demo needs to show."""
    if st.session_state.done:
        return
    s = st.session_state.state
    mask = env.legal()
    q = agent.q_values(s, mask)
    a = int(np.argmax(np.where(np.isfinite(q), q, -1e9)))
    s2, r, done, info = env.step(a)
    next_mask = env.legal()

    st.session_state.last = {"s": s, "a": a, "r": r, "s2": s2,
                             "done": done, "next_mask": next_mask,
                             "q_sa": float(q[a])}
    st.session_state.state = s2
    st.session_state.total += r
    st.session_state.done = done
    st.session_state.log.append({
        "step": info["steps"], "action": ACTION_NAMES[a],
        "depth": info["depth"], "units spent": info["units"],
        "reward": round(r, 2), "return": round(st.session_state.total, 2)})


if step_once:
    advance()
if run_all:
    for _ in range(len(ACTION_NAMES) + 1):
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
             caption=f"what the agent sees now (depth {env.node[0]}, "
                     f"encoded at {448 if env.node[0] == 0 else 224}px)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Depth", env.node[0])
    c2.metric("Compute spent", f"{env.spent} units")
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
    ax.set_ylabel("Q(s, a)")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Action values. Grey hatched bars are masked out — illegal at "
               "this depth, or unaffordable within the remaining budget.")

# ---- demo step 6: the learning update, for the transition just taken -------
if st.session_state.last is not None and hasattr(agent, "online"):
    st.divider()
    st.subheader("Learning update for the last transition")
    L = st.session_state.last
    with torch.no_grad():
        s2 = torch.from_numpy(L["s2"]).float().unsqueeze(0).to(device)
        m2 = torch.from_numpy(L["next_mask"]).bool().unsqueeze(0).to(device)
        NEG = torch.finfo(torch.float32).min
        # Double DQN: the ONLINE net picks the next action, the TARGET net
        # values it, and both are restricted to legal actions.
        a_star = agent.online(s2).masked_fill(~m2, NEG).argmax(1, keepdim=True)
        q_next = agent.target(s2).masked_fill(~m2, NEG).gather(1, a_star).squeeze(1)
        q_next = torch.nan_to_num(q_next, neginf=0.0)
        y = L["r"] + agent.gamma * (1.0 - float(L["done"])) * float(q_next)
    delta = y - L["q_sa"]
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Q(s, a)", f"{L['q_sa']:.3f}")
    e2.metric("reward r", f"{L['r']:.2f}")
    e3.metric("target y", f"{y:.3f}")
    e4.metric("TD error", f"{delta:+.3f}")
    st.caption(
        f"a* = argmax over legal a' of Q_online(s', a') = "
        f"**{ACTION_NAMES[int(a_star.item())] if not L['done'] else 'terminal'}**  ·  "
        f"y = r + γ(1−d)·Q_target(s', a*), γ = {agent.gamma}  ·  "
        f"δ = y − Q(s, a). Huber loss on δ is what the optimiser minimises. "
        f"Masking is applied to BOTH networks, so an illegal action can never "
        f"set the target.")

if st.session_state.log:
    st.subheader("Trace")
    st.dataframe(st.session_state.log, use_container_width=True)

st.divider()
st.subheader(f"Compute for this image (224-equivalent units)")
st.write({"agent so far": env.spent,
          "oracle (knows the answer)": ORACLE_UNITS,
          "odd-one-out heuristic": HEURISTIC_UNITS,
          "exhaustive scan to depth 3": EXHAUSTIVE_UNITS})
st.caption("One unit = one ResNet-18 forward pass on a 224x224 region. The "
           "depth-0 node is encoded at 448 and therefore costs 4 units, "
           "because a 448 pass is 4x the FLOPs.")

# ---- demo step 8: training evidence ---------------------------------------
curve = sorted(glob.glob(os.path.join("runs", agent_name, "seed*_lam*", "train_log.csv")))
if curve:
    st.divider()
    st.subheader("Training evidence")
    frames = []
    for p in curve:
        try:
            d = pd.read_csv(p)
            d["seed"] = p.split(os.sep)[-2].split("_")[0]
            frames.append(d)
        except Exception:
            pass
    if frames:
        T = pd.concat(frames, ignore_index=True)
        T["smooth"] = T.groupby("seed")["return"].transform(
            lambda x: x.rolling(500, min_periods=50).mean())
        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            for s, d in T.groupby("seed"):
                ax.plot(d.episode, d.smooth, lw=1.4, label=s)
            ax.set_xlabel("episode"); ax.set_ylabel("return (500-ep mean)")
            ax.legend(fontsize=7, frameon=False); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
        with g2:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            for s, d in T.groupby("seed"):
                ax.plot(d.episode, d.explore, lw=1.4, label=s)
            ax.set_xlabel("episode"); ax.set_ylabel("exploration (epsilon)")
            ax.legend(fontsize=7, frameon=False); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
        last = T.groupby("seed").tail(2000)
        m1, m2, m3 = st.columns(3)
        m1.metric("seeds", T.seed.nunique())
        m2.metric("episodes per seed", int(T.groupby("seed").size().max()))
        m3.metric("final success rate (last 2k eps)", f"{last.success.mean():.3f}")
