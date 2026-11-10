"""Compute every evaluation metric in the course's RL Evaluation Metrics notes
from data already on disk, and emit the tables and plots for submission.

Inputs  : results/eval_<agent>_seed<N>_lam0.1.csv   (13 files, 3 agents)
          runs/<agent>/seed<N>_lam0.1/train_log.csv (5 DQN seeds, 500-ep grid)
Outputs : results/metrics/*.csv  and  results/metrics/*.png

Nothing here re-runs the agent. Metrics that need per-episode rollouts
(state coverage, visitation entropy, TD error) are produced by
scripts/record_rollouts.py instead, and are listed as NOT_AVAILABLE here
rather than silently omitted.

Run:  python3 scripts/eval_metrics.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/metrics"
os.makedirs(OUT, exist_ok=True)

# Categorical palette, validated colourblind-safe on all pairs
# (OKLab dE 9.2 deutan / 24.0 normal vision against a light surface).
C = {"dqn": "#2a78d6", "ppo": "#eb6834", "a2c": "#1baf7a"}
GREY = "#52514e"
FAINT = "#c9c7c0"
AGENTS = ["dqn", "ppo", "a2c"]
LABEL = {"dqn": "Double DQN", "ppo": "PPO", "a2c": "A2C"}

# Environment constants, from zsrl/env.py and zsrl/__init__.py.
ORACLE_RETURN = 3 * 0.9 + 3.0   # 3 correct first-visit descents, then commit
ORACLE_COST = 3
BUDGET = 12
N_EVAL = 200                    # scripts/evaluate.py run_full_evaluation(n=200)
N_NODES = sum(4 ** d for d in range(5))   # 341 nodes, depths 0..4

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8f94", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": GREY, "ytick.color": GREY,
    "grid.color": "#e8e6e0", "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.facecolor": "white",
})


def finish(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, loc="left", fontsize=10, pad=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path)


def ci95(x):
    """Half-width of the 95% CI for the mean. t-critical for small samples,
    as the notes require (section 5.2)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 1.96)
    return float(tcrit * x.std(ddof=1) / np.sqrt(n))


# ----------------------------------------------------------------- load
ev = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/eval_*_lam0.1.csv"))],
               ignore_index=True)
logs = {}
for f in sorted(glob.glob("runs/*/seed*_lam*/train_log.csv")):
    parts = f.split(os.sep)
    agent, seed = parts[1], int(parts[2].split("_")[0].replace("seed", ""))
    d = pd.read_csv(f)
    d["agent"], d["seed"] = agent, seed
    logs[(agent, seed)] = d
train = pd.concat(logs.values(), ignore_index=True) if logs else pd.DataFrame()

clean = ev[ev.corruption == "clean"]
agent_clean = clean[clean.method.isin(AGENTS)]
print(f"loaded {ev.shape[0]} evaluation rows, {len(logs)} training logs")


# ------------------------------------------- 1. headline performance table
rows = []
for (m, b), g in agent_clean.groupby(["method", "budget"]):
    rows.append({
        "method": m, "budget": int(b), "seeds": len(g),
        "success_rate": g.recall.mean(), "success_std": g.recall.std(ddof=1),
        "success_ci95": ci95(g.recall.values),
        "per_level_acc": g.per_level_accuracy.mean(),
        "mean_episode_length": g.cost.mean(), "episode_length_std": g.cost.std(ddof=1),
    })
t1 = pd.DataFrame(rows).sort_values(["method", "budget"])
t1.round(4).to_csv(f"{OUT}/table1_performance_by_budget.csv", index=False)

bl = clean[~clean.method.isin(AGENTS)].groupby("method").agg(
    success_rate=("recall", "mean"), per_level_acc=("per_level_accuracy", "mean"),
    mean_episode_length=("cost", "mean")).reset_index()
bl["seeds"] = clean[~clean.method.isin(AGENTS)].groupby("method").size().values
bl.round(4).to_csv(f"{OUT}/table2_baselines.csv", index=False)


# ----------------------------------------------------- 2. regret table
rows = []
for m in AGENTS:
    g = agent_clean[(agent_clean.method == m) & (agent_clean.budget == BUDGET)]
    if g.empty:
        continue
    sr = g.recall.mean()
    rows.append({
        "method": m, "budget": BUDGET,
        "oracle_success_rate": 1.0, "policy_success_rate": sr,
        "per_episode_task_regret": 1.0 - sr,
        "cumulative_task_regret_200ep": (1.0 - sr) * N_EVAL,
        "oracle_cost": ORACLE_COST, "policy_cost": g.cost.mean(),
        "excess_cost_per_episode": g.cost.mean() - ORACLE_COST,
    })
t3 = pd.DataFrame(rows)
t3.round(4).to_csv(f"{OUT}/table3_regret.csv", index=False)


# ------------------------------------------------- 3. robustness table
rob = ev[(ev.corruption != "clean") & (ev.method.isin(AGENTS))]
rows = []
for (m, c, s), g in rob.groupby(["method", "corruption", "severity"]):
    base = agent_clean[(agent_clean.method == m) & (agent_clean.budget == BUDGET)].recall.mean()
    rows.append({"method": m, "corruption": c, "severity": int(s), "seeds": len(g),
                 "success_rate": g.recall.mean(),
                 "nominal_success_rate": base,
                 "robustness_gap": base - g.recall.mean(),
                 "retention_pct": 100.0 * g.recall.mean() / base if base else np.nan})
t4 = pd.DataFrame(rows).sort_values(["method", "corruption", "severity"])
t4.round(4).to_csv(f"{OUT}/table4_robustness.csv", index=False)


# --------------------------------------- 4. learning / convergence table
rows = []
if not train.empty:
    for (a, s), d in logs.items():
        d = d.sort_values("episode")
        final = d.eval_recall.iloc[-1]
        best = d.eval_recall.max()
        # "Convergence episode": first episode after which eval success rate
        # never again improves by more than 0.02 over its value there.
        conv = np.nan
        for i in range(len(d)):
            if (d.eval_recall.iloc[i:].max() - d.eval_recall.iloc[i]) <= 0.02:
                conv = int(d.episode.iloc[i]); break
        rows.append({"agent": a, "seed": s, "episodes_logged": int(d.episode.iloc[-1]),
                     "initial_return": d["return"].iloc[0], "final_return": d["return"].iloc[-1],
                     "initial_success": d.eval_recall.iloc[0], "final_success": final,
                     "best_success": best, "converged_at_episode": conv,
                     "final_episode_length": d.steps.iloc[-1],
                     "final_commit_rate": d.committed.iloc[-1]})
t5 = pd.DataFrame(rows)
t5.round(4).to_csv(f"{OUT}/table5_learning.csv", index=False)


# ------------------------------------------- 5. complexity / efficiency
STATE_DIM, H1, H2, NA = 2082, 512, 256, 6
qnet_params = STATE_DIM * H1 + H1 + H1 * H2 + H2 + H2 * NA + NA
eff = pd.DataFrame([
    {"metric": "Q-network parameters", "value": qnet_params, "unit": "params",
     "source": "derived from QNetwork(2082->512->256->6)"},
    {"metric": "Online + target parameters", "value": 2 * qnet_params, "unit": "params",
     "source": "Double DQN keeps two copies"},
    {"metric": "Model size on disk", "value": os.path.getsize("runs/dqn/seed2_lam0.1/best.pt")
     if os.path.exists("runs/dqn/seed2_lam0.1/best.pt") else np.nan,
     "unit": "bytes", "source": "runs/dqn/seed2_lam0.1/best.pt"},
    {"metric": "Frozen encoder parameters", "value": 11_689_512, "unit": "params",
     "source": "ResNet-18, not trained"},
    {"metric": "Replay buffer capacity", "value": 100_000, "unit": "transitions",
     "source": "DQNAgent default"},
    {"metric": "Replay buffer memory", "value": 100_000 * (2 * STATE_DIM + 2 + NA) * 4,
     "unit": "bytes", "source": "2x state + a,r,d + next-mask, float32"},
    {"metric": "Training episodes per seed", "value": 12_000, "unit": "episodes",
     "source": "train.py --episodes default"},
    {"metric": "Environment steps per seed", "value": None, "unit": "steps",
     "source": "mean episode length x 12000, filled below"},
    {"metric": "Gradient updates per seed", "value": None, "unit": "updates",
     "source": "env steps / train_every(=2), after learn_start=1000"},
    {"metric": "Transitions processed per seed", "value": None, "unit": "transitions",
     "source": "gradient updates x batch_size(=64)"},
    {"metric": "Wall-clock training time", "value": np.nan, "unit": "minutes",
     "source": "NOT_AVAILABLE - the wall column in train_log.csv is empty"},
    {"metric": "Inference latency", "value": np.nan, "unit": "ms/action",
     "source": "NOT_AVAILABLE - measure with scripts/record_rollouts.py"},
])
if not train.empty:
    mean_len = train[train.agent == "dqn"].steps.mean()
    env_steps = mean_len * 12_000
    updates = max(0.0, (env_steps - 1000) / 2)
    eff.loc[eff.metric == "Environment steps per seed", "value"] = round(env_steps)
    eff.loc[eff.metric == "Gradient updates per seed", "value"] = round(updates)
    eff.loc[eff.metric == "Transitions processed per seed", "value"] = round(updates * 64)
eff.to_csv(f"{OUT}/table6_complexity.csv", index=False)


# -------------------------------------------------- 6. sample efficiency
rows = []
for target in (0.10, 0.15, 0.20):
    for (a, s), d in logs.items():
        d = d.sort_values("episode")
        hit = d[d.eval_recall >= target]
        rows.append({"agent": a, "seed": s, "target_success_rate": target,
                     "episodes_to_target": int(hit.episode.iloc[0]) if len(hit) else np.nan})
t7 = pd.DataFrame(rows)
t7.to_csv(f"{OUT}/table7_sample_efficiency.csv", index=False)


# =============================================================== PLOTS
print("plots:")

# Plot 1 - learning curve, episode return. Notes section 3.1 / 3.2, Plot 1.
if not train.empty:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    piv = train[train.agent == "dqn"].pivot(index="episode", columns="seed", values="return")
    for s in piv.columns:
        ax.plot(piv.index, piv[s], color=FAINT, lw=1.0, zorder=2)
    mean, sd = piv.mean(axis=1), piv.std(axis=1, ddof=1)
    ax.fill_between(piv.index, mean - sd, mean + sd, color=C["dqn"], alpha=0.18, lw=0, zorder=3)
    ax.plot(piv.index, mean, color=C["dqn"], lw=2.0, zorder=4, label="mean of 5 seeds")
    ax.plot([], [], color=FAINT, lw=1.0, label="individual seeds")
    ax.axhline(ORACLE_RETURN, color=GREY, ls=":", lw=1.2, zorder=1)
    ax.annotate(f"oracle return {ORACLE_RETURN:.1f}", (piv.index[-1], ORACLE_RETURN),
                xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8, color=GREY)
    ax.annotate(f"{mean.iloc[-1]:.2f}", (piv.index[-1], mean.iloc[-1]), xytext=(5, 0),
                textcoords="offset points", fontsize=8, color=C["dqn"], va="center")
    finish(ax, "Training return rises but stays far below the oracle",
           "Training episode", "Mean episode return (500-episode window)")
    save(fig, "fig1_learning_curve_return.png")

    # Plot 2 - evaluation success rate. Notes 3.3, Plot 3.
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    piv = train[train.agent == "dqn"].pivot(index="episode", columns="seed", values="eval_recall")
    for s in piv.columns:
        ax.plot(piv.index, piv[s], color=FAINT, lw=1.0, zorder=2)
    mean, sd = piv.mean(axis=1), piv.std(axis=1, ddof=1)
    ax.fill_between(piv.index, mean - sd, mean + sd, color=C["dqn"], alpha=0.18, lw=0, zorder=3)
    ax.plot(piv.index, mean, color=C["dqn"], lw=2.0, zorder=4, label="mean of 5 seeds")
    ax.plot([], [], color=FAINT, lw=1.0, label="individual seeds")
    ax.annotate(f"{mean.iloc[-1]:.2f}", (piv.index[-1], mean.iloc[-1]), xytext=(5, 0),
                textcoords="offset points", fontsize=8, color=C["dqn"], va="center")
    finish(ax, "Evaluation success rate, still rising at the episode cap",
           "Training episode", "Success rate on 100 held-out episodes")
    save(fig, "fig2_success_rate.png")

    # Plot 3 - episode length and commit rate, two panels (never a dual axis).
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for col, ax, lab in [("steps", axes[0], "Mean episode length (steps)"),
                         ("committed", axes[1], "Fraction of episodes ending in COMMIT")]:
        piv = train[train.agent == "dqn"].pivot(index="episode", columns="seed", values=col)
        for s in piv.columns:
            ax.plot(piv.index, piv[s], color=FAINT, lw=1.0, zorder=2)
        m = piv.mean(axis=1)
        ax.plot(piv.index, m, color=C["dqn"], lw=2.0, zorder=4, label="mean of 5 seeds")
        ax.plot([], [], color=FAINT, lw=1.0, label="individual seeds")
        finish(ax, lab, "Training episode", "")
    axes[0].axhline(BUDGET, color=GREY, ls=":", lw=1.2)
    axes[0].annotate("budget 12 (timeout)", (0, BUDGET), xytext=(4, -11),
                     textcoords="offset points", fontsize=8, color=GREY)
    save(fig, "fig3_episode_length_and_commit_rate.png")

    # Plot 4 - exploration schedule. Notes 6.1, Plot 4.
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    piv = train[train.agent == "dqn"].pivot(index="episode", columns="seed", values="explore")
    ax.plot(piv.index, piv.mean(axis=1), color=C["dqn"], lw=2.0, label="epsilon (mean of 5 seeds)")
    ax.axhline(0.05, color=GREY, ls=":", lw=1.2)
    ax.annotate("floor 0.05", (piv.index[-1], 0.05), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=8, color=GREY)
    finish(ax, "Exploration decays linearly over 20,000 agent steps, then holds",
           "Training episode", "Probability of a random legal action")
    save(fig, "fig4_exploration_schedule.png")

# Plot 5 - success rate against compute budget, all methods. Notes 13 "matched budget".
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for m in AGENTS:
    g = t1[t1.method == m]
    if g.empty:
        continue
    ax.errorbar(g.mean_episode_length, g.success_rate, yerr=g.success_ci95,
                color=C[m], lw=2.0, marker="o", ms=5, capsize=3, zorder=4, label=LABEL[m])
    ax.annotate(LABEL[m], (g.mean_episode_length.iloc[-1], g.success_rate.iloc[-1]),
                xytext=(6, 0), textcoords="offset points", fontsize=8,
                color=C[m], va="center")
for _, r in bl.iterrows():
    if r["method"] in ("oracle", "exhaustive"):
        continue
    ax.scatter(r.mean_episode_length, r.success_rate, color=GREY, marker="s", s=28, zorder=5)
    ax.annotate(r["method"], (r.mean_episode_length, r.success_rate), xytext=(0, -12),
                textcoords="offset points", fontsize=8, color=GREY, ha="center")
ax.scatter([], [], color=GREY, marker="s", s=28, label="heuristic baselines")
ax.set_xlim(0, 14)
finish(ax, "Success rate at matched compute (error bars: 95% CI across seeds)",
       "Mean encoder passes per image", "Success rate")
save(fig, "fig5_success_vs_compute.png")

# Plot 6 - robustness. Notes 8.1.
fams = sorted(rob.corruption.unique())
if fams:
    fig, axes = plt.subplots(1, len(fams), figsize=(2.2 * len(fams) + 1.2, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        for m in AGENTS:
            g = t4[(t4.method == m) & (t4.corruption == fam)].sort_values("severity")
            if g.empty:
                continue
            base = g.nominal_success_rate.iloc[0]
            ax.plot([0] + list(g.severity), [base] + list(g.success_rate),
                    color=C[m], lw=2.0, marker="o", ms=4, label=LABEL[m])
        ax.set_xticks([0, 1, 2, 3])
        finish(ax, fam.replace("_", " "), "Severity (0 = clean)", "", legend=False)
    axes[0].set_ylabel("Success rate")
    axes[-1].legend(loc="upper right", fontsize=8)
    save(fig, "fig6_robustness.png")

# Plot 7 - box plot across seeds. Notes Plot 5.
fig, ax = plt.subplots(figsize=(4.6, 3.4))
data, labels, colors = [], [], []
for m in AGENTS:
    g = agent_clean[(agent_clean.method == m) & (agent_clean.budget == BUDGET)]
    if g.empty:
        continue
    data.append(g.recall.values); labels.append(f"{LABEL[m]}\n(n={len(g)})"); colors.append(C[m])
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                medianprops=dict(color="white", lw=1.6),
                flierprops=dict(marker="o", ms=4, mfc=GREY, mec="none"))
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col); patch.set_edgecolor(col); patch.set_alpha(0.85)
for i, d in enumerate(data):
    ax.scatter(np.full(len(d), i + 1), d, color="white", s=10, zorder=5, edgecolors=GREY, lw=0.6)
finish(ax, f"Final success rate across seeds, budget {BUDGET}", "", "Success rate", legend=False)
save(fig, "fig7_seed_spread.png")

# ------------------------------------------------------------- summary
summary = {
    "evaluation_episodes_per_point": N_EVAL,
    "budget": BUDGET,
    "oracle_return": ORACLE_RETURN,
    "quadtree_nodes_depth_0_to_4": N_NODES,
    "headline": {},
    "not_available_here": [
        "mean absolute TD error (notes 4.3) - needs rollouts, see record_rollouts.py",
        "Bellman residual (notes 4.4) - same",
        "state-space coverage (notes 6.2) - same",
        "state visitation entropy (notes 6.3) - same",
        "wall-clock training time (notes 7) - wall column in train_log.csv is empty",
        "inference latency (notes 7) - needs a timed rollout",
    ],
}
for m in AGENTS:
    g = t1[(t1.method == m) & (t1.budget == BUDGET)]
    if not g.empty:
        r = g.iloc[0]
        summary["headline"][m] = {
            "success_rate": round(float(r.success_rate), 4),
            "std": round(float(r.success_std), 4),
            "ci95_halfwidth": round(float(r.success_ci95), 4),
            "mean_episode_length": round(float(r.mean_episode_length), 3),
            "seeds": int(r.seeds),
        }
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== headline, budget 12 ===")
print(t1[t1.budget == BUDGET].round(4).to_string(index=False))
print("\nwrote 7 tables + summary.json to", OUT)
