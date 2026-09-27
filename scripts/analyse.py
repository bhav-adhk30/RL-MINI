"""Step 9 figures. Reads whatever results/eval_<agent>_seed*_lam*.csv and
runs/<agent>/seed*_lam*/train_log.csv files exist -- handles however many
agents have landed (currently DQN; PPO/A2C are built against the same file
layout and appear automatically once their runs exist), never assumes
exactly three. Does not build the app; see app/streamlit_app.py for that.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = "results"
RUNS = "runs"
CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness", "jpeg"]


def discover_agents():
    """Agent names with at least one eval CSV, in a stable order."""
    paths = glob.glob(os.path.join(RESULTS, "eval_*_seed*_lam*.csv"))
    agents = set()
    for p in paths:
        m = re.match(r"eval_(.+)_seed\d+_lam[\d.]+\.csv", os.path.basename(p))
        if m:
            agents.add(m.group(1))
    return sorted(agents)


def load_eval(agent):
    paths = sorted(glob.glob(os.path.join(RESULTS, f"eval_{agent}_seed*_lam*.csv")))
    if not paths:
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load_training_logs(agent):
    """{seed: dataframe} for every seed dir found under runs/<agent>/."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, agent, "seed*_lam*"))):
        m = re.search(r"seed(\d+)_lam", os.path.basename(d))
        log = os.path.join(d, "train_log.csv")
        if m and os.path.exists(log):
            out[int(m.group(1))] = pd.read_csv(log)
    return out


def explore_label(agent):
    return {"dqn": "epsilon", "ppo": "policy entropy", "a2c": "policy entropy"}.get(agent, "explore")


# ---------------------------------------------------------------------------
def figure_learning_curves(agents, random_recall):
    """Recall (or the best proxy available) vs episode, mean +/- sd shaded
    across seeds, one panel per agent, random baseline marked."""
    n = len(agents)
    if n == 0:
        print("no training logs found, skipping learning-curve figure")
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, agent in zip(axes, agents):
        logs = load_training_logs(agent)
        if not logs:
            ax.set_title(f"{agent.upper()} (no logs)")
            continue
        # Align on the episode index every seed shares.
        common_eps = sorted(set.intersection(*[set(df["episode"]) for df in logs.values()]))
        col = "eval_recall" if "eval_recall" in next(iter(logs.values())).columns else "return"
        mat = np.array([[df.set_index("episode").loc[ep, col] for ep in common_eps]
                         for df in logs.values()])
        mean, sd = mat.mean(axis=0), mat.std(axis=0)
        ax.plot(common_eps, mean, marker="o", ms=3, label=f"{agent.upper()} mean")
        ax.fill_between(common_eps, mean - sd, mean + sd, alpha=0.25,
                         label=f"+/- 1 sd (n={len(logs)} seeds)")
        ax.axhline(random_recall, ls="--", c="grey", label=f"random ({random_recall:.3f})")
        ax.set_xlabel("episode")
        ax.set_ylabel(col)
        ax.set_title(agent.upper())
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "learning_curves.png"), dpi=140)
    plt.close(fig)
    print(f"wrote {RESULTS}/learning_curves.png")


def figure_recall_vs_compute(agents):
    """Figure 1: recall vs. compute, log x-axis, agents as connected lines
    across their budget sweep, baselines as marked points."""
    fig, ax = plt.subplots(figsize=(8, 6))
    marks = {"oracle": "*", "random": "v", "odd_one_out": "s",
             "normal_model": "^", "exhaustive": "D"}
    baselines_plotted = set()

    for agent in agents:
        df = load_eval(agent)
        if df is None:
            continue
        sweep = df[df.budget.notna() & (df.corruption == "clean")]
        agg = sweep.groupby("budget").agg(
            recall=("recall", "mean"), recall_sd=("recall", "std"),
            cost=("cost", "mean")).reset_index().sort_values("cost")
        ax.errorbar(agg.cost, agg.recall, yerr=agg.recall_sd.fillna(0),
                    marker="o", capsize=3, label=agent.upper())

        base = df[df.budget.isna()].groupby("method").agg(
            recall=("recall", "mean"), cost=("cost", "mean")).reset_index()
        for _, r in base.iterrows():
            if r.method in baselines_plotted:
                continue
            baselines_plotted.add(r.method)
            ax.scatter(r.cost, r.recall, marker=marks.get(r.method, "x"), s=140,
                       c="black", zorder=5)
            ax.annotate(r.method, (r.cost, r.recall),
                        textcoords="offset points", xytext=(8, -4), fontsize=9)

    ax.set_xlabel("Full-resolution evaluations per image")
    ax.set_ylabel("Recall")
    ax.set_xscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "recall_vs_compute.png"), dpi=140)
    plt.close(fig)
    print(f"wrote {RESULTS}/recall_vs_compute.png")


def figure_degradation(agents):
    """Figure 4: recall vs corruption severity, one panel per corruption
    family, one line per agent, odd_one_out as a dashed reference line.
    Motion blur severity 3 called out explicitly, not left as an outlier."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)

    for ax, corr in zip(axes, CORRUPTIONS):
        ax.set_ylim(-0.02, 0.40)  # shared headroom so the callout below fits
        for agent in agents:
            df = load_eval(agent)
            if df is None:
                continue
            clean = df[(df.corruption == "clean") & (df.method == agent) & (df.budget == 12)]
            deg = df[(df.corruption == corr) & (df.method == agent)]
            agg = deg.groupby("severity").agg(recall=("recall", "mean"),
                                               recall_sd=("recall", "std")).reset_index()
            sev0 = pd.DataFrame({"severity": [0],
                                  "recall": [clean.recall.mean()],
                                  "recall_sd": [clean.recall.std()]})
            agg = pd.concat([sev0, agg]).sort_values("severity")
            ax.errorbar(agg.severity, agg.recall, yerr=agg.recall_sd.fillna(0),
                        marker="o", capsize=3, label=agent.upper())

        oo = pd.concat([load_eval(a) for a in agents if load_eval(a) is not None])
        oo_val = oo[(oo.method == "odd_one_out")].recall.mean()
        ax.axhline(oo_val, ls="--", c="grey", label="odd_one_out (clean)")

        if corr == "motion_blur":
            ax.annotate("severity 3: recall collapses\n(fine texture destroyed)",
                        xy=(3, 0.03), xytext=(0.9, 0.35),
                        arrowprops=dict(arrowstyle="->", color="crimson"),
                        color="crimson", fontsize=9, fontweight="bold")

        ax.set_title(corr)
        ax.set_xlabel("severity (0 = clean)")
        if corr == CORRUPTIONS[0]:
            ax.set_ylabel("recall")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "degradation.png"), dpi=140)
    plt.close(fig)
    print(f"wrote {RESULTS}/degradation.png")


def figure_steps_to_commit(agents):
    """Steps-to-commit (eval cost, i.e. nodes visited by the greedy policy
    on held-out data) over the course of training, mean +/- sd across
    seeds, one line per agent."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for agent in agents:
        logs = load_training_logs(agent)
        if not logs or "steps" not in next(iter(logs.values())).columns:
            continue
        common_eps = sorted(set.intersection(*[set(df["episode"]) for df in logs.values()]))
        mat = np.array([[df.set_index("episode").loc[ep, "steps"] for ep in common_eps]
                         for df in logs.values()])
        mean, sd = mat.mean(axis=0), mat.std(axis=0)
        ax.plot(common_eps, mean, marker="o", ms=3, label=f"{agent.upper()} mean")
        ax.fill_between(common_eps, mean - sd, mean + sd, alpha=0.25)
    ax.axhline(3, ls="--", c="grey", label="oracle (3)")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps to commit (held-out, greedy)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "steps_to_commit.png"), dpi=140)
    plt.close(fig)
    print(f"wrote {RESULTS}/steps_to_commit.png")


def figure_lambda_sweep():
    """Figure 2: mean cost vs recall, one point per lambda. Plotted
    alongside the seed-to-seed spread already measured at lambda=0.1 (five
    seeds, same everything else) as an honest reference for how much of
    any lambda-to-lambda difference could just be noise -- each lambda
    point here is a single seed, so on its own it cannot separate a real
    lambda effect from ordinary seed variance."""
    path = os.path.join(RESULTS, "lambda_sweep_dqn.csv")
    if not os.path.exists(path):
        print("no lambda sweep data, skipping")
        return
    df = pd.read_csv(path).sort_values("lam")

    lam01 = None
    eval01 = load_eval("dqn")
    if eval01 is not None:
        b12 = eval01[(eval01.method == "dqn") & (eval01.budget == 12)]
        if len(b12):
            lam01 = b12.recall

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(df.cost, df.recall, marker="o", ms=9, c="C0", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(f"lam={r.lam}", (r.cost, r.recall),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    if lam01 is not None and len(lam01) > 1:
        # the 5-seed spread at lam=0.1, budget=12, as a reference band for
        # how much scatter comes from seed noise alone at ONE lambda value
        ax.errorbar([df[df.lam == 0.1].cost.iloc[0]], [lam01.mean()],
                    yerr=[lam01.std()], fmt="none", ecolor="grey", capsize=5,
                    label=f"lam=0.1, 5-seed sd={lam01.std():.3f} (n=1 per point above)")
        ax.legend(fontsize=8)

    ax.set_xlabel("mean evaluations used (cost)")
    ax.set_ylabel("recall")
    ax.set_title("Lambda sweep (n=1 seed per point -- see caption)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "lambda_sweep.png"), dpi=140)
    plt.close(fig)
    print(f"wrote {RESULTS}/lambda_sweep.png")

    is_monotone_cost = list(df.cost) == sorted(df.cost, reverse=True)
    is_monotone_recall = (list(df.recall) == sorted(df.recall) or
                           list(df.recall) == sorted(df.recall, reverse=True))
    print(f"monotone in cost (decreasing as lambda rises): {is_monotone_cost}")
    print(f"monotone in recall: {is_monotone_recall}")
    if lam01 is not None:
        print(f"5-seed sd at lam=0.1 alone: {lam01.std():.3f} "
              f"(range of the single-seed lambda points: "
              f"{df.recall.max() - df.recall.min():.3f})")


def main():
    agents = discover_agents()
    print("agents with eval results:", agents or "(none)")

    random_recall = 0.015  # from results/dqn_step8_summary.md's baseline table
    for a in agents:
        df = load_eval(a)
        if df is not None and (df.method == "random").any():
            random_recall = df[df.method == "random"].recall.mean()
            break

    os.makedirs(RESULTS, exist_ok=True)
    figure_learning_curves(agents, random_recall)
    figure_recall_vs_compute(agents)
    figure_degradation(agents)
    figure_steps_to_commit(agents)
    figure_lambda_sweep()


if __name__ == "__main__":
    main()
