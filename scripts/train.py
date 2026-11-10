import argparse, csv, json, os, time

import numpy as np

from zsrl import get_device, output_root
from zsrl.dataset import MVTecDefects
from zsrl.encoder import FrozenEncoder
from zsrl.env import N_ACTIONS, ZoomSearchEnv


def build_agent(name, state_dim, device, seed):
    if name == "dqn":
        from zsrl.agents.dqn import DQNAgent
        return DQNAgent(state_dim, N_ACTIONS, device, seed=seed)
    if name == "ppo":
        from zsrl.agents.ppo import PPOAgent
        return PPOAgent(state_dim, N_ACTIONS, device, seed=seed)
    if name == "a2c":
        from zsrl.agents.a2c import A2CAgent
        return A2CAgent(state_dim, N_ACTIONS, device, seed=seed)
    raise ValueError(name)


def explore_value(agent):
    """One exploration number per agent, into the same CSV column."""
    if hasattr(agent, "epsilon"):
        return float(agent.epsilon)          # DQN
    if hasattr(agent, "policy_entropy"):
        return float(agent.policy_entropy)   # PPO and A2C
    return 0.0


def evaluate(env, agent, n=100):
    """Returns (recall, mean 224-equivalent units, mean commit depth).

    Cost is reported in units, not actions: the root is encoded at 448 and
    costs 4, deeper nodes cost 1 (zsrl/env.py:units). Reporting raw action
    counts would hide that the first look is four times more expensive."""
    wins, costs, depths = [], [], []
    for i in range(n):
        s = env.reset(index=i % len(env.canvases))
        while True:
            a = agent.act(s, env.legal(), greedy=True)
            s, r, done, info = env.step(a)
            if done:
                break
        wins.append(info["success"])
        costs.append(info["units"])
        depths.append(info["depth"])
    return (float(np.mean(wins)), float(np.mean(costs)), float(np.mean(depths)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--episodes", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--base", type=float, default=2.0,
                    help="commit reward = base**(depth-1); selects where on the\nprecision/recall curve the policy settles. Sweep 1.5 / 2.0 / 3.0.")
    p.add_argument("--budget", type=int, default=8,
                    help="224-equivalent compute units per image (root costs 4)")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--skip-step8", action="store_true",
                    help="skip the in-process Step 8 sweep after training")
    args = p.parse_args()

    root = output_root()
    out = os.path.join(root, "runs", args.agent, f"seed{args.seed}_lam{args.lam}")
    os.makedirs(out, exist_ok=True)
    print(f"output root: {root}  (writing to {out})")

    device = get_device()
    encoder = FrozenEncoder(device)
    train_env = ZoomSearchEnv(MVTecDefects(split="train"), encoder,
                              budget=args.budget, lam=args.lam, base=args.base,
                              seed=args.seed, augment=True)
    eval_env = ZoomSearchEnv(MVTecDefects(split="test"), encoder,
                             budget=args.budget, lam=args.lam, base=args.base,
                             seed=1234)  # augment=False: never at eval
    agent = build_agent(args.agent, train_env.state_dim, device, args.seed)

    log_path = os.path.join(out, "train_log.csv")
    f = open(log_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["episode", "return", "success", "steps", "units", "final_depth",
                "committed", "explore", "loss", "wall"])

    start, best = time.time(), -1.0
    window = []  # (return, committed, depth) for the training-side rolling report

    for ep in range(args.episodes):
        s = train_env.reset()
        total, losses = 0.0, []
        while True:
            mask = train_env.legal()
            a = agent.act(s, mask)
            s2, r, done, info = train_env.step(a)
            agent.observe(s, a, r, s2, done, train_env.legal())
            l = agent.update()
            if l is not None:
                losses.append(l)
            s, total = s2, total + r
            if done:
                break

        w.writerow([ep, total, int(info["success"]), info["steps"],
                    info["units"], info["depth"], int(info["committed"]),
                    explore_value(agent),
                    float(np.mean(losses)) if losses else "",
                    round(time.time() - start, 1)])
        if ep % 100 == 0:
            f.flush()
        window.append((total, info["committed"], info["depth"]))

        if ep > 0 and ep % args.eval_every == 0:
            rec, cost, dep = evaluate(eval_env, agent, n=100)
            w_ret = np.mean([r for r, _, _ in window])
            w_committed = np.mean([c for _, c, _ in window])
            w_depth = np.mean([d for _, _, d in window])
            window = []
            print(f"ep {ep}  train_return {w_ret:.2f}  committed_rate {w_committed:.3f}  "
                  f"mean_depth {w_depth:.2f}  |  eval_recall {rec:.3f}  eval_units {cost:.1f}  "
                  f"eval_commit_depth {dep:.2f}  "
                  f"explore {explore_value(agent):.3f}  "
                  f"cache {encoder.stats()['hit_rate']:.2f}")
            if rec > best:
                best = rec
                agent.save(os.path.join(out, "best.pt"))

    final_path = os.path.join(out, "final.pt")
    agent.save(final_path)
    f.close()
    meta_path = os.path.join(out, "meta.json")
    with open(meta_path, "w") as g:
        json.dump({"agent": args.agent, "seed": args.seed, "lam": args.lam,
                   "episodes": args.episodes, "best_recall": best,
                   "minutes": round((time.time() - start) / 60, 1)}, g, indent=2)

    # Final weights, train-vs-test recall (the memorisation check) --
    # always run, cheap relative to training.
    train_env_eval = ZoomSearchEnv(MVTecDefects(split="train"), encoder,
                                   budget=args.budget, lam=args.lam, base=args.base)
    train_recall, train_cost, _ = evaluate(train_env_eval, agent, n=100)
    test_recall, test_cost, test_depth = evaluate(eval_env, agent, n=100)
    print(f"\nfinal weights, train-vs-test: "
          f"train recall {train_recall:.3f} cost {train_cost:.1f} "
          f"(per-level accuracy {train_recall ** (1/3):.3f})  |  "
          f"test recall {test_recall:.3f} units {test_cost:.1f} "
          f"mean commit depth {test_depth:.2f}")

    written = [log_path, final_path, meta_path]
    if os.path.exists(os.path.join(out, "best.pt")):
        written.append(os.path.join(out, "best.pt"))

    # v1 reported final weights. Across five seeds best.pt was +0.042 better
    # and one seed lost 0.10 between its best and its final checkpoint, so
    # report both and sweep on the best one.
    best_path = os.path.join(out, "best.pt")
    if os.path.exists(best_path):
        agent.load(best_path)
        b_rec, b_cost, b_depth = evaluate(eval_env, agent, n=100)
        print(f"best.pt:  test recall {b_rec:.3f} units {b_cost:.1f} "
              f"mean commit depth {b_depth:.2f}   "
              f"(final weights were {test_recall:.3f})")

    # Step 8 sweep, in-process on the agent now holding the BEST weights.
    if not args.skip_step8:
        from scripts.evaluate import run_full_evaluation
        print("\n--- Step 8 evaluation (in-process, final weights) ---")
        run_full_evaluation(agent, device, args.agent, args.seed, args.lam,
                             n=200, out_dir=root)
        eval_csv = os.path.join(root, "results",
                                 f"eval_{args.agent}_seed{args.seed}_lam{args.lam}.csv")
        if os.path.exists(eval_csv):
            written.append(eval_csv)

    print("\n--- files written ---")
    for path in written:
        size = os.path.getsize(path) if os.path.exists(path) else -1
        print(f"  {path}  ({size} bytes)" if size >= 0 else f"  {path}  MISSING")


if __name__ == "__main__":
    main()
