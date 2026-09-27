import argparse, csv, json, os, time

import numpy as np

from zsrl import get_device
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
    wins, costs = [], []
    for i in range(n):
        s = env.reset(index=i % len(env.canvases))
        while True:
            a = agent.act(s, env.legal(), greedy=True)
            s, r, done, info = env.step(a)
            if done:
                break
        wins.append(info["success"])
        costs.append(info["steps"])
    return float(np.mean(wins)), float(np.mean(costs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--episodes", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=500)
    args = p.parse_args()

    out = os.path.join("runs", args.agent, f"seed{args.seed}_lam{args.lam}")
    os.makedirs(out, exist_ok=True)

    device = get_device()
    encoder = FrozenEncoder(device)
    train_env = ZoomSearchEnv(MVTecDefects(split="train"), encoder,
                              lam=args.lam, seed=args.seed, augment=True)
    eval_env = ZoomSearchEnv(MVTecDefects(split="test"), encoder,
                             lam=args.lam, seed=1234)  # augment=False: never at eval
    agent = build_agent(args.agent, train_env.state_dim, device, args.seed)

    f = open(os.path.join(out, "train_log.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["episode", "return", "success", "steps", "final_depth",
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
                    info["depth"], int(info["committed"]),
                    explore_value(agent),
                    float(np.mean(losses)) if losses else "",
                    round(time.time() - start, 1)])
        if ep % 100 == 0:
            f.flush()
        window.append((total, info["committed"], info["depth"]))

        if ep > 0 and ep % args.eval_every == 0:
            rec, cost = evaluate(eval_env, agent, n=100)
            w_ret = np.mean([r for r, _, _ in window])
            w_committed = np.mean([c for _, c, _ in window])
            w_depth = np.mean([d for _, _, d in window])
            window = []
            print(f"ep {ep}  train_return {w_ret:.2f}  committed_rate {w_committed:.3f}  "
                  f"mean_depth {w_depth:.2f}  |  eval_recall {rec:.3f}  eval_cost {cost:.1f}  "
                  f"explore {explore_value(agent):.3f}  "
                  f"cache {encoder.stats()['hit_rate']:.2f}")
            if rec > best:
                best = rec
                agent.save(os.path.join(out, "best.pt"))

    agent.save(os.path.join(out, "final.pt"))
    f.close()
    with open(os.path.join(out, "meta.json"), "w") as g:
        json.dump({"agent": args.agent, "seed": args.seed, "lam": args.lam,
                   "episodes": args.episodes, "best_recall": best,
                   "minutes": round((time.time() - start) / 60, 1)}, g, indent=2)


if __name__ == "__main__":
    main()
