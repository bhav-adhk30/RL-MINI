"""Experiment 2: turn the supervised probe into a DESCENT POLICY and race it
against the RL agent at matched compute.

This is the baseline the "ResNet is doing all the heavy lifting" objection is
really proposing. It gets:

  - the same frozen encoder
  - FULL per-step supervision (a label at every level), which the agent never has
  - no exploration, no credit assignment, no sparse reward

and it cannot do the three things the policy can: back out of a wrong branch,
stop early, or spend its budget unevenly. So any gap at matched cost is
attributable to search behaviour and nothing else.

Note what the supervised policy cannot do at all: it needs 4 passes per level
for 3 levels, so it does not exist below a budget of 12. The RL agent runs at
any budget. That inflexibility is itself a result.

Run (Kaggle):
  python3 scripts/probe_policy.py
  python3 scripts/probe_policy.py --checkpoint runs/dqn/seed2_lam0.1/best.pt
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from zsrl import CANVAS, MIN_COMMIT_DEPTH, get_device
from zsrl.baselines import odd_one_out_scores
from zsrl.dataset import MVTecDefects
from zsrl.encoder import FrozenEncoder
from zsrl.env import ZoomSearchEnv, child, contains_centre

PASSES_PER_LEVEL = 4
LEVELS = MIN_COMMIT_DEPTH


def node_at_depth(box, depth):
    node = (0, 0, 0)
    for _ in range(depth):
        nxt = next((child(node, q) for q in range(4)
                    if contains_centre(child(node, q), box)), None)
        if nxt is None:
            return None
        node = nxt
    return node


def child_features(img, ikey, node, encoder):
    """(4, 2048): each child encoded at its own resolution. Costs 4 passes."""
    return np.stack([encoder.encode_node(img, ikey, child(node, q)) for q in range(4)])


def collect(ds, encoder, tag):
    """Training data for the probe: child features at every level on the
    correct root-to-defect path, labelled with the correct next child."""
    X, y = [], []
    for i in range(len(ds)):
        path, box, label, key = ds[i]
        img = Image.open(path).convert("RGB").resize((CANVAS, CANVAS))
        ikey = key + "|clean|id"
        for d in range(LEVELS):
            node = node_at_depth(box, d)
            if node is None:
                break
            target = next((q for q in range(4)
                           if contains_centre(child(node, q), box)), None)
            if target is None:
                break
            X.append(child_features(img, ikey, node, encoder))
            y.append(target)
        if (i + 1) % 50 == 0:
            print(f"  {tag}: {i + 1}/{len(ds)}", flush=True)
    return np.stack(X), np.array(y)


class SharedHead:
    """One linear map 2048 -> 1, scored on each child, softmax over the four.
    Permutation-equivariant: the right inductive bias for 'which of these four'."""

    def __init__(self, device):
        self.device, self.head, self.mu, self.sd = device, None, None, None

    def fit(self, X, y, epochs=600, lr=3e-3, wd=1e-2):
        flat = X.reshape(-1, X.shape[-1])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6
        x = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32, device=self.device)
        t = torch.tensor(y, dtype=torch.long, device=self.device)
        self.head = nn.Linear(X.shape[-1], 1, bias=False).to(self.device)
        opt = torch.optim.Adam(self.head.parameters(), lr=lr, weight_decay=wd)
        lossf = nn.CrossEntropyLoss()
        for _ in range(epochs):
            opt.zero_grad()
            lossf(self.head(x).squeeze(-1), t).backward()
            opt.step()
        with torch.no_grad():
            return (self.head(x).squeeze(-1).argmax(1) == t).float().mean().item()

    @torch.no_grad()
    def choose(self, feats):
        x = torch.tensor((feats - self.mu) / self.sd,
                         dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.head(x).squeeze(-1).argmax(1).item())

    @torch.no_grad()
    def accuracy(self, X, y):
        x = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32, device=self.device)
        t = torch.tensor(y, dtype=torch.long, device=self.device)
        return (self.head(x).squeeze(-1).argmax(1) == t).float().mean().item()


def run_descent(ds, encoder, pick):
    """Greedy descent to MIN_COMMIT_DEPTH using `pick(img, ikey, node) -> quadrant`.
    Always costs PASSES_PER_LEVEL * LEVELS. Returns recall."""
    wins = []
    for i in range(len(ds)):
        path, box, label, key = ds[i]
        img = Image.open(path).convert("RGB").resize((CANVAS, CANVAS))
        ikey = key + "|clean|id"
        node = (0, 0, 0)
        for _ in range(LEVELS):
            node = child(node, pick(img, ikey, node))
        wins.append(contains_centre(node, box))
    return float(np.mean(wins))


def run_agent(ds, encoder, agent, budget):
    env = ZoomSearchEnv(ds, encoder, budget=budget)
    wins, costs = [], []
    for i in range(len(ds)):
        s = env.reset(index=i)
        while True:
            a = agent.act(s, env.legal(), greedy=True)
            s, r, done, info = env.step(a)
            if done:
                break
        wins.append(info["success"]); costs.append(info["steps"])
    return float(np.mean(wins)), float(np.mean(costs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="optional runs/<agent>/seed*_lam*/best.pt to race against")
    p.add_argument("--agent", default="dqn")
    args = p.parse_args()

    device = get_device()
    encoder = FrozenEncoder(device)
    tr_ds, te_ds = MVTecDefects(split="train"), MVTecDefects(split="test")
    if len(tr_ds) == 0:
        raise SystemExit("MVTec AD not found -- set MVTEC_ROOT")
    print(f"device {device} | train {len(tr_ds)} test {len(te_ds)}\n")

    print("collecting supervised training data for the probe...")
    Xtr, ytr = collect(tr_ds, encoder, "train")
    Xte, yte = collect(te_ds, encoder, "test")
    print(f"  {len(ytr)} train decisions, {len(yte)} test decisions\n")

    head = SharedHead(device)
    tr_acc = head.fit(Xtr, ytr)
    te_acc = head.accuracy(Xte, yte)
    print(f"probe per-decision accuracy: train {tr_acc:.3f}  test {te_acc:.3f} "
          f"(chance 0.250)\n")

    print("running probe-as-policy over the test split...")
    probe_recall = run_descent(
        te_ds, encoder,
        lambda img, k, n: head.choose(child_features(img, k, n, encoder)))

    print("running odd-one-out descent for reference...")
    odd_recall = run_descent(
        te_ds, encoder, lambda img, k, n: int(np.argmax(odd_one_out_scores(img, n))))

    rows = [
        ("odd-one-out greedy descent", "none", odd_recall,
         PASSES_PER_LEVEL * LEVELS, "no CNN, no learning"),
        ("probe-as-policy (supervised)", "per-step labels", probe_recall,
         PASSES_PER_LEVEL * LEVELS, "ResNet + full supervision, no RL"),
    ]

    if args.checkpoint:
        from scripts.train import build_agent
        from zsrl.env import N_ACTIONS
        dim = ZoomSearchEnv(te_ds, encoder).state_dim
        agent = build_agent(args.agent, dim, device, 0)
        agent.load(args.checkpoint)
        for b in (8, 12):
            rec, cost = run_agent(te_ds, encoder, agent, b)
            rows.append((f"{args.agent.upper()} (RL), budget {b}", "terminal reward only",
                         rec, cost, "learns backtracking and when to stop"))

    print("\n" + "=" * 88)
    print(f"{'method':<34}{'supervision':<22}{'recall':>8}{'passes':>9}   note")
    print("-" * 88)
    for m, sup, rec, cost, note in rows:
        print(f"{m:<34}{sup:<22}{rec:>8.3f}{cost:>9.2f}   {note}")
    print("=" * 88)
    print(f"""
Per-decision accuracy of the supervised probe is {te_acc:.3f}; chained over
{LEVELS} levels that predicts {te_acc ** LEVELS:.3f} end-to-end, and it measured
{probe_recall:.3f}.

Read the table as: the probe-as-policy row IS the "ResNet does the heavy
lifting" hypothesis, implemented, with per-step labels the agent never gets.
If the RL rows match or beat it at equal or lower cost, the difference is
search behaviour -- backing out of wrong branches and choosing when to stop --
because nothing else differs between them. If the RL rows lose, say so: a
one-step supervised policy beating sequential search is a real finding about
when RL earns its place.

Also note the supervised policy has no entry below {PASSES_PER_LEVEL * LEVELS}
passes. It cannot trade accuracy for compute at all. The RL policy runs at any
budget, which is the axis this project is actually about.""")


if __name__ == "__main__":
    main()
