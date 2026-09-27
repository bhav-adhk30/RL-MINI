import argparse, csv, os

import numpy as np
from PIL import Image

from zsrl import CANVAS, get_device
from zsrl.baselines import (exhaustive, normal_model_descent, oracle,
                             random_descent, saliency_descent)
from zsrl.corruptions import CORRUPTIONS, make
from zsrl.dataset import MVTecDefects
from zsrl.encoder import FrozenEncoder
from zsrl.env import N_ACTIONS, ZoomSearchEnv
from scripts.train import build_agent


def per_level(recall):
    """Per-level accuracy a=recall^(1/3): success requires 3 independent
    correct quadrant choices, so recall understates a smaller per-level
    gap by cubing it. Chance is 0.25 (one in four quadrants), not 0. See
    results/baseline_notes.md."""
    return float(recall) ** (1.0 / 3.0)


def run_agent(agent, canvases, encoder, budget, corruption, tag, n):
    env = ZoomSearchEnv(canvases, encoder, budget=budget,
                        corruption=corruption, corruption_tag=tag, seed=999)
    wins, costs, depths = [], [], []
    for i in range(n):
        s = env.reset(index=i % len(canvases))
        while True:
            s, r, done, info = env.step(agent.act(s, env.legal(), greedy=True))
            if done:
                break
        wins.append(info["success"])
        costs.append(info["steps"])
        depths.append(info["depth"])
    recall = float(np.mean(wins))
    return {"recall": recall, "cost": float(np.mean(costs)),
            "depth": float(np.mean(depths)), "per_level_accuracy": per_level(recall)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    device = get_device()
    canvases = MVTecDefects(split="test")
    run_dir = os.path.join("runs", args.agent, f"seed{args.seed}_lam{args.lam}")

    probe = FrozenEncoder(device)
    state_dim = ZoomSearchEnv(canvases, probe).state_dim
    agent = build_agent(args.agent, state_dim, device, args.seed)
    agent.load(os.path.join(run_dir, "best.pt"))

    rows = []

    # 1. budget sweep on clean canvases
    for budget in (4, 6, 8, 10, 12, 16):
        enc = FrozenEncoder(device)
        m = run_agent(agent, canvases, enc, budget, None, "clean", args.n)
        m.update({"method": args.agent, "budget": budget,
                  "corruption": "clean", "severity": 0, "seed": args.seed})
        rows.append(m)
        print(f"budget {budget:2d}  recall {m['recall']:.3f}  "
              f"per_level {m['per_level_accuracy']:.3f}  cost {m['cost']:.2f}")

    # 2. the baselines, once -- oracle, random, saliency (odd_one_out),
    # exhaustive, and normal_model_descent if its reference matches the
    # current (frozen) encoder's dimensionality
    ref_path = os.path.join("runs", "baselines", "normal_reference.npz")
    reference, ref_encoder = None, None
    if os.path.exists(ref_path):
        cand = np.load(ref_path)
        first_cat = next(iter(cand.files), None)
        if first_cat is not None and cand[first_cat].shape[-1] == probe.dim:
            reference, ref_encoder = cand, FrozenEncoder(device)
        else:
            got = cand[first_cat].shape[-1] if first_cat else "?"
            print(f"note: {ref_path} has dim {got}, current encoder is "
                  f"dim {probe.dim} -- stale (built before the spatial-"
                  f"pooling change). Skipping normal_model_descent; rebuild "
                  f"with `python -m scripts.build_normal_reference` first.")

    rng = np.random.default_rng(0)
    base = {"oracle": [], "random": [], "odd_one_out": [], "exhaustive": []}
    if reference is not None:
        base["normal_model"] = []
    for i in range(args.n):
        path, box, label, key = canvases[i % len(canvases)]
        # MVTec images are 700-1024px depending on category; box_from_mask
        # already scales boxes to CANVAS, so the image must match.
        img = Image.open(path).convert("RGB").resize((CANVAS, CANVAS))
        base["oracle"].append(oracle(box))
        base["random"].append(random_descent(box, rng))
        base["odd_one_out"].append(saliency_descent(img, box))  # saliency_descent scores by odd-one-out, see zsrl/baselines.py
        base["exhaustive"].append(exhaustive(box))
        if reference is not None:
            cat = label.split("/")[0]
            cat_ref = reference[cat] if cat in reference else None
            base["normal_model"].append(
                normal_model_descent(img, box, cat_ref, ref_encoder.encode_pil))
    for name, rs in base.items():
        recall = float(np.mean([r["success"] for r in rs]))
        rows.append({"method": name, "budget": None,
                     "recall": recall,
                     "cost": float(np.mean([r["cost"] for r in rs])),
                     "depth": None, "per_level_accuracy": per_level(recall),
                     "corruption": "clean", "severity": 0, "seed": args.seed})
        print(f"{name:12s} recall {recall:.3f}  per_level {per_level(recall):.3f}")

    # 3. degraded canvases at the default budget
    for name in CORRUPTIONS:
        for sev in (1, 2, 3):
            fn, tag = make(name, sev)
            enc = FrozenEncoder(device)
            m = run_agent(agent, canvases, enc, 12, fn, tag, args.n)
            m.update({"method": args.agent, "budget": 12,
                      "corruption": name, "severity": sev, "seed": args.seed})
            rows.append(m)
            print(f"{name:15s} sev{sev}  recall {m['recall']:.3f}  "
                  f"per_level {m['per_level_accuracy']:.3f}")

    os.makedirs("results", exist_ok=True)
    keys = ["method", "budget", "recall", "per_level_accuracy", "cost", "depth",
            "corruption", "severity", "seed"]
    with open(f"results/eval_{args.agent}_seed{args.seed}_lam{args.lam}.csv",
              "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=keys)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({k: r.get(k) for k in keys})


if __name__ == "__main__":
    main()
