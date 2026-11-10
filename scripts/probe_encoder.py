"""Diagnostic: how much of the defect's POSITION is linearly recoverable from
the encoder, and does encoding each child directly recover more than pooling
the parent?

This decides the v2 design. Two feature constructions for the same decision
("which of this node's four children holds the defect?"):

  pooled   (v1) one encoder call on the NODE. FrozenEncoder 2x2-pools layer4
                into four contiguous 512-d blocks, one per child. Cost 1 pass.
                Each block is an average over ~3.5x3.5 layer4 cells of an
                image already downsampled to 224 -- at depth 0 that means the
                whole 1024px canvas.

  children (v2) four encoder calls, one per CHILD crop, each resized to 224 in
                its own right. 2048-d per child. Cost 4 passes, which is what
                odd_one_out and normal_model_descent already spend per level.

Both are probed with full supervision -- a per-step label the RL agent never
receives -- so each number is an UPPER BOUND on what that representation makes
linearly available at one step. Chance is 0.25.

Two probe heads, because they answer different questions:

  shared   one linear map D->1 scored on each child, softmax over the four.
           Permutation-equivariant, D parameters, the inductive bias a sensible
           policy would have. This is the fair comparison between pooled and
           children.
  full     concatenate the four blocks, one linear map 4D->4. More expressive,
           overfits on a few hundred images -- read its train/test gap, not its
           test score alone.

Also reported, as controls that use no learning at all:
  norm     pick the child with the largest feature L2 norm
  oddone   pick the child least like its three siblings (zsrl.baselines), the
           quantity saliency_descent wins 0.145 recall with and which v1's
           state vector does not contain

Run:  python3 scripts/probe_encoder.py
      python3 scripts/probe_encoder.py --depth 1
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from zsrl import CANVAS, get_device
from zsrl.baselines import odd_one_out_scores
from zsrl.dataset import MVTecDefects
from zsrl.encoder import FrozenEncoder
from zsrl.env import child, contains_centre, node_box


def node_at_depth(box, depth):
    """The node at `depth` on the correct root-to-defect path, or None."""
    node = (0, 0, 0)
    for _ in range(depth):
        for q in range(4):
            c = child(node, q)
            if contains_centre(c, box):
                node = c
                break
        else:
            return None
    return node


def build(ds, encoder, depth, tag):
    """Returns pooled (N,4,512), children (N,4,2048), oddone (N,4), y (N,)."""
    P, C, O, y = [], [], [], []
    for i in range(len(ds)):
        path, box, label, key = ds[i]
        node = node_at_depth(box, depth)
        if node is None:
            continue
        target = next((q for q in range(4)
                       if contains_centre(child(node, q), box)), None)
        if target is None:
            continue

        img = Image.open(path).convert("RGB").resize((CANVAS, CANVAS))
        ikey = key + "|clean|id"

        # v1: one call on the node, split into the four 512-d child blocks.
        P.append(encoder.encode_node(img, ikey, node).reshape(4, 512))
        # v2: one call per child crop, at the child's own resolution.
        C.append(np.stack([encoder.encode_node(img, ikey, child(node, q))
                           for q in range(4)]))
        O.append(np.array(odd_one_out_scores(img, node), dtype=np.float32))
        y.append(target)

        if (i + 1) % 50 == 0:
            print(f"  {tag}: {i + 1}/{len(ds)}", flush=True)
    return np.stack(P), np.stack(C), np.stack(O), np.array(y)


def standardise(tr, te):
    flat = tr.reshape(-1, tr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return (tr - mu) / sd, (te - mu) / sd


def shared_probe(Xtr, ytr, Xte, yte, device, epochs=600, lr=3e-3, wd=1e-2):
    """One linear map D->1 applied to every child; softmax over the four."""
    Xtr, Xte = standardise(Xtr, Xte)
    xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    xte = torch.tensor(Xte, dtype=torch.float32, device=device)
    ttr = torch.tensor(ytr, dtype=torch.long, device=device)
    tte = torch.tensor(yte, dtype=torch.long, device=device)

    head = nn.Linear(Xtr.shape[-1], 1, bias=False).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(xtr).squeeze(-1), ttr).backward()
        opt.step()
    with torch.no_grad():
        tr = (head(xtr).squeeze(-1).argmax(1) == ttr).float().mean().item()
        te = (head(xte).squeeze(-1).argmax(1) == tte).float().mean().item()
    return tr, te


def full_probe(Xtr, ytr, Xte, yte, device, epochs=600, lr=1e-3, wd=1e-1):
    """Concatenate the four blocks; one linear map 4D->4."""
    Xtr, Xte = standardise(Xtr, Xte)
    ntr, nte = len(Xtr), len(Xte)
    xtr = torch.tensor(Xtr.reshape(ntr, -1), dtype=torch.float32, device=device)
    xte = torch.tensor(Xte.reshape(nte, -1), dtype=torch.float32, device=device)
    ttr = torch.tensor(ytr, dtype=torch.long, device=device)
    tte = torch.tensor(yte, dtype=torch.long, device=device)

    head = nn.Linear(xtr.shape[1], 4).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(xtr), ttr).backward()
        opt.step()
    with torch.no_grad():
        tr = (head(xtr).argmax(1) == ttr).float().mean().item()
        te = (head(xte).argmax(1) == tte).float().mean().item()
    return tr, te


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=0,
                   help="node depth whose four children the probe chooses between")
    args = p.parse_args()

    device = get_device()
    encoder = FrozenEncoder(device)
    print(f"device {device} | choosing among the children of a depth-{args.depth} "
          f"node (-> depth {args.depth + 1})\n")

    tr_ds, te_ds = MVTecDefects(split="train"), MVTecDefects(split="test")
    if len(tr_ds) == 0:
        raise SystemExit("MVTec AD not found under data/mvtec -- see zsrl/dataset.py")
    print(f"train {len(tr_ds)} images, test {len(te_ds)} images")

    Ptr, Ctr, Otr, ytr = build(tr_ds, encoder, args.depth, "train")
    Pte, Cte, Ote, yte = build(te_ds, encoder, args.depth, "test")
    print(f"\nusable: train {len(ytr)}, test {len(yte)}  "
          f"| test label balance {np.bincount(yte, minlength=4)}")

    rows = [("chance", "", 0.25, 0.25)]
    rows.append(("norm rule (no learning)", "pooled",
                 float((np.linalg.norm(Ptr, axis=2).argmax(1) == ytr).mean()),
                 float((np.linalg.norm(Pte, axis=2).argmax(1) == yte).mean())))
    rows.append(("norm rule (no learning)", "children",
                 float((np.linalg.norm(Ctr, axis=2).argmax(1) == ytr).mean()),
                 float((np.linalg.norm(Cte, axis=2).argmax(1) == yte).mean())))
    rows.append(("odd-one-out (no learning, no CNN)", "-",
                 float((Otr.argmax(1) == ytr).mean()),
                 float((Ote.argmax(1) == yte).mean())))

    for name, Xtr, Xte in (("pooled  (v1, 1 pass)", Ptr, Pte),
                           ("children(v2, 4 passes)", Ctr, Cte)):
        rows.append(("shared linear probe", name, *shared_probe(Xtr, ytr, Xte, yte, device)))
    for name, Xtr, Xte in (("pooled  (v1, 1 pass)", Ptr, Pte),
                           ("children(v2, 4 passes)", Ctr, Cte)):
        rows.append(("full linear probe", name, *full_probe(Xtr, ytr, Xte, yte, device)))

    print("\n" + "=" * 74)
    print(f"{'method':<34}{'features':<24}{'train':>8}{'test':>8}")
    print("-" * 74)
    for m, f, tr, te in rows:
        print(f"{m:<34}{f:<24}{tr:>8.3f}{te:>8.3f}")
    print("=" * 74)

    sh = {f: te for m, f, _, te in rows if m == "shared linear probe"}
    if len(sh) == 2:
        a = sh["pooled  (v1, 1 pass)"]; b = sh["children(v2, 4 passes)"]
        print(f"\nshared probe, children minus pooled: {b - a:+.3f}")
        print("""
How to read this. The shared-probe row is the fair comparison: same head, same
parameter count, only the representation differs. A large positive gap means v1
is throwing away recoverable position information by pooling the parent instead
of encoding each child, and the v2 redesign is targeting a real defect. A gap
near zero means pooling loses little, the bottleneck is elsewhere, and 4x the
encoder cost would buy nothing.

Whatever the probe reaches, the remainder up to 1.0 is what no single-step
representation provides: it has to be recovered by moving, backtracking and
deciding when to stop under a budget. That part is the policy's job.""")


if __name__ == "__main__":
    main()
