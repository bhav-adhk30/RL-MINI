# CLAUDE.md

## What this repository is

An RL agent that locates a small surface defect in a high-resolution photograph of a
manufactured part by searching a quadtree: it starts at the whole image, repeatedly
descends into one quadrant, and commits when it believes the defect is inside the
current region. The point is to find the defect using far fewer full-resolution model
evaluations than exhaustive tile scanning.

University mini project. Assessed on a live demo plus analysis of convergence,
computational cost and exploration behaviour.

**Full specification: `docs/IMPLEMENTATION.md`. Read it before writing any code.**
It contains the environment design, the reward function, reference implementations,
expected numbers at each checkpoint, and the troubleshooting table. Follow it rather
than inventing an alternative design.

## Your scope

Build these, in this order:

1. `zsrl/__init__.py` — constants and device selection
2. `zsrl/dataset.py` — MVTec AD loading, mask to bounding box
3. `zsrl/encoder.py` — frozen ResNet-18 with a node-keyed cache
4. `zsrl/env.py` — the quadtree environment
5. `zsrl/baselines.py` — oracle, random, saliency, exhaustive
6. `zsrl/agents/dqn.py` — Double DQN with action masking
7. `scripts/train.py` — training loop and CSV logging
8. `zsrl/corruptions.py` — four corruption families, three severities
9. `scripts/evaluate.py` — budget sweep, baselines, degraded conditions
10. `scripts/analyse.py` — the figures
11. `app/streamlit_app.py` — the demo

## Do not do these

- **Do not write `zsrl/agents/ppo.py` or `zsrl/agents/a2c.py`.** Two other people
  are writing those against the interface in `zsrl/agents/base.py`. Create the base
  interface, leave the implementations alone.
- **Do not generate, synthesise or augment training data.** MVTec AD is used exactly
  as distributed. Ground-truth boxes come from the mask PNGs they ship. If the data
  is missing, stop and say so rather than fabricating a substitute.
- **Do not change the environment after Step 4 passes** without flagging it loudly.
  Results collected before and after a change to the state, actions or reward are not
  comparable, and two other people will be training against it.
- **Do not swap the algorithm.** Double DQN with experience replay was chosen and
  justified in the report. If you think something else is better, say so; do not
  silently substitute.
- **Do not use a pretrained object detector anywhere.** The whole claim is about
  search cost. Bringing in a detector destroys the comparison.

## Hard constraints

**Data.** MVTec AD, read-only. On Kaggle it mounts under `/kaggle/input/`; locally it
lives in `data/mvtec/`. Resolve the root from `MVTEC_ROOT` with a default of
`data/mvtec`. Never write into the dataset directory.

**Devices.** The same code runs on an Apple M1 (`mps`) and a Kaggle P100 (`cuda`).
Use the `get_device()` helper everywhere; never hardcode a device. Do not assume CUDA
is present.

**Sessions.** Kaggle sessions terminate after about 12 hours. No single training run
may require longer. Checkpoint to disk periodically, never only at the end.

**Compute accounting.** One encoder forward pass on one region is one unit of compute.
This is the project's headline metric. During evaluation, count the nodes the policy
visited, not `encoder.calls`, because the cache makes repeat visits free and that is
not what a real system would spend.

**Determinism.** Every script takes `--seed`. Seed numpy and torch. Three agents times
three seeds must be reproducible and comparable.

## The gate

`scripts/check_env.py` produces baseline numbers. **Measured on the actual MVTec
test split** (the guide's original 0.3-0.6 saliency range was calibrated on VOC
natural photos before the project switched to MVTec; it does not transfer,
because pixel-variance saliency tracks foreground *objects*, not subtle
industrial surface defects -- see docs/IMPLEMENTATION.md's Step 4 for the full
diagnosis, including the train/good/-based normal_model_descent baseline added
alongside it):

| Method | Recall | Cost |
| --- | --- | --- |
| Oracle | 1.000 | 3.0 |
| Exhaustive | 1.000 | 64.0 |
| `normal_model_descent` | 0.187 | 12.0 |
| `saliency_descent` (odd-one-out) | 0.134 | 12.0 |
| Random | 0.022 | 3.0 |

Separately: a frozen-ResNet-18 "distance from a per-category normal" model
gets **0.575** accuracy picking the correct quadrant at depth 1 alone (chance
is 0.25) -- confirming the encoder features do separate quadrants and the task
is learnable. `normal_model_descent`'s full depth-3 score is lower than that
(0.187) only because depths 2-3 fall back to the cheaper odd-one-out heuristic,
which fails on rigid single-object categories (bottle, screw, transistor,
metal_nut all score 0 there despite strong depth-1 accuracy). DQN, using the
same frozen features at every step rather than a depth-1-only reference, is
expected to close that gap.

Then DQN must overfit a single image: return near 5.7 within roughly 250 episodes,
committing successfully in about 4 steps.

**If either check fails, the environment is wrong. Stop and debug it. Do not tune
hyperparameters and do not proceed to full training.** The troubleshooting section of
`docs/IMPLEMENTATION.md` lists the checks in order.

## Conventions

- Python 3.10+, PyTorch, numpy, PIL, matplotlib, pandas, streamlit. No other
  dependencies without asking.
- No RL libraries. Stable-Baselines3, RLlib and similar are not permitted; the
  assessment requires the learning update to be our own code.
- Type hints on public functions. Docstrings where behaviour is not obvious from the
  name.
- Runs write to `runs/<agent>/seed<N>_lam<L>/`. Never overwrite an existing run
  directory.
- Figures and CSVs go to `results/` and are committed. `data/` and `runs/` are
  gitignored.
- Keep functions short enough to explain in a viva. Clever is worse than clear here.

## Verify as you go

After each file, run its check from the guide and report the actual numbers before
moving on. Do not build three modules and then test.

Specifically:
- After `dataset.py`: how many images survive the filter, and do the boxes land on
  visible defects in `results/dataset_check.png`
- After `encoder.py`: feature shape, root-versus-child distance non-zero, cache hit
  rate near 1.0 on repeats
- After `env.py` and `baselines.py`: the four baseline numbers above
- After `dqn.py`: the single-image overfit
- After `train.py`: the first 2000 episodes, and whether `committed` is rising

## When something is ambiguous

Ask. Do not guess at the reward function, the state layout or the action semantics;
they are specified in the guide and other people are depending on them. If the guide
and this file disagree, this file wins, and tell me about the conflict.
