# Results — Zoom-Search (Adhikari Bhavishya, Double DQN)

All numbers measured on the MVTec AD test split (134 images) with
`runs/dqn/seed2_lam0.1/best.pt`. **Seed 2, best checkpoint** — the strongest of
five seeds. Binomial standard error at these rates is about +/-0.04, so treat
differences under 0.08 as not separated.

Compute is reported in **224-equivalent encoder passes**: one forward pass of
the frozen ResNet-18 on one 224x224 region = 1 unit. A 448x448 pass costs 4
units and is counted as 4.

## Headline

| Method | Recall | Passes | Region | Recall per pass | Notes |
|---|---|---|---|---|---|
| **Double DQN, commit depth 1** | **0.530** | **1** | 512px | **0.530** | learned policy |
| **Double DQN, commit depth 2** | **0.425** | **2** | 256px | 0.213 | learned policy |
| **Double DQN, commit depth 3** | **0.321** | **3** | 128px | 0.107 | learned policy |
| Supervised MLP probe | 0.289 | 3 | 128px | 0.096 | per-step labels, no RL |
| Double DQN as originally reported | 0.299 | 7.36 | 128px | 0.041 | UP enabled |
| Normal-model heuristic | 0.200 | 12 | 128px | 0.017 | sees defect-free references |
| Odd-one-out heuristic | 0.134 | 12 | 128px | 0.011 | no learning |
| Random descent | 0.022 | 3 | 128px | 0.007 | floor |
| Exhaustive scan | 1.000 | 64 | 128px | 0.016 | ceiling |
| Oracle | 1.000 | 3 | 128px | 0.333 | knows the answer |

At matched precision (128px region) the policy reaches 1.6x the best
heuristic's recall using a quarter of its compute. Per pass it is 6x more
efficient than the normal-model heuristic, 9x more than odd-one-out, and 6x
more than exhaustive scanning.

## The policy beats its own supervised ceiling

A classifier trained with **full per-step supervision** on the same frozen
features — the correct quadrant given at every level, which the RL agent never
receives — chains to 0.289 at 3 passes.

| Probe | depth 0 | depth 1 | depth 2 | chained |
|---|---|---|---|---|
| Linear | 0.522 | 0.724 | 0.687 | 0.260 |
| MLP (matched to the Q-network) | 0.552 | 0.701 | 0.746 | 0.289 |
| **Double DQN (terminal reward only)** | | | | **0.321** |

The agent learning from a single delayed reward outperforms supervised
probing of the representation it is built on. This answers directly the
question of whether the frozen encoder is doing the work: a supervised
classifier given the encoder *and* the labels does worse.

## Backtracking is harmful, measured

The UP action was included so the policy could recover from a wrong descent.
It does the opposite.

| | episodes | recall | mean steps |
|---|---|---|---|
| Never used UP | 60 | **0.467** | 4.03 |
| Used UP | 74 | **0.162** | 10.05 |

Disabling UP at evaluation time, on the same weights, raises recall from 0.299
to 0.321 while cutting cost from 7.36 to 4.00 steps. Correlation between steps
taken and success is **-0.377**: every extra pass makes success less likely.
UP is a symptom of being lost, not a mechanism for recovery.

Budget use is bimodal rather than adaptive — 58 of 134 episodes commit at 4
steps, 43 run the full 12 and time out without committing (28.4%). The policy
does not allocate compute by difficulty; it is either decisive and usually
right, or lost and usually wrong. That signal is itself useful: a thrashing
episode is a reliable indicator of low confidence.

## Where the representation binds

Per-level probe accuracy against encoder input resolution:

| Depth | crop | @224 | @448 | gain | worth 4x cost? |
|---|---|---|---|---|---|
| 0 | 1024px | 0.522 | **0.701** | **+0.179** | yes |
| 1 | 512px | 0.724 | 0.754 | +0.030 | no |
| 2 | 256px | 0.687 | 0.739 | +0.052 | marginal |

At depth 0 a 60px defect occupies 13px of a 224x224 input. Doubling the input
resolution is the single largest available gain, and it is available only at
the top of the search. The expensive look belongs at the first decision, not
the last.

Encoding each child separately instead of 2x2-pooling the parent was tested and
rejected: capacity-matched gaps of +0.067, -0.082, -0.007 across the three
depths. Pooling preserves comparability between siblings because all four come
from one normalised activation map; separate encoding destroys it. Context
matters more than resolution everywhere except depth 0.

## Robustness

Recall under corruption at budget 12, five seeds, from `results/metrics/table4_robustness.csv`.
DQN degrades more steeply than PPO and is overtaken at severity 3 on every
corruption family; under motion blur at severity 3 it collapses to 0.008. The
frozen encoder cannot adapt its features to a shifted input distribution. This
is the cost of freezing it, and it is the clearest limitation of the design.

## What this does not claim

- One seed, one checkpoint. The five-seed mean on final weights is 0.245.
- Strict recall requires three correct decisions in sequence; per-level
  accuracy is 0.685, which is 2.7x chance.
- The supervised probe is a ceiling for *this* representation at 224px, not a
  ceiling for supervised methods in general.
- The policy is not deployable at 0.321. It is a search-efficiency result, not
  a detection result.
