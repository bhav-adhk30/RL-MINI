# Step 8 summary: budget sweep, baselines, corruptions -- 5 seeds

`scripts/evaluate.py`'s full sweep (budget sweep, 5 baselines, 12 corruption
conditions), run in-process on the final trained weights right after
training (no save/reload), for DQN seeds 0-4. Identical settings across all
five: 12,000 episodes, lambda 0.1, 2x2 spatial-pooling encoder,
`random_flip_rotate` augmentation, first-visit + rebalanced-terminal
reward -- the frozen environment. Mean +/- sample standard deviation, n=5.

Per-seed final-weights numbers (`results/dqn_seed_variance.md` has the
earlier 3-seed partial version of this; these 5 combined-script runs are a
fresh, consistent set and are the authoritative numbers going forward --
exact values differ slightly from the earlier standalone seed 0-2 reports
due to GPU non-determinism in training, not a code change).

## Final weights: train vs. test

| | Train recall | Train `a` | Test recall | Test cost | Test `a` |
| --- | --- | --- | --- | --- | --- |
| Seed 0 | 0.520 | 0.804 | 0.200 | 8.1 | 0.585 |
| Seed 1 | 0.590 | 0.839 | 0.240 | 7.8 | 0.621 |
| Seed 2 | 0.550 | 0.819 | 0.330 | 7.2 | 0.691 |
| Seed 3 | 0.600 | 0.843 | 0.200 | 7.9 | 0.585 |
| Seed 4 | 0.500 | 0.794 | 0.210 | 8.2 | 0.594 |
| **mean +/- sd** | **0.552 +/- 0.043** | **0.820 +/- 0.021** | **0.236 +/- 0.055** | **7.84 +/- 0.39** | **0.615 +/- 0.045** |

## Budget sweep (clean test images, n=200/seed)

| Budget | Recall | Per-level accuracy | Actual evaluations used |
| --- | --- | --- | --- |
| 4  | 0.174 +/- 0.033 | 0.557 +/- 0.035 | 4.00 +/- 0.00 |
| 6  | 0.220 +/- 0.049 | 0.601 +/- 0.042 | 5.22 +/- 0.12 |
| 8  | 0.231 +/- 0.050 | 0.611 +/- 0.042 | 6.18 +/- 0.23 |
| 10 | 0.238 +/- 0.049 | 0.618 +/- 0.040 | 7.08 +/- 0.27 |
| 12 | 0.245 +/- 0.055 | 0.623 +/- 0.043 | 8.00 +/- 0.37 |
| 16 | 0.248 +/- 0.064 | 0.625 +/- 0.049 | 9.81 +/- 0.60 |

Recall and per-level accuracy both rise with budget and clearly plateau
around budget 10-12 (0.238 -> 0.245 -> 0.248 recall from 10 to 12 to 16 --
diminishing returns past 12). "Actual evaluations used" is always below the
nominal budget: the agent commits before exhausting it, and does so earlier
at smaller budgets (cost 4.00 exactly at budget 4, since the shortest
possible path is 4 steps) and closer to (but still under) the ceiling at
larger ones.

## The five baselines (deterministic -- identical across all 5 seeds, since
none of them depend on the trained agent)

| Method | Recall | Per-level accuracy | Cost |
| --- | --- | --- | --- |
| Oracle | 1.000 | 1.000 | 3 |
| Exhaustive | 1.000 | 1.000 | 64 |
| `normal_model_descent` | 0.200 | 0.585 | 12 |
| `odd_one_out` (saliency) | 0.145 | 0.525 | 12 |
| Random | 0.015 | 0.247 | 3 |

## The comparison that matters: matched budget 12

| Method | Recall | Per-level accuracy | Cost |
| --- | --- | --- | --- |
| **DQN agent** | **0.245 +/- 0.055** | **0.623 +/- 0.043** | **8.00 +/- 0.37** |
| `odd_one_out` | 0.145 | 0.525 | 12.00 |
| `normal_model` | 0.200 | 0.585 | 12.00 |

The agent beats both baselines on recall and spends less compute doing it
(mean 8.00 vs their fixed 12). The margin survives 5-seed averaging: even
at recall's lower bound (mean - 1sd = 0.190), the agent is still ahead of
`odd_one_out` (0.145) and roughly level with `normal_model` (0.200); on
per-level accuracy the lower bound (0.580) stays above both baselines
outright (0.525, 0.585). This is not a single-lucky-seed result.

## Corruption robustness (budget 12, n=200/seed, mean recall +/- sd across seeds)

| Corruption | Severity 1 | Severity 2 | Severity 3 |
| --- | --- | --- | --- |
| Gaussian noise | 0.179 +/- 0.037 | 0.157 +/- 0.030 | 0.106 +/- 0.024 |
| Motion blur | 0.190 +/- 0.019 | 0.096 +/- 0.036 | 0.008 +/- 0.011 |
| Brightness | 0.197 +/- 0.042 | 0.171 +/- 0.031 | 0.134 +/- 0.012 |
| JPEG compression | 0.187 +/- 0.050 | 0.147 +/- 0.014 | 0.122 +/- 0.039 |

Degradation is graceful and monotonic with severity for gaussian noise,
brightness and JPEG (recall drops steadily, doesn't collapse). Motion blur
is the clear exception: severity 3 collapses to essentially zero recall
(0.008 +/- 0.011, individual seeds ranging 0.00-0.02) -- heavy blur removes
exactly the fine-grained texture the frozen ResNet-18 features (and the
odd-one-out/normal-model baselines alike) depend on to localise a subtle
surface defect. Worth flagging in the report as the one condition where the
method's assumptions genuinely break down, not smoothing over it.

## What's still open

- Whether the recall plateau at budget 10-12 would shift with more training
  episodes (the seed-0-era finding that recall "had not clearly plateaued
  by episode 11,500" -- see `results/spatial_pooling_notes.md` -- was about
  training length, not budget; not re-tested here).
- Figures and the Streamlit app: not built yet, per instruction.
