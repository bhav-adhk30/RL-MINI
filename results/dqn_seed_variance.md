# DQN, three seeds: variance on the frozen environment

Identical settings across all three: 12,000 episodes, lambda 0.1, 2x2 pooled
encoder (2082-512-256-6 QNetwork), `random_flip_rotate` augmentation
(training only, p=0.75), first-visit descent reward, rebalanced terminal
reward (wrong commit -1, timeout -3). Same protocol as
`results/spatial_pooling_notes.md`'s seed 0 run. Final-weights train/test
evaluation, same greedy `evaluate()` protocol, n=100 each split.

| Seed | Train recall | Train `a` | Test recall | Test cost | Test `a` | best_recall (mid-run) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.560 | 0.824 | 0.320 | 7.8 | 0.684 | 0.29 |
| 1 | 0.560 | 0.824 | 0.160 | 8.5 | 0.543 | 0.23 |
| 2 | 0.550 | 0.819 | 0.150 | 9.3 | 0.531 | 0.28 |
| **mean +/- sd** | **0.557 +/- 0.006** | **0.823 +/- 0.003** | **0.210 +/- 0.095** | **8.53 +/- 0.75** | **0.586 +/- 0.085** | **0.267 +/- 0.032** |

(sample standard deviation, n=3.)

**Read honestly, not as a single clean win.** All three seeds clear the 0.43
per-level-accuracy stop-line, and the three-seed mean (0.586) is still above
`odd_one_out` (a=0.512, fixed) -- but only seed 0 clears it comfortably
(0.684); seeds 1 and 2 land close to it (0.543, 0.531), just barely ahead.
Test recall has real spread (0.15-0.32, sd=0.095 against a mean of 0.210) --
more than double between the best and worst seed. Train recall and train
per-level accuracy are essentially seed-independent (sd 0.006 and 0.003),
which narrows the source of the variance to generalisation, not how much
the agent learns about the training set. Test cost mean (8.53) stays
comfortably below `odd_one_out`'s fixed 12.0 and exhaustive's 64 across all
three seeds -- the compute-side advantage is the more stable of the two
claims. For the report: state the claim as "beats `odd_one_out` on mean
per-level accuracy across 3 seeds, at lower and more consistent cost," not
as "beats it," since a single unlucky seed (1 or 2) would only barely clear
the baseline on accuracy alone.
