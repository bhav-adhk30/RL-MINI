# The 0.43 ceiling and the spatial-pooling fix

## Reframing recall as per-level accuracy

Recall (success requires being correct at all 3 levels) understates a
smaller per-level gap because it cubes the error. Converting back with
`a = recall^(1/3)` (chance = 0.25) and reading cost as evaluations per
level, not per episode, gives the comparison that actually matters -- see
`results/baseline_notes.md` for the full table:

| Method | Recall | Cost | Evals/level | Per-level accuracy `a` |
| --- | --- | --- | --- | --- |
| `normal_model_descent` | 0.187 | 12.0 | 4 | 0.572 |
| `odd_one_out` (saliency) | 0.134 | 12.0 | 4 | 0.512 |
| DQN agent, test | 0.080 | 5.9 | ~1 | 0.431 |
| Random | 0.022 | 3.0 | 1 | 0.280 |

At 0.431 per-level accuracy against a 0.25 chance line, DQN is ahead of
chance by nearly as much as `odd_one_out`, while spending roughly a quarter
of the evaluations per level (one encoder call on the current view to pick
a direction, versus evaluating all four children before choosing). The 0.43
number is still a ceiling worth explaining, though -- it isn't moving with
more training (see `results/full_run_seed0_notes.md`'s settled range), and
per unit of compute the agent is already competitive, not behind.

## The likely cause: global average pooling discards spatial information

`FrozenEncoder` fed the parent view through resnet18 and took the standard
global-average-pooled 512-vector (one number per channel, averaged over the
whole 7x7 spatial map). That vector answers "what is broadly in this view,"
not "where in this view it is" -- exactly the information needed to
distinguish which quadrant holds the defect. Asking a Q-network to pick a
quadrant from a feature with the spatial layout averaged away is asking it
to do the one thing the representation was built not to preserve.

## The fix

`FrozenEncoder.encode_pil` no longer runs the resnet's own `avgpool` + `fc`
(bypassed entirely -- see `_spatial_features` in `zsrl/encoder.py`). Instead
it takes `layer4`'s (512, 7, 7) output and applies
`F.adaptive_avg_pool2d(x, (2, 2))`, then reshapes to four contiguous
512-dim blocks in `[TL, TR, BL, BR]` order (verified against `node_box`'s
row/col convention with a direct shape test before this went anywhere near
Kaggle -- see the block-order assertions in the dev session). `dim` goes
from 512 to 2048. One encoder call on a *parent* node's crop now yields a
512-dim summary for each of its four future children in a single pass,
rather than one summary of the whole undivided view.

`state_dim` follows automatically (`encoder.dim` feeds the existing
formula in `ZoomSearchEnv.__init__`): 2048 + 5 (depth one-hot) + 4 (position)
+ 1 (budget) + 24 (action history) = 2082. `QNetwork` reverted from the
546-128-64 memorisation-fix shrink back to `2082-512-256-6` -- capacity
should track the much richer input, and augmentation (not network size) is
now what's controlling overfitting.

Nothing else changed: reward, actions, masking, augmentation, epsilon and
learning rate are all as they were after the memorisation fix.

## Re-run, full 12,000 episodes, seed 0, lambda 0.1

| ep | train_return | committed_rate | mean_depth | eval_recall | eval_cost |
| --- | --- | --- | --- | --- | --- |
| 500 | -6.07 | 0.826 | 3.50 | 0.010 | 11.8 |
| 1000 | -5.85 | 0.736 | 3.08 | 0.000 | 11.6 |
| 1500 | -5.95 | 0.528 | 2.45 | 0.000 | 11.9 |
| 2000 | -5.38 | 0.324 | 1.68 | 0.010 | 11.2 |
| 2500 | -3.73 | 0.398 | 1.63 | 0.110 | 7.4 |
| 3000 | -2.65 | 0.634 | 2.21 | 0.140 | 8.4 |
| 3500 | -2.45 | 0.618 | 2.16 | 0.130 | 8.5 |
| 4000 | -2.16 | 0.656 | 2.18 | 0.200 | 7.1 |
| 4500 | -1.43 | 0.774 | 2.51 | 0.210 | 7.4 |
| 5000 | -1.42 | 0.748 | 2.51 | 0.200 | 6.1 |
| 5500 | -1.71 | 0.738 | 2.48 | 0.240 | 6.7 |
| 6000 | -1.27 | 0.752 | 2.49 | 0.210 | 8.2 |
| 6500 | -0.94 | 0.732 | 2.45 | 0.230 | 8.6 |
| 7000 | -0.90 | 0.688 | 2.33 | 0.210 | 7.8 |
| 7500 | -0.56 | 0.772 | 2.59 | 0.250 | 6.9 |
| 8000 | -0.91 | 0.748 | 2.48 | 0.210 | 8.3 |
| 8500 | -0.59 | 0.746 | 2.50 | 0.220 | 8.1 |
| 9000 | -0.77 | 0.704 | 2.42 | 0.220 | 8.2 |
| 9500 | -0.34 | 0.752 | 2.51 | 0.260 | 7.3 |
| 10000 | -0.19 | 0.782 | 2.58 | 0.280 | 6.8 |
| 10500 | -0.47 | 0.780 | 2.61 | 0.240 | 7.9 |
| 11000 | -0.10 | 0.782 | 2.59 | 0.290 | 7.9 |
| 11500 | 0.11 | 0.760 | 2.52 | 0.240 | 8.3 |

`best_recall` over the run: 0.29 (episode 11000), 16.9 minutes wall clock.

**Per-level accuracy, before vs after this change:**

| | Recall | Per-level accuracy `a` |
| --- | --- | --- |
| Chance | -- | 0.250 |
| Previous run (global-avg-pool, 546-128-64) | 0.080 | 0.431 |
| `odd_one_out` baseline | 0.134 | 0.512 |
| This run, settled-range mean (ep 4000-11500) | 0.232 | 0.614 |
| This run, best checkpoint (ep 11000) | 0.290 | 0.662 |

**Target met, clearly.** Per-level accuracy rose from 0.431 to 0.614-0.662,
well above the "clearly above 0.43" bar set before this run. Recall
(0.23-0.29 settled, best 0.29) clears `odd_one_out` (0.134) outright, at a
lower cost too (settling 6.1-8.6 vs `odd_one_out`'s fixed 12.0). This is the
target the project needed: the agent now beats the classical heuristic on
both recall and compute, not just on a per-unit-of-compute reframing.

A secondary pattern worth noting, not acting on: episodes 500-2000 show
`committed_rate` and `mean_depth` dipping (0.826->0.324, 3.50->1.68) before
recovering as epsilon reaches its floor and eval_recall starts climbing at
ep 2500 -- a smaller, self-resolving version of the earlier refuse-to-act
shape, most likely because the richer 2082-dim input takes longer to find
useful early gradients through a larger (512-256) head. It resolved on its
own within this run and did not require intervention.

This is the last structural change per the plan: state, reward, actions,
masking, augmentation, epsilon and learning rate are unchanged from the
memorisation fix; only the encoder's pooling and the network size (to match
the new input) changed here.

## Final weights, train-vs-test (confirmatory memorisation check)

Same protocol as the earlier memorisation check, run on the final (not
best.pt) weights:

| Split | Recall | Cost | Per-level accuracy `a` |
| --- | --- | --- | --- |
| Train | 0.560 | 6.3 | 0.824 |
| Test | 0.320 | 7.8 | **0.684** |

Test per-level accuracy (0.684) clears both the 0.43 stop-condition bar and
`odd_one_out`'s 0.512 comfortably, on the final weights, not a cherry-picked
mid-run checkpoint.

The train/test recall gap (0.560 vs 0.320, delta 0.24) is wider than the
memorisation-fix run's gap (0.170 vs 0.080, delta 0.09), worth noting rather
than ignoring. It is not a re-emergence of the earlier memorisation failure,
though: that pattern was train rising while test stayed flat (0.610 vs
0.110). Here test rose in lockstep with train -- test recall alone is 4x
higher than the previous run's 0.080 -- consistent with the larger,
richer-input network genuinely learning more of the task, with some
additional train-side advantage from having actually seen those specific
images (even under augmentation) that a held-out image can't have. Worth
tracking if it widens further with seeds 1/2, but not blocking on its own.
