# Full 12,000-episode run, seed 0, lambda 0.1 -- stopped per the recall gate

Run with both reward fixes in place (see `results/reward_shaping_notes.md`),
no further changes to reward, state, actions or hyperparameters. GPU, ~13
minutes wall clock.

| ep | train_return | committed_rate | mean_depth | eval_recall | eval_cost | explore |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | -5.81 | 0.872 | 3.51 | 0.000 | 11.9 | 0.816 |
| 1000 | -5.58 | 0.794 | 3.14 | 0.000 | 11.9 | 0.625 |
| 1500 | -5.11 | 0.636 | 2.57 | 0.000 | 11.8 | 0.409 |
| 2000 | -4.29 | 0.570 | 2.24 | 0.020 | 9.5 | 0.184 |
| 2500 | -1.86 | 0.720 | 2.46 | 0.070 | 7.8 | 0.050 |
| 3000 | -1.06 | 0.814 | 2.59 | 0.080 | 8.6 | 0.050 |
| 3500 | -0.98 | 0.796 | 2.66 | 0.070 | 7.0 | 0.050 |
| 4000 | -0.22 | 0.850 | 2.74 | 0.040 | 8.7 | 0.050 |
| 4500 | 0.41 | 0.832 | 2.72 | 0.090 | 8.4 | 0.050 |
| 5000 | 0.28 | 0.816 | 2.71 | 0.100 | 8.1 | 0.050 |
| 5500 | 0.69 | 0.862 | 2.74 | 0.080 | 9.7 | 0.050 |
| 6000 | 1.41 | 0.866 | 2.81 | 0.110 | 7.5 | 0.050 |
| 6500 | 1.81 | 0.882 | 2.81 | 0.090 | 9.7 | 0.050 |
| 7000 | 1.09 | 0.822 | 2.68 | 0.090 | 10.1 | 0.050 |
| 7500 | 1.93 | 0.862 | 2.77 | 0.080 | 9.1 | 0.050 |
| 8000 | 1.96 | 0.874 | 2.82 | 0.100 | 10.2 | 0.050 |
| 8500 | 1.79 | 0.874 | 2.81 | 0.090 | 9.9 | 0.050 |
| 9000 | 1.60 | 0.874 | 2.82 | 0.060 | 9.9 | 0.050 |
| 9500 | 2.51 | 0.914 | 2.91 | 0.090 | 10.2 | 0.050 |
| 10000 | 2.40 | 0.898 | 2.85 | 0.030 | 10.0 | 0.050 |
| 10500 | 2.43 | 0.888 | 2.83 | 0.060 | 10.6 | 0.050 |
| 11000 | 2.48 | 0.882 | 2.83 | 0.070 | 11.3 | 0.050 |
| 11500 | 2.49 | 0.884 | 2.82 | 0.090 | 9.8 | 0.050 |

`best_recall` over the whole run: 0.11 (episode 6000). Final episode index is
11999 (0-indexed, `range(12000)`), so ep 11500 is the last logged checkpoint
and stands in for "at 12,000."

## Judged against the four criteria set before this run

1. **recall clearly beats random (0.022), approaches/passes odd_one_out
   (0.134).** Half true: settled-range average recall is ~0.078, comfortably
   above random, but it never gets close to odd_one_out's 0.134 -- it
   oscillates in a 0.03-0.11 band for the entire second half of training
   with no clear upward trend past ep 6000.
2. **greedy mean_depth >= 3.** Not cleanly measurable from what this run
   logged: `mean_depth` here is a training-side rolling average that still
   includes the residual 5% epsilon-exploration, not a pure-greedy number.
   It settles around 2.7-2.9, never reaching 3. `scripts/train.py`'s
   `evaluate()` only returns `(recall, mean_cost)` -- it does not currently
   report eval-side mean depth, which is a real gap for judging this
   criterion specifically and should be added before the next run.
3. **eval_cost settles well below 12.** Partially: it dips to 7.0-8.7
   through ep 3500, then drifts back up to 9.7-11.3 by ep 7500-11500 -- a
   dip-then-partial-climb, not a clean settle.
4. **committed_rate stays high at the floor.** True: 0.72-0.91 throughout
   the settled range, consistent with the refuse-to-act fix holding.
5. **Stop condition: recall >= 0.10 at 12,000, epsilon at floor.** Not met.
   Last checkpoint (ep 11500) is 0.090.

## Decision

Per the pre-agreed stop condition, this is treated as a genuine capacity or
state problem, not something to fix by touching the reward again (two
reward changes already made, both evidence-driven; a third now would be
fitting to noise). No further reward, state, action or hyperparameter
changes made. Seeds 1 and 2 not run. Investigation now turns to the
features/state representation rather than the reward function.

## Follow-up: train_return climbing while eval_recall stays flat -- memorisation

Looking at train_return alongside eval_recall from the table above:

| Episode | train_return | eval_recall |
| --- | --- | --- |
| 2500 | -1.86 | 0.070 |
| 5000 | 0.28 | 0.100 |
| 7500 | 1.93 | 0.080 |
| 10000 | 2.40 | 0.030 |
| 11500 | 2.49 | 0.090 |

train_return rises steadily from -5.81 to +2.49 across the whole run while
eval_recall stays flat around 0.09 from episode 2500 onward (9000+ episodes
with no real movement). That divergence -- return improving while held-out
recall does not -- is the standard signature of memorising the training set
rather than learning a transferable search policy. The state's 512-dim
frozen encoder feature is close to a per-image fingerprint; with ~530
distinct training images seen roughly 20x each over 12,000 episodes, a
546-512-256 (~300k parameter) Q-network has more than enough capacity to
learn "this specific image -> this specific descent path" instead of
"defects tend to look like X."

**Confirmed, not assumed.** Loaded `best.pt` and ran the exact same greedy
`evaluate()` protocol (n=100, `index=i % len(canvases)`) on the train split
instead of test:

| Split | Recall | Cost |
| --- | --- | --- |
| Train (first 100 train images) | **0.610** | 6.0 |
| Test (first 100 test images) | 0.110 | 7.5 |

Train recall (0.610) is far above the 0.4 threshold set in advance to call
this overfitting, versus test recall (0.110) -- confirms memorisation, not
a capacity/feature problem (which would have shown both splits near 0.1).

**Fix, two changes, nothing else** (reward, state layout, action space,
epsilon and learning rate all unchanged):

1. **Augmentation, training only.** `ZoomSearchEnv` gained an `augment`
   flag (default `False`, so eval is untouched). When `True`, `reset()`
   applies `random_flip_rotate`: with probability 0.75, one of {horizontal
   flip, vertical flip, 90/180/270 degree rotation} chosen uniformly, applied
   identically to the image and its ground-truth box. The quadtree
   partitions the canvas into equal quadrants, so it is symmetric under
   these transforms -- this re-presents the same real image with its real
   defect in a different quadrant, not synthesised data (`box_from_mask`'s
   ground truth is still exactly derived from MVTec's own mask, just
   geometrically re-oriented along with the image it came from). Every box
   formula was verified against actual PIL output on a synthetic test image
   before use (all 5 transforms matched exactly). One subtlety: the
   encoder's cache key now includes which transform was applied (or `"id"`
   for none), since augmented views of the same source image are different
   crops at a given quadtree node and must not collide in the cache.
2. **Network capacity.** `QNetwork` shrunk from `546-512-256-6`
   (~300k params) to `546-128-64-6` (~74k params), removing memorisation
   headroom directly.

**Re-run of the full 12,000-episode run, seed 0, lambda 0.1, with both
changes** (augment=True training only, QNetwork 546-128-64-6):

| ep | train_return | committed_rate | mean_depth | eval_recall | eval_cost |
| --- | --- | --- | --- | --- | --- |
| 500 | -5.86 | 0.874 | 3.51 | 0.000 | 12.0 |
| 1000 | -5.48 | 0.850 | 3.25 | 0.000 | 11.1 |
| 1500 | -4.97 | 0.862 | 3.07 | 0.010 | 9.0 |
| 2000 | -4.52 | 0.880 | 2.96 | 0.020 | 7.5 |
| 2500 | -4.25 | 0.782 | 2.61 | 0.030 | 7.8 |
| 3000 | -3.87 | 0.724 | 2.24 | 0.030 | 6.4 |
| 3500 | -3.25 | 0.866 | 2.65 | 0.040 | 4.8 |
| 4000 | -3.47 | 0.794 | 2.45 | 0.020 | 7.5 |
| 4500 | -3.11 | 0.856 | 2.67 | 0.030 | 6.7 |
| 5000 | -3.04 | 0.900 | 2.76 | 0.070 | 6.2 |
| 5500 | -3.28 | 0.818 | 2.54 | 0.040 | 5.6 |
| 6000 | -2.70 | 0.914 | 2.79 | 0.030 | 5.3 |
| 6500 | -2.93 | 0.874 | 2.69 | 0.020 | 7.0 |
| 7000 | -2.67 | 0.942 | 2.87 | 0.040 | 5.2 |
| 7500 | -2.81 | 0.854 | 2.63 | 0.040 | 7.1 |
| 8000 | -2.99 | 0.854 | 2.63 | 0.060 | 6.9 |
| 8500 | -2.57 | 0.878 | 2.69 | 0.050 | 6.6 |
| 9000 | -2.45 | 0.864 | 2.67 | 0.030 | 6.3 |
| 9500 | -2.55 | 0.900 | 2.75 | 0.030 | 7.6 |
| 10000 | -2.63 | 0.872 | 2.69 | 0.070 | 5.6 |
| 10500 | -2.46 | 0.914 | 2.80 | 0.070 | 6.3 |
| 11000 | -2.22 | 0.908 | 2.80 | 0.050 | 6.3 |
| 11500 | -2.20 | 0.898 | 2.75 | 0.070 | 7.1 |

**Final weights, train-vs-test** (same greedy `evaluate()` protocol, n=100):

| Split | Recall | Cost |
| --- | --- | --- |
| Train | 0.170 | 5.5 |
| Test | 0.080 | 5.9 |

**Memorisation: confirmed fixed.** The train/test recall gap shrank from
0.610/0.110 (delta 0.50) to 0.170/0.080 (delta 0.09). The fix worked as
diagnosed.

**Reframed as per-level accuracy** (`a = recall^(1/3)`, chance = 0.25; see
`results/baseline_notes.md` for the full table and reasoning): test recall
0.080 is `a = 0.431`, against `odd_one_out` (recall 0.134, cost 12, 4
evaluations/level) at `a = 0.512`. Per evaluation, the agent (~1 encoder
call/level) is much closer to competitive than the raw recall numbers
suggest -- the headline gap is inflated by cubing a smaller per-level gap.

**Target (recall > 0.134 at cost <= 12, or ~0.134 at cost clearly below 12):
not met.** eval_recall settles in a 0.02-0.07 band for essentially the
whole run, ending at 0.070-0.080, well under odd_one_out's 0.134. Two other
things changed alongside the recall plateau, worth noting rather than
acting on: `eval_cost` dropped substantially and consistently (settling
~5-7, versus ~9-11 in the un-augmented run) -- the agent commits faster and
cheaper, just not more accurately -- and `train_return` never crosses zero
this run (stays -5.86 to -2.20 throughout), unlike the un-augmented run
where it climbed to +2.49. `committed_rate` (0.72-0.94) and `mean_depth`
(plateauing ~2.6-2.8) look similar to the un-augmented run's settled range.
Overfitting is genuinely resolved, but the fix traded it for a harder
optimisation problem (8x more visual variety per image via augmentation,
4x less network capacity) rather than for the recall gain the target
needed. No further changes made pending direction.
