# Design decisions: reward shaping and the encoder pooling ablation

Consolidated report section. Full curves, diagnostics and raw data behind
each decision are in `results/reward_shaping_notes.md`,
`results/spatial_pooling_notes.md`, `results/full_run_seed0_notes.md`, and
`results/dqn_seed_variance.md`. This is the condensed narrative: what broke,
how it was confirmed (not assumed), and what changed.

## Part 1: reward shaping

Two reward problems were found during gate verification and training, in
this order, each confirmed with evidence before being fixed, each a
deliberate design decision recorded here rather than a bug quietly patched.

### 1a. Reward cycling (a "down-up-down" farm)

The single-image DQN overfit gate reached the optimal return (5.7) by
episode 300, but episodes 200 and 250 scored a moderately *positive* return
(5.3, 2.2) while using the full 12-step budget without ever committing.
Under the original reward table, a down-up-down cycle into the same
already-correct node paid +0.9 (descend) - 0.2 (up) + 0.9 (redescend,
paid again) = +0.7 net, repeatable for free with no localisation progress.

**Fix.** `ZoomSearchEnv` tracks `visited_nodes`, the set of nodes entered
this episode. The +-1 descent reward is paid only on a node's first visit;
a revisit pays the lambda compute cost alone. A down-up-down cycle now nets
+0.6 the first time (unaffected) but -0.3 on every repeat. The optimal
path (which never revisits a node) is unaffected -- state, actions and
masking untouched, reward-function change only.

### 1b. Refuse-to-act (a "safe do-nothing" escape)

With 1a fixed, the first full-dataset training run (2000 episodes, random
image per episode) showed `committed_rate` and `mean_depth` falling
monotonically as epsilon decayed (0.873 to 0.036, and 3.59 to 1.13), and
the greedy eval policy never committed once, on any of 100 test images, at
any of 7 checkpoints (`eval_cost` pinned at exactly 12.0 throughout). The
agent had learned to refuse the task, not to fail at it.

**The mechanism, confirmed by the numbers, not assumed.** With wrong commit
at -3, correct commit at +3, and timing out at 0:

```
E[commit] = 3p - 3(1-p) = 6p - 3
```

negative for any localisation accuracy `p < 0.5`. Timing out guaranteed 0 --
a safe escape that beats a losing gamble below 50% accuracy. The depth-1
learnability diagnostic (frozen-encoder distance from a `train/good/`
reference) measured `p = 0.575` at *one* level; compounded over three
independent levels (`p^3`) that is well under 50%, so a partly-trained
agent was *correctly* judging commit a losing bet by the reward as written,
and learning to run out the clock instead -- a cliff at p=0.5, not a
gradient, and self-reinforcing: improving localisation requires committing
and learning from the outcome, but committing was punished until
localisation was already good.

**Fix, two changes, nothing else** (state, actions, masking, descent reward
and the 1a fix untouched; learning rate and epsilon schedule untouched):

1. Wrong commit: -3 -> **-1**. Correct commit stays +3.
2. Timing out without committing: 0 -> **-3**. No free option to give up.

`E[commit] = 3p - 1(1-p) = 4p - 1`, positive for any `p > 0.25` -- break-even
accuracy **0.5 -> 0.25**, comfortably inside what a partly-trained agent can
reach. Since timing out now guarantees -3, `4p - 1 > -3` for every `p >= 0`:
attempting a commit beats refusing at *any* accuracy, not just above a
threshold. Re-running the single-image overfit gate confirmed the optimal
ceiling (5.7) was unaffected, since the optimal path never touches either
changed value. The re-run on real data confirmed the collapse was gone:
`committed_rate` stayed high (0.72-0.94) instead of falling to near-zero,
and `eval_cost` came off the 12.0 pin.

## Part 2: encoder pooling ablation

With both reward fixes in place, reframing recall as per-level accuracy
(`a = recall^(1/3)`, chance = 0.25 -- see `results/baseline_notes.md`)
showed the agent plateaued at `a = 0.431` and would not move with more
training.

| | Global avg pool (`dim=512`) | 2x2 spatial pool (`dim=2048`) |
| --- | --- | --- |
| QNetwork | 546-128-64-6 | 2082-512-256-6 |
| Test recall (final weights) | 0.080 | 0.320 |
| Test per-level accuracy | 0.431 | **0.684** (seed 0) |
| Test cost | 5.9 | 7.8 |
| Train recall (final weights) | 0.170 | 0.560 |
| Train/test recall gap | 0.09 | 0.24 |

**The mechanism.** The action being chosen is *which of four quadrants*.
Global average pooling collapses the encoder's 7x7 spatial map into one
512-number summary per view, averaging away exactly the information needed
to make that choice -- it answers "what is broadly in this view," not
"where in this view it is." Replacing it with `adaptive_avg_pool2d` to 2x2
on the same `layer4` output keeps a separate 512-dim summary per quadrant
(2x2x512=2048); one encoder call on a parent node now yields a feature for
each of its four possible children in a single pass, instead of one
averaged feature with the quadrant information washed out.

**Result.** Test per-level accuracy rose from 0.431 to 0.684 on seed 0 --
clear of the 0.43 stop-line set before this change and of the `odd_one_out`
baseline (0.512). Across all three seeds, mean test per-level accuracy is
**0.586 +/- 0.085** (see `results/dqn_seed_variance.md`) -- still ahead of
`odd_one_out` on average, though seeds 1 and 2 individually landed close to
it (0.543, 0.531); only seed 0 cleared it comfortably. Test cost stayed
below `odd_one_out`'s fixed 12.0 across all three seeds (mean 8.53 +/- 0.75).

**Flagged, not hidden.**

1. **The train/test gap widened**, from 0.09 (smaller 128-64 network) to
   0.24-0.41 (larger 512-256 network, seed 0: 0.560 vs 0.320). This is not
   the earlier memorisation failure in the same shape -- that pattern was
   train rising while test stayed flat; here test rose in lockstep with
   train, 4x higher than the smaller-network run. Plausibly some part of
   the gap is the larger network partly re-memorising specific training
   images even under augmentation, alongside a genuine generalisation gain.
   Worth watching, not yet a blocker.
2. **Test recall had not clearly plateaued by episode 11,500** on seed 0 --
   the last five checkpoints (0.260, 0.280, 0.240, 0.290, 0.240) were
   noisy around an upward-leaning band, not the flat settled value the
   failed runs showed. More episodes might move this further; the run was
   not extended past the agreed 12,000-episode budget to check, since the
   result already cleared the target.

## What this section demonstrates

Every fix here followed the same pattern: an anomaly in the numbers, a
falsifiable hypothesis, a measurement that confirmed or ruled it out before
any code changed, then the smallest change that addressed the confirmed
cause. Nothing was tuned on intuition or reverted quietly. That is the
convergence/CO4 story this project can tell in the viva.
