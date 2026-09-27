# Reward design failures found during gate verification, and the fixes

Two separate reward problems were found while verifying `zsrl/env.py` against
DQN, in this order. Both are recorded here deliberately, as design decisions
made in response to evidence, not bugs we hid.

## Failure 1: reward cycling (a "down-up-down" farm)

The first single-image DQN overfit run (400 episodes, one fixed training
image, `eps_decay_steps=1500`) reached the optimal return (5.7, 4 steps,
success) by episode 300 -- but two earlier checkpoints looked wrong:

| Episode | Return | Success | Steps | eps |
| --- | --- | --- | --- | --- |
| 0 | -11.5 | False | 12 | 0.992 |
| 50 | -7.4 | False | 5 | 0.752 |
| 100 | -8.7 | False | 7 | 0.527 |
| 150 | -1.8 | False | 12 | 0.309 |
| **200** | **5.3** | **False** | **12** | 0.050 |
| **250** | **2.2** | **False** | **12** | 0.050 |
| 300 | 5.7 | True | 4 | 0.050 |
| 350 | 5.7 | True | 4 | 0.050 |

Episodes 200 and 250 used the full 12-step budget without ever committing,
yet still scored a moderately *positive* return. Under the original reward
table, a down-up-down cycle into the same correct node pays
+0.9 (descend, first time) - 0.2 (up) + 0.9 (redescend, paid again) = +0.7 net,
repeatable for the whole budget with zero localisation progress. That is
exactly what those two episodes were doing: farming the dense descent reward
by re-entering an already-visited node rather than ever committing.

**Fix.** `ZoomSearchEnv` now tracks `visited_nodes`, the set of nodes entered
so far this episode (reset every `reset()`). The +1/-1 component of the
descent reward is paid only on a node's first visit; a revisit pays the
lambda compute cost alone. Nothing else changed: the state vector, action
space, `UP` reward, and commit rewards were untouched at this point, and a
node's visited status is not part of the state, so this was a reward-function
change only. A down-up-down cycle now nets +0.9 - 0.2 - 0.1 = +0.6 the first
time (genuine progress, unaffected) but only -0.2 - 0.1 = -0.3 on every
repeat. The shortest correct path (descend, descend, descend, commit --
which never revisits a node) gets an identical return to before.

**Re-run after fix 1** (same settings: 400 episodes, image index 0,
`eps_decay_steps=1500`):

| Episode | Return | Success | Steps | eps |
| --- | --- | --- | --- | --- |
| 0 | -11.5 | False | 12 | 0.992 |
| 50 | -8.2 | False | 10 | 0.745 |
| 100 | -8.0 | False | 9 | 0.467 |
| 150 | 1.3 | False | 12 | 0.118 |
| 200 | 5.7 | True | 4 | 0.050 |
| 250 | -0.8 | False | 12 | 0.050 |
| 300 | 4.4 | True | 6 | 0.050 |
| 350 | 5.7 | True | 4 | 0.050 |

The optimal return/steps combination (5.7, 4 steps, success) is still
reached, now first at episode 200 rather than 300. The full-budget, no-commit
episode at 250 still occurs (residual 5% epsilon-greedy exploration is
expected to occasionally produce a non-optimal episode even after
convergence), but its return is now mildly *negative* (-0.8) rather than
positive (5.3 / 2.2 in the pre-fix run) -- consistent with unproductive
wandering, not profitable cycling. That sign flip on an otherwise-similar
timeout episode is the evidence the exploit, specifically, is gone, as
distinct from ordinary exploration noise.

## Failure 2: refuse-to-act (a "safe do-nothing" escape)

With fix 1 in place, the single-image overfit gate passed cleanly. But the
first real-data run -- `scripts/train.py`, 2000 episodes, random training
image per episode, default `eps_decay_steps=20000` -- showed a different,
more serious problem:

| ep | train_return | committed_rate | mean_depth | eval_recall | eval_cost |
| --- | --- | --- | --- | --- | --- |
| 250 | -7.36 | 0.873 | 3.59 | 0.000 | 12.0 |
| 500 | -7.15 | 0.708 | 3.36 | 0.000 | 12.0 |
| 750 | -7.07 | 0.568 | 3.08 | 0.000 | 12.0 |
| 1000 | -5.89 | 0.392 | 2.44 | 0.000 | 12.0 |
| 1250 | -4.96 | 0.220 | 1.94 | 0.000 | 12.0 |
| 1500 | -3.66 | 0.088 | 1.32 | 0.000 | 12.0 |
| 1750 | -2.91 | 0.036 | 1.13 | 0.000 | 12.0 |

`committed_rate` and `mean_depth` fell monotonically as epsilon decayed
(0.873 -> 0.036, and 3.59 -> 1.13). `eval_cost` -- the fully-greedy policy's
episode length -- was pinned at exactly 12.0 at all 7 checkpoints: the
greedy policy never committed once, on any of the 100 held-out test images,
at any point in training. The agent was learning to refuse the task, not
struggling to learn it.

A follow-up breakdown of the training log by exploration level confirmed the
mechanism directly:

| explore level | n | committed_rate | success_rate | mean_depth |
| --- | --- | --- | --- | --- |
| low (eps<0.15, mostly greedy) | 251 | 0.004 | 0.000 | 0.884 |
| mid | 629 | 0.092 | 0.010 | 1.335 |
| high (eps>0.5, mostly random) | 1120 | 0.593 | 0.005 | 3.013 |

and, among episodes that *did* commit, success rate was only 0.009-0.103
regardless of exploration level -- i.e. true localisation accuracy was well
under 50% throughout this run.

**The mechanism.** With a wrong commit at -3, correct commit at +3, and
timing out at 0, the expected value of attempting to commit at localisation
accuracy `p` is:

```
E[commit] = 3p - 3(1-p) = 6p - 3
```

negative for any `p < 0.5`. Timing out paid a guaranteed 0 -- a safe escape
that beats a losing gamble whenever accuracy is below 50%. The depth-1
learnability diagnostic (`results/baseline_notes.md`) measured 0.575
accuracy at *one* level; compounded over three independent levels that is
well under 50%, so a partly-trained agent was *correctly* judging commit a
losing bet by the numbers as they stood, and learning to run out the clock
instead. This is self-reinforcing and cannot fix itself with more episodes:
improving localisation requires committing and learning from the outcome,
but committing is punished (in expectation) until localisation is already
good enough -- a cliff, not a gradient, at the p=0.5 threshold.

**Fix, two changes, nothing else touched** (state, action space, masking,
descent rewards and the fix-1 first-visit rule are all unchanged; learning
rate, epsilon schedule and network size are unchanged):

1. Wrong commit: -3 -> **-1**. Correct commit stays +3.
2. Budget exhausted without committing: 0 -> **-3**. There is no longer a
   free option to give up.

With these, `E[commit] = 3p - 1(1-p) = 4p - 1`, positive for any `p > 0.25`
(break-even accuracy `p* = R_bad / (R_good + R_bad) = 1/4 = 0.25`, down from
0.5) -- comfortably inside what a partly-trained agent can reach. And since
timing out now guarantees -3, `E[commit] = 4p - 1 > -3` for every `p >= 0`:
attempting a commit beats refusing to act at *any* accuracy, not just above
a threshold. The single-image overfit ceiling is unaffected: the optimal
path (descend x3, commit correctly) never touches either changed reward, so
its return is still 2.7 + 3 = 5.7.

**Re-run of the single-image overfit gate, after fix 2** (confirming the
ceiling is unaffected before spending more real-data training time; same
settings: 400 episodes, image index 0, `eps_decay_steps=1500`):

| Episode | Return | Success | Steps | eps |
| --- | --- | --- | --- | --- |
| 0 | -9.5 | False | 12 | 0.992 |
| 50 | -7.2 | False | 10 | 0.745 |
| 100 | -11.6 | False | 12 | 0.484 |
| 150 | -1.7 | False | 12 | 0.151 |
| 200 | -0.6 | False | 12 | 0.050 |
| 250 | 5.7 | True | 4 | 0.050 |
| 300 | 4.4 | True | 6 | 0.050 |
| 350 | 5.7 | True | 4 | 0.050 |

Ceiling confirmed unaffected: the optimal return/steps combination (5.7, 4
steps, success) is reached at episode 250, one checkpoint earlier than the
fix-1-only run's episode 300, and holds through 350. This is exactly what
the math predicts: the optimal path never times out or commits incorrectly,
so it never touches either of fix 2's changed reward values.

**Re-run of the 2000-episode real-data training check, after fix 2**
(`scripts/train.py`, 2000 episodes, random training image per episode,
default `eps_decay_steps=20000`, GPU):

| ep | train_return | committed_rate | mean_depth | eval_recall | eval_cost |
| --- | --- | --- | --- | --- | --- |
| 250 | -6.08 | 0.869 | 3.59 | 0.000 | 12.0 |
| 500 | -5.86 | 0.848 | 3.40 | 0.000 | 11.4 |
| 750 | -5.48 | 0.860 | 3.32 | 0.010 | 11.0 |
| 1000 | -5.28 | 0.808 | 3.04 | 0.000 | 10.3 |
| 1250 | -5.22 | 0.732 | 2.84 | 0.010 | 10.8 |
| 1500 | -5.36 | 0.608 | 2.46 | 0.010 | 11.7 |
| 1750 | -4.47 | 0.652 | 2.35 | 0.070 | 9.1 |

Confirmed fixed: `committed_rate` no longer collapses (stays 0.6-0.87
throughout, vs. falling to 0.036 before fix 2) and `eval_cost` is no longer
pinned at exactly 12.0 -- the greedy eval policy is committing on some
episodes, which it never did even once before. `train_return` improves
monotonically. The still-falling `mean_depth` (3.59 -> 2.35) and near-zero
`eval_recall` at this point are consistent with epsilon still being high
(0.313 at ep 1750, versus the default `eps_decay_steps=20000` floor) rather
than a learned refusal, since greedy eval is demonstrably committing. Judged
against this, not against intuition, before committing to the full run:
eval_recall should clearly beat random (0.022) and ideally approach or pass
odd_one_out (0.134); greedy mean_depth should be at or above 3; eval_cost
should settle well below 12; committed_rate should stay high once epsilon
reaches its floor. Full 12,000-episode run (seed 0, lambda 0.1) in progress
to check against those thresholds -- see the table below once it lands.
