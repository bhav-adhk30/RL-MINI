# Environment v2 — specification

Every change below is justified by a measurement in `results/FINAL_RESULTS.md`.
Nothing here is a hypothesis. Do not add changes that are not on this list.

v1 stays frozen at tag `env-frozen-v1`. v2 gets its own tag `env-frozen-v2`,
applied BEFORE anyone starts training, because all three agents must train
against the same environment for the comparison to mean anything.

---

## 1. Action space: remove UP

**Actions become 5:** `TL, TR, BL, BR, COMMIT` (indices 0-4).

Evidence: episodes that used UP scored 0.162 recall at 10.05 steps; episodes
that never used it scored 0.467 at 4.03. Disabling UP at evaluation time on
unchanged weights raised recall 0.299 -> 0.321 and cut cost 7.36 -> 4.00.
Correlation between steps taken and success is -0.377.

Consequences:
- No node can be revisited, so the first-visit bookkeeping in `step()` goes.
- `history_len * N_ACTIONS` becomes `4 * 5 = 20`.
- Maximum episode length is `MAX_DEPTH` descents + 1 commit = 5 steps.

## 2. COMMIT becomes legal at depth >= 1

`MIN_COMMIT_DEPTH` drops from 3 to 1.

This is the point of v2. With UP gone and a forced depth-3 commit, the problem
degenerates into three independent 4-way classifications and stops being an RL
problem at all. Letting the policy choose its own commit depth restores a
genuine optimal-stopping decision: is one more descent worth the risk of
entering the wrong quadrant?

Measured trade-off the policy will be choosing along:

| commit depth | recall | region |
|---|---|---|
| 1 | 0.530 | 512px |
| 2 | 0.425 | 256px |
| 3 | 0.321 | 128px |

## 3. Reward: commit payoff scales with depth

```
correct commit at depth d   ->  +BASE ** (d - 1)        BASE default 2.0
wrong commit at any depth   ->  -1.0
descent into correct child  ->  +0.9  - cost(depth)
descent into wrong child    ->  -1.1  - cost(depth)
budget exhausted            ->  -3.0
```

where `cost(depth) = lam * units(depth)` and `units` comes from section 4.

`BASE` is the knob that selects the operating point on the precision/recall
curve. Sweep it over `{1.5, 2.0, 3.0}` and report where each lands. This is the
experiment the v1 lambda sweep failed to be: lambda modulated a per-step cost and
produced a null result, while BASE acts directly on the terminal payoff and
should move the commit depth. If it does not move it either, that is a second
null result and is reported as one.

Sanity check before training, using v1's measured conditionals
(p(d2|d1)=0.80, p(d3|d2)=0.755) and lam=0.1:

- BASE=2.0: at depth 2, commit pays 2.0; descending pays
  -0.1 + 0.755*4.0 - 0.245 = 2.68, so descending wins and the policy should
  reach depth 3.
- BASE=1.5: at depth 2, commit pays 1.5; descending pays
  -0.1 + 0.755*2.25 - 0.245 = 1.35, so committing wins and the policy should
  stop at depth 2.

If a trained policy does not behave roughly this way, the reward is not
driving the stopping decision and something is wrong — stop and debug rather
than tuning.

## 4. Encoder resolution: 448 at depth 0, 224 below

`FrozenEncoder.encode_node` picks its input size from the node's depth:

```
depth 0      -> 448x448 input,  costs 4 units
depth >= 1   -> 224x224 input,  costs 1 unit
```

Evidence, linear probe on pooled features, per-level accuracy:

| depth | crop | @224 | @448 | gain | 4x cost justified |
|---|---|---|---|---|---|
| 0 | 1024px | 0.522 | **0.701** | **+0.179** | yes |
| 1 | 512px | 0.724 | 0.754 | +0.030 | no |
| 2 | 256px | 0.687 | 0.739 | +0.052 | no |

At depth 0 a 60px defect occupies 13px of a 224 input and 26px of a 448 input.
Below depth 0 the crop is already close to native resolution and upsampling
buys almost nothing.

**Compute accounting is not optional.** A 448 pass is 4x the FLOPs of a 224
pass. Report every cost in **224-equivalent units**, with a 448 pass counted as
4. Do not report raw pass counts — it would silently make each pass four times
more expensive and the headline metric dishonest. Put that reasoning in a
comment at the counting site.

The 2x2 adaptive pooling stays exactly as it is. Encoding each child separately
was tested and rejected: capacity-matched probe gaps of +0.067, -0.082, -0.007
across depths 0, 1, 2. Pooling preserves comparability between siblings because
all four come out of one normalised activation map.

## 5. State vector

```
2048  frozen ResNet-18 features, 2x2 pooled (unchanged)
   5  depth one-hot (depths 0-4)
   4  node position and size
   1  remaining budget, normalised
  20  last four actions, one-hot over 5 actions   (was 24 over 6)
----
2078  total                                       (was 2082)
```

## 6. Training-run hygiene — v1 lost most of its evidence

| v1 problem | v2 requirement |
|---|---|
| 10 of 11 checkpoints lost | save every seed under `/kaggle/working`, download the zip before the session ends |
| `train_log.csv` condensed to 500-episode rows | write one row per episode, as `train.py` already does |
| `loss` and `wall` columns empty | populate them; Part II asks for training time and TD error |
| evaluation ran on final weights | evaluate `best.pt`; best vs final was +0.042 and one seed lost 0.10 |
| 12,000 episodes, never converged | train at least 3x longer. Episodes are ~2x shorter without UP, so this costs less wall-clock than it sounds |
| seeds not comparable across people | tag `env-frozen-v2` before anyone trains |

## 7. What must be re-measured after training

1. Recall and 224-equivalent cost at each learned commit depth, 5 seeds.
2. The BASE sweep: which commit depth each value selects.
3. The supervised probe ceiling recomputed with 448 depth-0 features — v1's
   agent (0.321) already exceeded the v1 ceiling (0.289), and the v2 ceiling
   needs recomputing before claiming anything about v2.
4. Corruption robustness. v1's frozen encoder collapsed to 0.008 under motion
   blur severity 3; 448 features may be worse, not better, since finer detail
   is more fragile. Check rather than assume.
5. Whether the agent's chosen commit depth varies per image. If it always
   picks the same depth, there is no adaptive allocation and the result should
   say so plainly.

## 8. What NOT to change

- the frozen ResNet-18 backbone, and the fact it is frozen
- 2x2 adaptive pooling
- MVTec AD used as distributed, no synthetic data
- the quadtree geometry and the masking mechanism
- the algorithm: Double DQN with experience replay stays
- the evaluation protocol and the baselines

## 9. Projection, and what counts as success

Expected: **~0.37 recall across five seeds at 6 224-equivalent units**, against
v1's 0.245 at 8 passes. Plausible range 0.30-0.45.

Derivation: 5-seed best-weights mean 0.254, x1.074 for removing UP (measured on
seed 2), x1.343 for the depth-0 resolution change (probe chain 0.349/0.260).

Success is not "recall went up". Success is a result set where every design
decision has a measurement behind it, including the ones that came out
negative. Two already have: per-child encoding was rejected, and backtracking
was found to be harmful. Report both.
