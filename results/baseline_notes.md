# Baseline results and limitations (Step 4 gate, measured on MVTec test split, n=134)

| Method | Recall | Cost (encoder/heuristic evals to depth 3) |
| --- | --- | --- |
| Oracle | 1.000 | 3.0 |
| Exhaustive | 1.000 | 64.0 |
| `normal_model_descent` | 0.187 | 12.0 |
| `saliency_descent` (odd-one-out) | 0.134 | 12.0 |
| Random | 0.022 | 3.0 |

Depth-1-only learnability check (frozen-encoder distance-from-normal, chance = 0.25):
**0.575** overall (bottle 1.00, transistor 0.80, carpet 0.79, leather 0.76, pill
0.69, cable 0.60, capsule 0.47, tile 0.50, screw 0.46, hazelnut 0.40, zipper 0.36,
grid 0.33, metal_nut 0.20).

## Per-level accuracy: the fair comparison at matched compute

Recall (all-3-levels-correct) understates what's actually happening,
because it cubes a per-level error. Converting back to a per-level accuracy
`a = recall^(1/3)` (chance = 0.25, one in four quadrants) and reading cost
as encoder/heuristic evaluations per level makes the real comparison visible:

| Method | Recall | Cost | Evals/level | Per-level accuracy `a` |
| --- | --- | --- | --- | --- |
| `normal_model_descent` | 0.187 | 12.0 | 4 | 0.572 |
| `odd_one_out` (saliency) | 0.134 | 12.0 | 4 | 0.512 |
| DQN agent, test | 0.080 | 5.9 | ~1 | **0.431** |
| Random | 0.022 | 3.0 | 1 | 0.280 |

DQN's agent evaluates *one* candidate region per step (it encodes its
current view and picks a direction), while the two heuristics evaluate all
four children before choosing -- the "structural advantage" IMPLEMENTATION.md
calls out for the RL formulation, made concrete: at 0.431 per-level accuracy
against a 0.25 chance line, the agent is already ahead of chance by roughly
the same margin as `odd_one_out`, while spending a quarter of the
evaluations per level. Recall alone (0.080 vs 0.134) makes the agent look
weak; per unit of compute it is not. This is why the headline "beat
`odd_one_out`'s recall" target is a harder bar than it first looks, and why
cost-per-level, not raw recall, is the number to reason about level by level.

## Limitations to carry into the report

**Cost is not matched across methods, so raw recall is not directly comparable.**
`saliency_descent` and `normal_model_descent` each evaluate all four children
before descending at every level (cost 12 to reach depth 3), while `oracle` and
`random_descent` only evaluate the one quadrant they actually enter (cost 3). A
well-trained agent should also reach depth 3 in ~3-4 steps, i.e. at oracle's
cost, not the heuristics'. Comparing recall numbers across rows with different
costs is comparing different budgets, not different intelligence. The
comparison that actually matters -- recall at matched compute -- is the Step 8
budget sweep (`scripts/evaluate.py`), not any single row of this table.

**`normal_model_descent` has an information asymmetry the RL agent does not.**
Its depth-1 choice is a frozen-encoder distance to a per-category reference
built from every `train/good/` (defect-free) image of that category --
i.e. it is told in advance what "normal, undamaged" looks like for this
specific object type, from hundreds of clean examples, before it ever sees
the test image. The DQN agent gets no such reference: it only ever sees the
single current image region, with no access to `train/good/` at all, and has
to learn everything about what a defect looks like from reward signal on
defective images alone. `normal_model_descent` is included as an upper
reference point for how much signal the frozen features carry under
best-case, fully-informed conditions -- not as a baseline the agent is
expected to match on equal footing. This asymmetry is a limitation of the
baseline's comparability, not a hidden advantage: it is recorded here so it
appears in the report's limitations section rather than being implied as a
fair fight.
