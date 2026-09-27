# Data provenance for the Step 9 figures and eval CSVs

The `results/eval_dqn_seed{0-4}_lam0.1.csv` files and the training-curve
data behind `results/learning_curves.png` and `results/steps_to_commit.png`
are transcribed from the Kaggle notebooks' printed console output, not read
from the original files on disk. This matters for two reasons, both
downstream of the same root cause.

**What happened.** All five DQN training runs (and their Step 8 sweeps)
ran in interactive Kaggle notebook sessions. `/kaggle/working/` only
persists past the end of a session if "Save Version" is clicked before the
tab closes. None of the five sessions had that done, so the actual files
(checkpoints, the full per-episode `train_log.csv`, the eval CSVs) were
lost when each session ended -- only what was printed to the console and
copied into this conversation survived. `scripts/train.py` and
`scripts/evaluate.py` were subsequently fixed (`zsrl.output_root()`, an
explicit `/kaggle/working` root, and a printed file manifest) so this
should not recur, but that fix came after these five runs.

**What this means for each file:**

- `results/eval_dqn_seed{0-4}_lam0.1.csv`: exact values, transcribed
  directly from the full Step 8 console output (budget sweep, all five
  baselines, all twelve corruption conditions) -- not reconstructed or
  approximated, just re-serialised from text into CSV.
- `runs/dqn/seed{0-4}_lam0.1/train_log.csv` (gitignored, local only):
  **reconstructed and coarser than the original.** The real per-episode
  log had one row per episode (12,000 rows); only the every-500-episode
  checkpoint printouts survived (23 rows). `return`, `final_depth` and
  `committed` in these files are the 500-episode window averages that were
  printed, not single-episode values, and `steps` uses `eval_cost` (the
  greedy policy's cost on held-out test images) rather than a training
  episode's own step count. `results/learning_curves.png` and
  `results/steps_to_commit.png` are built from this coarser data --
  correct in the numbers they show, but at 1/500th the time resolution of
  what a real run would produce.

None of this affects `results/dqn_step8_summary.md` or
`results/design_decisions.md`, which were written from the same console
output directly and don't depend on the reconstructed CSVs.

**Update: a real checkpoint was recovered.** The Kaggle session for seeds
0-4 turned out to still be alive with all five runs' output sitting in
`/kaggle/working` (they'd been run sequentially in one notebook rather than
five separate ones). A zip-and-download workaround (`IPython.display.FileLink`
didn't work via Kaggle's proxy; downloading the individual file through the
Output panel's file-preview view did) recovered `seed2_lam0.1/best.pt` --
seed 2 being the best-performing of the five (test recall 0.330, per-level
accuracy 0.691). It's committed at `runs/dqn/seed2_lam0.1/best.pt`
(force-added past the `runs/` gitignore rule, since the demo needs at least
one real checkpoint). The full per-episode `train_log.csv` files were not
part of that recovery, though -- only the checkpoints, eval CSVs (already
had those) and `normal_reference.npz` were in the zip -- so the
learning-curve/steps-to-commit figures above are still built from the
reconstructed, coarser data described below.

If a genuine full-resolution `train_log.csv` is retrieved later, or a
future training run completes with `Save Version` used correctly (the
`output_root()` fix should make plain re-runs safer regardless), it drops
into the same path and `scripts/analyse.py` needs no changes to pick it up.
