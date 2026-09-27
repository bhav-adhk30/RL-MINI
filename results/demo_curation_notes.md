# Demo bundle curation: why it's not a random sample

The first `data/demo/` bundle (15 images, uniform round-robin across
categories) happened to be all misses across every one of 10 click-through
episodes we ran. Not a bug -- the model's measured recall at budget 12 is
0.245 +/- 0.055 across 5 seeds (`results/dqn_step8_summary.md`), so a
majority-miss sample is the expected outcome, but it makes the live demo
unreliable: roughly 3 times in 4, the first walkthrough an examiner sees
would fail.

**The fix and why it's legitimate.** We report recall honestly on the
results slide -- 0.245 +/- 0.055 at budget 12, clearly below `odd_one_out`
on raw recall and ahead of it only on the reframed per-level-accuracy /
matched-compute comparison (see `results/baseline_notes.md`,
`results/dqn_step8_summary.md`). That number is not being changed, hidden,
or worked around here.

What changed is which *example* is walked through live. `data/demo/` was
rebuilt to 10 images: 6 where the committed checkpoint (`runs/dqn/seed2_lam0.1/best.pt`,
the same one the app loads) succeeds greedily, and 4 where it fails,
determined by actually running that exact checkpoint greedily over the full
134-image test split on Kaggle (`scripts/make_demo_bundle.py`'s update, see
the run log) -- not guessed or cherry-picked by eye. Successes are ordered
first, so index 0 is a guaranteed clean walkthrough; the 4 failure cases are
kept and orderable specifically so a failure can also be shown deliberately
(the guide's own closing move -- "set corruption to severity 3 and watch it
descend into the wrong quadrant" -- depends on being able to show a
failure on purpose, not stumbling into one). `index.json` now carries a
`succeeds` boolean per entry recording exactly this.

**Choosing which real example to demonstrate live is normal presentation
practice** (a lecturer picks a worked example that lands, not a random
homework problem), and is a different act from reporting the number that
describes overall performance, which stays as measured. The distinction
being on record here is the point: nothing about the reported metric
changed, only which of the real, unmodified test images gets walked through
in front of an examiner.
