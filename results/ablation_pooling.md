# Ablation: global average pooling vs. 2x2 spatial pooling

Single change, everything else held fixed (reward, actions, masking,
augmentation, epsilon, learning rate, 12,000 episodes, seed 0, lambda 0.1).
Full curves and diagnosis in `results/spatial_pooling_notes.md`; this is the
condensed comparison for the report.

| | Global avg pool (`dim=512`) | 2x2 spatial pool (`dim=2048`) |
| --- | --- | --- |
| QNetwork | 546-128-64-6 | 2082-512-256-6 |
| Test recall (final weights) | 0.080 | 0.320 |
| Test per-level accuracy `a=recall^(1/3)` | 0.431 | **0.684** |
| Test cost | 5.9 | 7.8 |
| Train recall (final weights) | 0.170 | 0.560 |
| Train per-level accuracy | 0.554 | 0.824 |
| Train/test recall gap | 0.09 | 0.24 |
| `best_recall` over the run | 0.11 | 0.29 |

**The change.** Global average pooling collapses the encoder's 7x7 spatial
map into one 512-number summary per view -- it discards *where* in the view
something is, which is exactly the information needed to tell which
quadrant holds the defect. Replacing it with `adaptive_avg_pool2d` to 2x2 on
the same `layer4` output keeps a separate 512-dim summary per quadrant
(2x2x512 = 2048), so one encoder call on a parent node yields a feature for
each of its four possible children in a single pass instead of one averaged
feature with the quadrant information washed out.

**Result.** Test per-level accuracy rose from 0.431 to 0.684 -- clear of
both the 0.43 stop-line set before this change and the `odd_one_out`
baseline's 0.512. Test recall nearly quadrupled (0.080 to 0.320).

**Two things to flag, not to hide.**

1. **The train/test gap widened**, from 0.09 (global-avg-pool run, smaller
   128-64 network) to 0.24 (spatial-pool run, larger 512-256 network) in
   recall terms. This is not the earlier memorisation failure re-emerging in
   the same shape: that pattern was train rising while test stayed flat
   (0.610 vs 0.110); here test rose in lockstep with train, and test alone
   is 4x better than the previous run. Read together with the larger
   network's larger capacity, some part of the gap is plausibly the network
   using that extra capacity to partly re-memorise specific training images
   even under augmentation, alongside a genuine generalisation gain. Worth
   watching across seeds 1 and 2, not yet a blocker.
2. **Test recall had not clearly plateaued by episode 11,500.** The last
   five logged checkpoints (9500-11500) were 0.260, 0.280, 0.240, 0.290,
   0.240 -- noisy around an upward-leaning band, not a flat settled value
   the way the earlier failed runs' recall was flat. It is plausible more
   episodes would still move this further; the run was not extended to
   check, since 12,000 was the agreed budget and the result already cleared
   the target.
