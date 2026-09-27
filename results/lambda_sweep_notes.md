# Lambda sweep: does it trace a monotone frontier?

**No.** Neither cost nor recall is monotone in lambda, and the effect (if
any) is not distinguishable from ordinary seed-to-seed noise with the data
collected.

| Lambda | Recall | Cost | Seed |
| --- | --- | --- | --- |
| 0.0 | 0.270 | 8.06 | 0 |
| 0.1 | 0.200 | 8.10 | 0 |
| 0.3 | 0.230 | 8.19 | 0 |
| 0.6 | 0.310 | 6.96 | 0 |

Cost does *not* decrease monotonically as lambda rises (it rises slightly
from 0.0 to 0.3, then drops at 0.6). Recall does not move monotonically
either. Naively read, lambda=0.6 looks best on both axes simultaneously --
exactly the opposite of the guide's expectation that a higher per-descent
penalty should trade recall for a cheaper, more decisive policy.

## Why this reading should not be trusted as a real lambda effect

Each lambda point above is a **single seed**. The 5-seed DQN run at
lambda=0.1 (identical settings otherwise, `results/dqn_step8_summary.md`)
measured a standard deviation of **0.065** in budget-12 recall *from seed
noise alone*, with individual seeds ranging as far as 0.15 to 0.34. The
total spread across all four lambda points here is **0.110** (0.200 to
0.310) -- less than twice that single-lambda seed-noise band. A single
unlucky or lucky seed at any one lambda value could easily produce a
recall difference this size with lambda held fixed, which is exactly what
`results/lambda_sweep.png` shows visually: the grey error bar (lambda=0.1's
5-seed spread) very nearly spans the same range as the four-point "sweep"
itself.

**Conclusion: this data cannot separate a real lambda effect from seed
noise.** It would take multiple seeds per lambda value (the same 5-seed
protocol used for lambda=0.1) to make that separation -- a substantially
larger compute commitment (5 seeds x 4 lambda values x ~17 min, versus the
4 seeds x ~16 min spent here) that was not run, since the brief was one
seed each. Report the numbers as measured, and report this limitation
alongside them rather than reading a trend into noise.

## What can be said honestly

- All four lambda values land in a broadly similar recall/cost region
  (recall 0.20-0.31, cost 7.0-8.2) -- lambda's effect, if any exists in
  this range, is not dramatic on either axis at 12,000 episodes.
- The extremes of the guide's own prediction (lambda so low the agent
  never learns to stop, or so high it becomes too cautious to commit at
  all) are not reached here -- lambda=0.6 still shows `committed_rate`
  around 0.81-0.89 through training (from the console log), not the
  "eventually too cautious to find anything" collapse the guide describes
  as the expected far end of the sweep. A wider lambda range (e.g. 1.0,
  2.0) might be needed to actually see that predicted collapse.
- If a genuine frontier claim is wanted for the report, it needs multiple
  seeds per lambda value, not one.
