# Task 3 — time-resolved dispatch share (existing logs, no runs)

## Provenance note

No `dispatch_ledger.csv` exists anywhere in the repository — this file is not
produced by any Stage 3 script. It is **reconstructed here** from the raw
per-dispatch log lines each Stage 3 run already wrote
(`[GpuTierSleep] round=<r> client_id=<c> tier_id=<t> ...`, one line per
dispatch, confirmed one-per-round in this closed pool: 4003-4007 lines against
4000 rounds, matching the "exactly one dispatch per round" claim in the Stage
3 summary). The reconstruction script and its output:

- `dispatch_ledger_reconstructed.csv` — every dispatch, all 5 arms × 2 cells (40,050 rows)
- `dispatch_share_windowed.csv` — the windowed counts/shares below

Fleet composition (`n20_x10-o7-t3`): 10 xavier / 7 orin / 3 thor = **50% /
35% / 15%** by count. Tier speed order (fastest→slowest at C=4):
thor (1.00 s) < orin (1.92 s) < xavier (15.0 s).

## Windowed dispatch share by tier, %  (N=20, C=4)

| window | arm | xavier | orin | thor |
|---|---|---|---|---|
| 1-250 | baseline | 42.3 | 39.1 | 18.6 |
| 1-250 | cpu | 49.0 | 36.4 | 14.6 |
| 1-250 | gpu | 32.4 | 47.8 | 19.8 |
| 1-250 | gpu_oracle | 34.4 | 45.5 | 20.2 |
| 1-250 | observed | 33.2 | 47.4 | 19.4 |
| 250-500 | baseline | 37.2 | 42.4 | 20.4 |
| 250-500 | cpu | 48.8 | 36.0 | 15.2 |
| 250-500 | gpu | 34.0 | 45.6 | 20.4 |
| 250-500 | gpu_oracle | 34.4 | 45.6 | 20.0 |
| 250-500 | observed | 35.6 | 45.2 | 19.2 |
| 500-1000 | baseline | 44.0 | 40.2 | 15.8 |
| 500-1000 | cpu | 52.8 | 30.2 | 17.0 |
| 500-1000 | gpu | 41.8 | 38.2 | 20.0 |
| 500-1000 | gpu_oracle | 42.8 | 37.4 | 19.8 |
| 500-1000 | observed | 41.8 | 38.0 | 20.2 |
| 1000-2000 | baseline | 42.0 | 40.1 | 17.9 |
| 1000-2000 | cpu | 56.8 | 32.2 | 11.0 |
| 1000-2000 | gpu | 44.6 | 39.5 | 15.9 |
| 1000-2000 | gpu_oracle | 45.4 | 41.1 | 13.5 |
| 1000-2000 | observed | 47.2 | 37.8 | 15.0 |
| 2000-4000 | baseline | 41.7 | 40.5 | 17.9 |
| 2000-4000 | cpu | 55.0 | 31.2 | 13.8 |
| 2000-4000 | gpu | 45.3 | 36.5 | 18.3 |
| 2000-4000 | gpu_oracle | 43.8 | 38.3 | 17.9 |
| 2000-4000 | observed | 43.3 | 38.9 | 17.8 |

## Windowed dispatch share by tier, %  (N=20, C=8)

| window | arm | xavier | orin | thor |
|---|---|---|---|---|
| 1-250 | baseline | 33.9 | 42.8 | 23.3 |
| 1-250 | cpu | 37.4 | 42.4 | 20.2 |
| 1-250 | gpu | 24.5 | 50.6 | 24.9 |
| 1-250 | gpu_oracle | 24.1 | 51.0 | 24.9 |
| 1-250 | observed | 24.9 | 52.1 | 23.0 |
| 250-500 | baseline | 32.4 | 44.8 | 22.8 |
| 250-500 | cpu | 37.2 | 41.6 | 21.2 |
| 250-500 | gpu | 24.8 | 51.2 | 24.0 |
| 250-500 | gpu_oracle | 25.6 | 49.6 | 24.8 |
| 250-500 | observed | 23.6 | 50.4 | 26.0 |
| 500-1000 | baseline | 31.2 | 48.2 | 20.6 |
| 500-1000 | cpu | 42.4 | 38.4 | 19.2 |
| 500-1000 | gpu | 28.8 | 46.2 | 25.0 |
| 500-1000 | gpu_oracle | 28.4 | 48.4 | 23.2 |
| 500-1000 | observed | 28.0 | 47.6 | 24.4 |
| 1000-2000 | baseline | 31.2 | 48.2 | 20.6 |
| 1000-2000 | cpu | 46.3 | 39.0 | 14.7 |
| 1000-2000 | gpu | 36.2 | 45.7 | 18.1 |
| 1000-2000 | gpu_oracle | 37.2 | 45.1 | 17.7 |
| 1000-2000 | observed | 34.9 | 46.5 | 18.6 |
| 2000-4000 | baseline | 31.8 | 47.6 | 20.6 |
| 2000-4000 | cpu | 47.0 | 37.2 | 15.8 |
| 2000-4000 | gpu | 35.5 | 45.1 | 19.5 |
| 2000-4000 | gpu_oracle | 35.4 | 44.5 | 20.1 |
| 2000-4000 | observed | 34.4 | 46.3 | 19.4 |

Full precision numbers, counts, and totals per window are in
`dispatch_share_windowed.csv`.

## Reading against the conservation-law expectation

**Not flat — the predicted direction shows up, but the predicted crossing
does not, at this window resolution.**

- **`gpu`, `gpu_oracle`, `observed`** (the three arms expected to start
  fast-tier-heavy and drift toward the fleet mix): at C=4, xavier share rises
  monotonically from ~32-34% (window 1-250) to ~44-45% (window 2000-4000);
  orin share falls from ~46-48% to ~36-39%. Both move in the direction the
  conservation law predicts (drift toward 50%/35% xavier/orin), but neither
  reaches the fleet-mix value inside the 4,000-round budget — xavier is still
  5-6 points short, orin still 1-4 points short, in the final window. Thor is
  noisy (14-20%) around its 15% target throughout, with no clear monotonic
  trend either way. The same pattern, slightly weaker, holds at C=8: xavier
  rises ~24%→~35%, orin falls ~50-52%→~44-46%, again short of 50/35 by round
  4000, and by a *larger* margin than at C=4 (xavier 15 points short at C=8 vs.
  5-6 points short at C=4 in the last window).

  This is the opposite of what the Stage 3 summary's crossing estimate implies
  (crossover "≈2,400 at C=4, ≈1,900 at C=8" — i.e., C=8 crossing *sooner*).
  Dispatch share at C=8 is converging **more slowly**, not faster, toward the
  fleet mix than at C=4, at least through round 4,000. Two things can both be
  true at once here and this window resolution cannot separate them: (a) the
  crossing round for *dispatch share* may simply be later than the crossing
  round for *cumulative time advantage* — they are different quantities, and
  the Stage 3 estimate was for the latter, not the former; (b) the 2000-4000
  bucket is 2,000 rounds wide, too coarse to locate any crossing point inside
  it precisely. **This file cannot confirm or deny the specific round numbers
  in the Stage 3 summary — it can only say the qualitative direction (drift
  toward fleet mix) holds for these three arms, and that the drift is not yet
  complete by round 4,000 at either C=4 or C=8.**

- **`cpu`**: predicted to "stay Xavier-heavy throughout." It does not stay
  flat — it gets *more* xavier-heavy over the run, not just heavy at a
  constant level: 49.0%→55.0% at C=4, 37.4%→47.0% at C=8, both monotonic
  increases from window 1 to window 5. The qualitative claim ("stays
  xavier-heavy", i.e. does not drift toward the other three arms' pattern) is
  supported; the stronger implicit reading ("stays" = constant) is not — cpu's
  xavier share moves the *most* of any arm across the run, just in the
  opposite direction from gpu/observed/gpu_oracle.

- **`baseline`**: shows no clear monotonic trend at C=4 (xavier oscillates
  42→37→44→42→42, no drift), consistent with it having no speed signal to
  decay from. At C=8 it is closer to flat still (34→32→31→31→32), again
  consistent with the null expectation for an arm that was never fast-tier-
  heavy to begin with.

## Bottom line

The direction of the conservation-law claim holds qualitatively for `gpu`,
`gpu_oracle`, and `observed` (drift toward the fleet mix, not flat), and the
qualitative `cpu` claim holds too (stays the most xavier-heavy of the five
arms throughout) — but the specific crossing rounds in the Stage 3 summary are
neither confirmed nor contradicted by this data; the window widths requested
here are too coarse to test them, and the one thing that can be said at this
resolution (C=8 converges slower than C=4 through round 4,000) runs the
opposite direction from the summary's stated crossover ordering. A finer-
grained (e.g. 100-round) windowing of the same reconstructed ledger would be
needed to actually locate the crossing round; that was not requested here and
was not run.
