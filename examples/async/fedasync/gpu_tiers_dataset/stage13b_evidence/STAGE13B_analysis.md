# Stage 13B: CIFAR-100 at 10,000 rounds, analysis (seed 4126)

> **One seed per config, not a result.** Six runs: the five methods plus one
> replicate. Raw tables: `STAGE13B_report.md` (auto-generated). Series:
> `series_<#>_*.csv`. Same metric definitions as `STAGE13_analysis.md`.

---

## 1. Soundness

- All six runs reached round 10,000. The baseline / none run's last merge is
  logged as round 9,999.
- Conservation error is at most 0.4%.
- Fallback is at most 0.6%.
- Predictor agreement is 100%.

There were no failures.

---

## 2. Does longer training raise the plateau?

Median accuracy over windows of rounds:

| # | run | r 2–3k | r 3–4k | r 5–6k | r 7–8k | **r 9–10k** | Stage 13 plateau (4k) | best |
|---|---|---|---|---|---|---|---|---|
| 1 | baseline / none | 45.7 | 46.2 | 48.7 | 50.4 | **50.7** | 46.6 | 52.0 |
| 2 | baseline + F | 45.1 | 46.7 | 48.5 | 50.0 | **50.8** | 47.7 | 51.8 |
| 3 | gpu + F | 44.7 | 45.9 | 46.9 | 48.8 | **50.1** | 46.4 | 51.5 |
| 4 | observed + F | 45.0 | 46.2 | 47.4 | 49.6 | **50.1** | 45.9 | 51.4 |
| 5 | cpu + F | 44.3 | 46.1 | 47.0 | 48.8 | **49.7** | 44.8 | 50.7 |
| 6 | baseline + F, replicate | 44.2 | 45.2 | 47.1 | 48.8 | **49.9** | 46.6 | 50.8 |

- **Yes, but slowly.** The plateau rises by about 3–4 points, from about
  46–47% at 4k to about 50–51% at 10k.
- It is still climbing at the end: +0.4 to +1.3 points from rounds 7–8k to
  9–10k. The model has not converged at 10,000 rounds, but each further
  round adds very little.
- **50% is now reachable**, by every run. 60% and above is not.
- **The arms end level.** The final plateaus span 49.7–50.8%. The replicate
  pair (#6 vs #2) is itself 0.9 points apart, so there is no difference
  between methods in final accuracy.
- Single-evaluation noise is unchanged (SD 3.2–3.8 points), and dips are as
  common as in Stage 13: 4.1–5.9% of evaluations (DIP definition).

---

## 3. Time per round: the method's advantage holds over 10,000 rounds

| # | run | T @ 10,000 (s) | vs baseline + F | waste | share x / o / t |
|---|---|---|---|---|---|
| 1 | baseline / none | 18,801 | +63.5% | 13.82% | 36.3 / 42.5 / 21.1 |
| 2 | baseline + F | 11,499 | — | 0.38% | 22.0 / 50.0 / 28.0 |
| 3 | gpu + F | **10,176** | **−11.5%** (−13.2% vs #6) | 0.25% | 18.3 / 52.1 / 29.5 |
| 4 | observed + F | 11,232 | −2.3% | 0.64% | 21.5 / 50.3 / 28.2 |
| 5 | cpu + F | 13,783 | +19.9% | 0.62% | 29.0 / 43.6 / 27.3 |
| 6 | baseline + F, replicate | 11,729 | +2.0% | 0.36% | 22.8 / 49.5 / 27.7 |

The ordering is the same as in Stage 13 and on CIFAR-10:
- gpu+F is the fastest per round;
- the filter cuts wall time by 39% vs baseline / none;
- cpu+F is the slowest filtered arm.

The replicate's T differs by 2.0%, within the CIFAR-10 noise band, whereas
Stage 13's 4k replicate differed by 5.2%.

---

## 4. Time to accuracy

**Wall-clock (E0):**

| pair | TTA 30% | TTA 40% | **TTA 50%** | rounds to 50% |
|---|---|---|---|---|
| #6 vs #2, replicate (noise) | −4.1% | +39.2% | **+17.9%** | +13.9% |
| #2 vs #1, filter | −35.7% | −31.5% | **−32.7%** | +11.9% |
| #3 vs #2, gpu vs baseline (+F) | −21.0% | −16.1% | **+2.8%** | +16.0% |
| #3 vs #6, gpu vs replicate | −17.0% | −39.5% | **−12.7%** | +1.8% |
| #4 vs #2, observed vs baseline (+F) | −21.0% | +16.9% | **+19.7%** | +20.2% |
| #5 vs #3, cpu vs gpu (+F) | +42.9% | +56.3% | **+48.6%** | +8.4% |

**Mean accuracy over the first 30 minutes, 1 hour and 3 hours of simulated
time** (higher is better):

| # | run | AUC 30 min | AUC 1 h | AUC 3 h |
|---|---|---|---|---|
| 1 | baseline / none | 36.11 | 39.55 | 43.90 |
| 2 | baseline + F | 38.11 | 41.08 | 45.53 |
| 3 | gpu + F | **38.84** | **41.46** | **45.77** |
| 4 | observed + F | 37.90 | 41.02 | 45.06 |
| 5 | cpu + F | 36.91 | 40.04 | 44.27 |
| 6 | baseline + F, replicate | 37.25 | 40.10 | 44.52 |

**Readings:**
- **The filter result holds at every target**, 30–50%: −32% to −36%. It is
  beyond noise at all three AUC horizons (+1.5 to +2.0 points vs baseline /
  none).
- **gpu+F has the highest AUC at every horizon.**
  - It is +0.2 to +0.7 points above #2 and +1.3 to +1.6 above #6.
  - But the replicate pair is itself 0.9–1.0 points apart, so against #2 it
    is inside noise.
  - Its TTA 50% is +2.8% vs #2 and −12.7% vs #6, both inside the 17.9%
    replicate gap.
- **The rounds-vs-speed trade-off from CIFAR-10 appears again.** gpu+F needs
  16% more rounds to reach 50% than #2, but each round is about 12% shorter,
  and the two largely cancel.
- **cpu+F is clearly the worst filtered arm** at every metric.

---

## 5. The first 4,000 rounds of 13B are a second replicate of Stage 13

Nothing in the configuration depends on the total number of rounds: there is
no LR scheduler, and the basic trainer never reads `rounds`. So rounds
1–4,000 of each 13B run are an independent re-run of its Stage 13 partner
(same seed; PLATO's host-time component of the clock differs). This gives
**two runs per arm**, and four for baseline+F.

| arm | runs | T @ 4,000 (s) | AUC 30 min |
|---|---|---|---|
| baseline / none | 13, 13B | 7,139 / 7,324 | 36.23 / 36.11 |
| baseline + F | 13 #2, 13 #6, 13B #2, 13B #6 | 4,650 / 4,410 / 4,535 / 4,616 (mean 4,553) | 38.65 / 38.25 / 38.11 / 37.25 (mean 38.07, SD 0.59) |
| gpu + F | 13, 13B | **3,908 / 3,946** | **38.99 / 38.84** |
| observed + F | 13, 13B | 4,422 / 4,443 | 38.59 / 37.90 |
| cpu + F | 13, 13B | 5,377 / 5,334 | 36.56 / 36.91 |

**Readings:**
- **gpu+F is faster per round than every baseline+F run.** T@4000 is −14%
  against the four-run mean, with no overlap.
- **gpu+F's two AUC30 values (38.99, 38.84) are above all four baseline+F
  runs** (37.25–38.65). The mean difference is +0.85 points, about 1.4
  baseline+F SDs.
  - This is the most direct accuracy-over-time evidence so far for the score
    on CIFAR-100.
  - It is still only two runs against four, at one data / physics seed. It
    is suggestive, not established.
- observed+F lies within the baseline+F spread.
- cpu+F and baseline / none lie clearly below it.
- **The noise yardstick is now better estimated.** The SD of baseline+F's
  AUC30 is 0.59 points, and its T@4000 range is 4,410–4,650 s (−3.1% to
  +2.1% about the mean).

---

## 6. Summary

1. **Final accuracy:** 10,000 rounds lift CIFAR-100 from about 46–48% to
   about 50–51%, still slowly rising. Every method ends at the same level,
   within the 0.9-point replicate gap.
2. **Speed:** the method's time-per-round gains are unchanged over 10,000
   rounds. Filter −39%, gpu+F −11 to −13% vs baseline+F, cpu+F the slowest.
3. **Accuracy over time:** the filter's gain is clear. gpu+F leads on mean
   accuracy over 30 min to 3 h in both experiments, by about the size of the
   noise in each one alone. Pooling Stage 13 and 13B's first 4k rounds puts
   both gpu+F runs above all four baseline+F runs on AUC30. More seeds are
   needed to turn this into a result.
4. **Noise:** single evaluations still scatter by about 3–4 points, with
   dips in 4–6% of evaluations in every arm (see `DIP_analysis_report.md`).
   TTA at a single target swings by up to about 40% between replicates. The
   AUC metrics and T@N are the reliable ones here.

**Caveat:** windows, AUC horizons and the pooling of section 5 go beyond the
report's standard metrics. They were added for this analysis (user request,
2026-10-05) from existing logs only; no run was added.
