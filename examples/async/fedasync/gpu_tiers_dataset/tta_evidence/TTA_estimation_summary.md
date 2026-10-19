# TTA estimation: what it shows (2026-10-02)

Companion to `TTA_estimation_report.md`, which is generated and holds every number.
Stage 11 is partial: seed 4126 complete, seed 4127 in progress. Re-run
`python3 sumo_open_loop/tta_estimation.py apply` when Phase 2 finishes; the choice
below stays fixed.

## The measurement (Step 0)
- One evaluation every 10 rounds, on the full CIFAR-10 test set (10,000 images).
  Binomial SE of one reading at 80% is 0.40 points.
- Near the crossing, the curve is flat and noisy. In `gpu+F` open s4126,
  rounds 600–1,600, accuracy crosses 80% **12 times upward and 11 times
  downward**.
- The current TTA (E0) is already smoothed: an 11-evaluation rolling median,
  then the first evaluation ≥ target, with no interpolation.

## The choice (Step 2, fixed before any arm comparison)

Replicate noise is |Δ| over the 7 identical-config pairs at 80%. Seed CV is
the mean across 30 cells.

| estimator | median noise | max noise | seed CV | sanity |
|---|---|---|---|---|
| E0 current | 5.4% | 21.9% | 10.2% | pass |
| E1 sustained ×3 | 10.5% | 37.4% | 12.8% | pass |
| **E2 MA-5, interpolated** | **5.0%** | 25.9% | **10.0%** | pass |
| E3 isotonic | 18.7% | 29.3% | 16.2% | pass |
| E4 saturating fit | n/a | n/a | n/a | fails in all 116 runs |
| E5 AUC, 30 min | 0.3% | **0.9%** | 1.1% | pass |

- **E2 is the headline estimator.** E0 and E2 tie, being within 1 point on
  median noise, and E2 has the lower seed spread.
- E4: the fit RMSE is 2.6 points (median), so the curve is not a single
  exponential.

## What it means
1. **No crossing-time estimator removes the noise.** E2's worst replicate
   pair is still 26%, against E0's 22%.
   - The noise is not in how the crossing is read off the curve. Re-runs of
     the same configuration reach the 80% band at genuinely different times,
     because they sit on a flat plateau that is only 2–3 points above the
     target.
   - So TTA@80 cannot resolve effects smaller than about ±25%.
2. **The threshold-free metric E5 is 25–30 times less noisy** (max 0.9%
   relative) and passes the sanity check. Caveat: a % of mean accuracy is not
   a % of time, so this ratio flatters E5. What matters is that its effects
   still clear its own noise: the filter's +1–5% does.
   - It changes the reading of several comparisons. Example: gpu+F vs
     baseline+F, open, is **+0.4 / +0.3 / +0.8 points of mean accuracy** over
     the first 30 minutes. That is positive, but only s4128 is beyond noise.
   - With E2 the same comparison is −1.2 / +13.5 / +1.6%: all inside noise.
3. **Rounds vs seconds per round (gpu+F vs baseline+F, open, 80%, E2).**
   - gpu+F needs **+22 / +41 / +17% more rounds**.
   - Each of its rounds is **19 / 19 / 13% shorter** in sim-time.
   - The two cancel at the 80% target. The extra rounds survive the better
     estimator.
   - In the closed population, rounds-to-target are about equal and
     gpu+F's rounds are about 8–9% shorter. s4126 is the exception (its
     rounds are about 6% longer).
4. Counted over all 132 seed × comparison pairs at 80%, 40 are beyond E2's
   noise. E0 against its own bound also gives 40. The headline
   **T @ 4,000 / waste** results (Stages 9–10) are not affected; this is
   about TTA only.

**Recommendation:** use E2 as TTA@80 (per the rule), always next to E5, and
treat any TTA@80 difference under ~26% as inside noise. For claims about
reaching accuracy sooner, E5 is the metric that can actually resolve them.

**Note:** the project doc `tta80-waste-by-stage` could not be opened from
this session. §3.1–3.2 of the report reconstruct its scope (Stages 7b–10,
both regimes, both α, ± filter, per seed). The comparison list should be
checked against the doc.

---

## CIFAR-100 (Stage 13 / 13B), added 2026-10-05

Full tables: `TTA_estimation_cifar100.md`; script:
`sumo_open_loop/tta_estimation_c100.py`.

**Method.**
- Same estimators and parameters as above.
- Targets are 30 / 40 / 50%, because 75–80% is never reached on CIFAR-100.
- The headline stays **E2**, fixed on CIFAR-10.
- **Replicate pairs:** 8 pairs at 30% and 40%:
  - #6 vs #2 in Stage 13 and in 13B;
  - the first 4,000 rounds of each 13B run vs its Stage 13 partner. These
    are re-runs, since nothing depends on the total number of rounds.
- At 50% there is only one pair, because only 13B reaches 50%.

**Results.**
1. **TTA is even noisier on CIFAR-100.**
   - E2's replicate |Δ| is median 13–15% and max 31% (30%) / 50% (40%).
   - E0 is no better: max 21% / 39%.
   - Applied here, the CIFAR-10 rule would pick E0 (E0 and E2 tie, and E0
     has the lower spread). That is for information only; it does not change
     the headline. No estimator fixes the noise, for the same reason as on
     CIFAR-10.
2. **E5 (AUC 30 min) remains far steadier.** Its max replicate Δ is 2.6% of
   mean accuracy, about 1 point. It passes the sanity check.
3. **Which comparisons survive the CIFAR-100 noise bounds:**

   | comparison | survives? |
   |---|---|
   | filter (baseline+F vs none) | beyond noise on E5 in both stages, and on E2 at 30% and 50% |
   | cpu+F slower than gpu+F | beyond noise everywhere |
   | gpu+F vs baseline+F | inside noise on TTA at every target (E2 −0.3% to −27%) |
   | gpu+F on E5 | +0.35 / +0.73 points vs #2 (inside); +0.74 / +1.59 vs #6, of which 13B's is beyond noise |
   | observed+F vs baseline+F | inside noise |

4. **Rounds × seconds-per-round.**
   - gpu+F's rounds are shorter in every pair and target: −2% to −21%.
   - Its rounds-to-target varies from −19% to +16%.
   - So the speed advantage is consistent. Whether it turns into a TTA
     advantage depends on the rounds-to-target term, which is where the
     replicate noise sits.
5. E4 fails on every run (RMSE about 3 points), as on CIFAR-10.

**Recommendation for CIFAR-100: the same as for CIFAR-10.** Report E5 (and
T@N) as the resolving metrics, and TTA only next to them, treating TTA
differences under about 30–50% as noise.
