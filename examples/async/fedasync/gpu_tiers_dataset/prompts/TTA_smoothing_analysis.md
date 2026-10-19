# TTA estimation — reduce the noise on time-to-accuracy. Log-only, no training runs.

This needs no GPU. Run it alongside Stage 11 Phase 2. **Stage 12 is on hold**; do not
start it.

Do not touch anything under `plato/`. Do not modify any run folder; read only.

---

## Why — written before any result

Headline metrics for the thesis are **time-to-accuracy (TTA, target 80%)** and **waste**.
T @ 4,000 is a backup metric only.

TTA@80 is too noisy to resolve the effects we care about. Identical-config re-runs of
random selection differ by up to ±22% at TTA@80:

| pair (same config, same seed) | 4126 | 4127 | 4128 |
|---|---|---|---|
| `baseline / none` open: Stage 7b/8 vs Stage 10 re-run | +18% | −2% | +5% |
| `baseline / none` closed: Stage 7b/8 vs Stage 10 re-run | +3% | −22% | +5% |

The same pairs differ by ≤ 3.5% on T @ 4,000. So the noise is in **when the accuracy curve
first touches 80%**, not in the clock. The 80% target sits close to the 82–83% plateau,
where the curve is flat and a single noisy evaluation decides the crossing.

The question is whether a better estimator of "when the model reached 80%" removes most of
that noise, **without changing what is measured**.

---

## Step 0 — report the measurement itself

For a representative run (Stage 10 `gpu + F` open, s4126) and in general:

1. evaluation schedule: every how many rounds, and how many evaluation points fall before
   the 80% crossing
2. test-set size, and the binomial standard error of one accuracy reading at 80%
3. the raw accuracy-vs-simulated-time series for that run around the crossing
   (rounds 600–1,600), to show how many times it crosses 80% up and down
4. how TTA is computed now (first evaluation ≥ target? interpolated?), with the code
   location

## Step 1 — estimators, fixed now

Compute each from the per-evaluation series `(round, sim_time, accuracy)`, for targets
75 / 78 / 80%:

| id | estimator |
|---|---|
| E0 | **current method**, unchanged (reference) |
| E1 | **sustained crossing**: first evaluation where accuracy ≥ target and stays ≥ target for the next 3 evaluations |
| E2 | **centred moving average** of accuracy over 5 evaluations, then first crossing (interpolated in sim-time) |
| E3 | **isotonic (monotone non-decreasing) regression** of accuracy on sim-time, then first crossing |
| E4 | **saturating fit** `acc(t) = a − b·exp(−c·t)` by least squares on the whole run, inverted at the target. Report fit RMSE; mark a run as "fit failed" if RMSE > 1 point or the target lies above `a` |
| E5 | **threshold-free**: mean accuracy over sim-time `[0, T_ref]` (normalised area under the curve), `T_ref` = 30 simulated minutes for every run. Higher = better |

Also, for E0–E4, split TTA into **rounds-to-target** and **seconds-per-round up to the
target**. This is the decomposition Stage 11 Step 0b asked for.

Use exactly these parameters (3 evaluations, window 5, `T_ref` = 30 min). Do not try
others.

## Step 2 — choose the estimator by a rule written now, before any arm comparison

Score each estimator **only** on data where the true difference is zero or known:

1. **Replicate noise**: |Δ%| on the identical-config pairs: the six `baseline / none`
   pairs above, plus the Stage 9 invariance run (`baseline_open` s4126 vs 7b). Report
   the median and max over the 7 pairs
2. **Seed spread**: within-cell coefficient of variation across seeds 4126/4127/4128,
   averaged over all Stage 7b–10 cells
3. **Sanity**: the known large effect must survive. `baseline + F` vs `baseline / none`
   must be negative in all 6 seed × regime pairs

**Rule:** among the estimators that pass (3), the headline estimator is the one with the
lowest median replicate noise at the 80% target; ties (within 1 point) are broken by
seed spread. E5 is always reported next to it as the threshold-free check.

Write the scores and the choice into the report **before** computing any arm comparison
in Step 3. Do not change the choice after Step 3.

## Step 3 — apply it everywhere

With the chosen estimator, at 75 / 78 / 80%, and with E5:

1. regenerate every table in the project doc `tta80-waste-by-stage` (Stages 7b–10, both
   regimes, both α, with and without the filter), per seed. Include E0 next to the new
   estimator so the change is visible
2. the paired comparisons of that doc, per seed, each marked as **beyond** or **inside**
   the chosen estimator's replicate noise (its max |Δ| from Step 2)
3. the Stage 11 curve over `p_return` = 0 / 0.25 / 0.5 / 0.75 / 1.0 (`p = 1` = Stage 10
   closed), for `gpu + F`, `baseline + F`, `observed + F`, `cpu + F` and `baseline / none`,
   for every seed available. When Phase 2 finishes, re-run this step for all three seeds
4. rounds-to-target and seconds-per-round for `gpu + F` vs `baseline + F`, open and closed,
   per seed. This answers whether `gpu + F`'s 14–27% extra rounds (E0) survive a better
   estimator

Waste needs no change. Carry it through unchanged next to every TTA.

## Output

- `TTA_estimation_report.md`: Step 0, the Step 2 scores and choice, then Step 3
- `tta_estimates.csv`: one row per run × estimator × target, with rounds-to-target and
  seconds-per-round

## Do not

- start Stage 12, or any training run
- choose or change the estimator after looking at arm comparisons
- tune window, k, `T_ref` or the fit form per run, per arm or per stage
- drop runs because their estimate looks odd. Report fit failures as such
- change waste, plateau or any existing analysis output
