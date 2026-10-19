# Stage 5b — fix `R_i`. Two runs, one question.

Stage 5's six runs are analysed and accepted. The headline number needs one thing
resolved before it can be written up.

## What we found, and why this stage exists

`baseline + tier_spec` finished **41.7% faster** than `baseline / none`
(4,388.2 s vs 7,525.2 s). Decomposing that against the conservation law
`T = rounds × mean_sleep(mix) / (C × (1 − waste))`:

| channel | contribution |
|---|---|
| waste eliminated (13.17% → 0.32%) | **−12.9%** |
| dispatch mix shifted (xavier share 39.0% → 22.3%) | **−32.9%** |

The waste channel is what Stage 5 set out to measure and it matches the
pre-registered 12–18% prediction. The mix channel was not predicted, and it comes
from how `R_i` was implemented:

```
R_i(t) = window_start + 2*radius/v_max - t          # = window_start + 41.142857 - t
```

`41.142857 s = 800 m / 19.444 m/s` is the **fastest possible** crossing, not the
typical one. Stage 5's own validation found the true crossing exceeds it in 99.86% of
windows. So this expression is not "time remaining"; it is "time remaining *if the
vehicle had already been travelling at the speed limit*", and its error grows with
time-in-range. Past `t = window_start + 41.14` it returns ≤ 0 for every vehicle,
however much dwell actually remains.

Rearranged, the predicate `T̂_i ≤ R_i` becomes
`time-in-range < 41.142857 − T̂_i`, i.e. each vehicle is eligible only during a fixed
opening slice of its window whose length is set by its own training time — 26.1 s for
xavier, 39.2 s for orin, 40.1 s for thor. **That is a speed filter, not a feasibility
predicate.** The confusion matrix agrees: of everything the filter rejected, 75.7% of
orin rejections and 83.9% of thor rejections would in fact have completed in range.

The logic `T̂_i ≤ R_i` is correct and is not in question. Only `R_i` is.

## The fix being tested

Keep the worst-case assumption — it is what makes the filter safe — but apply it to
the future rather than the past:

```
R_i(t) = |x_exit_boundary(direction) - x_i(t)| / v_max
```

Still a guaranteed lower bound on true remaining dwell, since SUMO enforces
`speed ≤ v_max` everywhere. Still requires no prediction of how the vehicle will
accelerate. But it reads the vehicle's **actual current position** instead of inferring
distance from elapsed time, so the error does not accumulate.

This is deployable: position arrives in the vehicle's own V2X basic safety message, and
`v_max` is the road's posted limit. It is not an oracle.

Everything else is held fixed. Do not touch anything under `plato/`.

---

## Step 1 — wire position in, and implement two `R_i` variants

`sumo_mobility.py`'s docstring records that the per-second trace
(`trace.csv`, with `x_m` / `speed_mps` / `dist_to_rsu_m`) exists but was never loaded —
only `coverage_windows.csv`. Load it for this stage.

Add `stage0_r_i` as a config knob alongside the existing `stage0_t_hat`:

| `stage0_r_i` | expression | deployable | role |
|---|---|---|---|
| `vmax_elapsed` | `window_start + 2*radius/v_max − t` | yes | Stage 5's existing behaviour — the control, already run |
| **`vmax_distance`** | `|x_exit_boundary − x_i(t)| / v_max` | **yes** | **the variant under test (B)** |
| `oracle_exit` | `window_end − t` | no | exact remaining dwell — the ceiling (A) |

`vmax_elapsed` must remain the default so every existing config, including all Stage 5
and Task 4 runs, stays byte-identical. Assert this in a test.

### Direction handling

Coverage is the span `[rsu_x − radius, rsu_x + radius] = [200, 1000]` m on a 1,200 m
road and vehicles traverse it in both directions. `x_exit_boundary` is 1,000 m for a
vehicle travelling in +x and 200 m for one travelling in −x. Derive the direction from
the trace (sign of the change in `x_m`, or the sign of `speed_mps` if the trace carries
it signed) — **do not assume one direction.** Report the split of windows by direction.

### Validation before any run

1. **`vmax_distance` must be a lower bound on true remaining dwell.** Check it against
   `window_end − t` for every candidate-round in the trace. Report the violation rate.
   Stage 5 measured 99.86% compliance for `vmax_elapsed`; if `vmax_distance` is worse
   than that, the geometry or the direction handling is wrong — **stop and report.**
2. **`vmax_distance` must be tighter than `vmax_elapsed`**, i.e. closer to
   `window_end − t` from below, at essentially every point past the start of a window.
   Report the median gap to truth for both.
3. **Crossing-time distribution** — median, p10, p90 of the true time each vehicle
   spends inside coverage, across all 10,180 windows, plus the fraction exceeding
   41.142857 s. This single distribution sets the size of the effect this stage is
   testing, and it has never been reported.

---

## Step 2 — the runs

Two runs. Selector `baseline`, `stage0_t_hat = tier_spec` in both — the `T̂` ladder is
held constant so that `R_i` is the only thing varying.

| # | selector | `stage0_t_hat` | `stage0_r_i` | |
|---|---|---|---|---|
| — | baseline | none | — | control, Task 4: **7,525.2 s**, waste 13.17%, xavier 39.0% |
| — | baseline | tier_spec | `vmax_elapsed` | control, Stage 5 run 2: **4,388.2 s**, waste 0.32%, xavier 22.3% |
| **1** | baseline | tier_spec | **`vmax_distance`** | the deployable fix (B) |
| **2** | baseline | tier_spec | **`oracle_exit`** | the ceiling (A) |

Environment identical to Stage 5 in every other respect: SUMO closed population,
radius-400 trace `e1_closed_seed4126_range400`, N=20, C=4, seed 4126, α = 0.8,
4,000 rounds, `upload_policy = coverage_gated`, fleet file
`n20_x10-o7-t3_xa15_p2500_j0p149_s4126`, `V_i`-weighted lottery unchanged, Stage 0
fallback rule unchanged (fall back to the unfiltered pool, never idle, log it).

**Do not re-run either control.**

---

## Step 3 — what to report

Mirror `analyze_stage5.py`'s output so the new runs drop straight into the existing
tables, plus the items below.

**Table 1 — the `R_i` ladder.** The four rows above: total time at round 4,000, waste,
dispatch share per tier, useful dispatches per reconstructed second in the last window.

**Table 2 — the decomposition.** For each row, split the gain against
`baseline / none` into its mix channel and its waste channel, using the conservation
law exactly as Stage 5's analysis did. **This is the table the stage exists for.**

**Table 3 — filter quality.** Per tier, both denominators, stated explicitly:
- of everything rejected, the fraction that would in fact have completed
- of everything that would have completed, the fraction rejected
- of everything admitted, the fraction that was wasted

Stage 5's `vmax_elapsed` numbers for the first of these were xavier 22.8% / orin 75.7%
/ thor 83.9%.

**Table 4 — eligibility window.** Per tier and per variant: the mean fraction of each
coverage window during which a vehicle of that tier passes the predicate. This is the
speed bias expressed directly, and it is the clearest single diagnostic of what changed.

**Table 5 — fairness.** Dispatches per client per tier, the thor:xavier ratio, and
Jain's fairness index over the 20 clients. Reference points: `baseline / none` gives
thor:xavier ≈ 1.76×, Stage 5's `vmax_elapsed` gives 4.05×.

**Table 6 — conservation law**, measured against predicted `T`.

**Table 7 — outcome.** Plateau, noise, TTA and rounds-to-target at 70 / 75 / 78 / 80%,
alongside both controls.

Also report: `stage0_fallback` rate, pool sizes after each gate (`eligible_after_busy`,
`eligible_after_range_gate`, `eligible_after_stage0`), per window.

---

## Step 4 — written down before the runs start

| # | prediction |
|---|---|
| 1 | `vmax_distance` lands **between** the two controls, i.e. between −12.9% and −41.7% against `baseline / none`. Where it lands is the answer: close to −41.7% means the approximation cost nothing and Stage 5's number stands as reported; close to −12.9% means most of Stage 5's headline was the approximation, not feasibility |
| 2 | `vmax_distance` rejects far more precisely: the "rejected but would have completed" fraction for orin and thor drops well below Stage 5's 75.7% / 83.9% |
| 3 | xavier dispatch share under `vmax_distance` sits between 22.3% and 39.0% |
| 4 | `oracle_exit` is faster than `vmax_distance` but not by much; the gap between them is the price of the safety margin, and it is the only part of the filter's behaviour that is not deployable |
| 5 | Waste stays at or below ~1% in both runs. The filter does not need the over-rejection to prevent waste |
| 6 | Fairness improves relative to `vmax_elapsed`: thor:xavier falls below 4.05×, without returning to the control's 1.76× — some speed bias is genuine feasibility and should remain |

Report each as held or not held. A failed prediction is a result, not something to
reconcile.

Prediction 1 is the decision. It determines whether Stage 5's 41.7% is reported as
measured or re-attributed, and it decides which `R_i` the thesis describes as the
method.

---

## Stop conditions

Stop and report rather than continuing if:

- `vmax_distance` violates the lower-bound property more often than `vmax_elapsed` did
  (Step 1 validation 1)
- `stage0_fallback` fires in more than 10% of rounds in either run
- median `eligible_after_stage0` falls below `C = 4`
- the conservation law's error exceeds 3% in either run
- either run's plateau falls more than 2 points below the Stage 5 range (82.1–83.7%)

---

## Do not

- change the selector, the scoring function, the selection rule, sigma, the fleet mix,
  the tier files, the training anchor, the trace, the radius, `v_max`, the dataset, the
  model, or the optimiser
- change `stage0_t_hat` — it is `tier_spec` in both runs, on purpose
- implement or run a `current_speed` variant of `R_i` in this stage; it is a later
  ablation and is not safe in the same way
- change the default value of `stage0_r_i`
- re-run any control
- tune the predicate, the boundary, or the direction rule to make a prediction come true
