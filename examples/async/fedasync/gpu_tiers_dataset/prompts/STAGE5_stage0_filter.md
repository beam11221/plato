# Stage 5 — turn Stage 0 on. Six runs, one pre-check that can cancel them.

Stage 4 is analysed and accepted. Two of its findings set up this stage:

- **The conservation law holds with mobility:**
  `T = rounds × mean_sleep(mix) / (C × (1 − waste))`, reproducing all eight Task 4 runs
  to within 0.9%. Waste is itself a function of the dispatch tier mix (per-tier rates
  29.3% / 3.6% / 2.4% for xavier / orin / thor, effectively identical in every arm).
- **Prediction 2 failed because the mechanism it assumed was switched off.** Stage 0
  was disabled in Task 4, so no arm could evaluate `R_i ≥ T̂_i + T_comm` and no arm had
  any way to avoid waste except by asking the slow tier less often.

The scoring function changes the *mix*, which is subject to a negative feedback loop
between `s(τ̂)` and `G_i` (Task 2: `s(τ̂)` is a fixed per-tier constant; `G_i` diverges by
tier until it cancels that preference around round 1,000–2,000) and therefore decays.
**The filter changes the `(1 − waste)` term, which is a multiplicative factor on the
whole run and should not decay.** This stage tests that.

Environment, fixed in every cell and identical to Task 4: SUMO closed population,
radius-400 trace, `N=20`, `C=4`, seed 4126, α = 0.8, 4,000 rounds, fleet file
`n20_x10-o7-t3_xa15_p2500_j0p149_s4126`, `upload_policy = coverage_gated`.

Do not touch anything under `plato/`.

---

## Step 0 — pre-check that can cancel this whole stage. No runs.

Task 4 Step D.4 replayed the Stage 0 counterfactual with two values of `T̂`: the tier
mean and the per-client oracle. It did not replay the one a reviewer will ask about
first: **a fleet-mean `T̂`**, which any server can compute from the training times it
has already observed, with no hardware specs at all.

Replay the same counterfactual on the existing Task 4 logs with
`T̂_i = E[T] = 8.323077 s` for every client regardless of tier, and report the same
table as Step D.4:

| T̂ source | waste caught | viable work killed | waste remaining |
|---|---|---|---|
| fleet mean (8.323 s) | | | |
| tier mean (from D.4) | 92.8–96.0% | 1.1–1.8% | 0.6–1.0% |
| per-client oracle (from D.4) | 95.5–98.3% | 0.0% | — |

Break the fleet-mean row down **per tier** as well — where it under-protects and where
it over-filters are different failures and the aggregate hides both.

**Stop condition.** If the fleet-mean filter catches ≥ 90% of waste while killing < 3%
of viable work, **stop and report before running anything.** That would mean the
hardware-spec signal adds little to the filter, the information ladder is flat, and the
six runs below are not worth spending. This is a decision to take with the numbers in
hand, not a parameter to work around.

---

## Step 1 — implement Stage 0

Insert a feasibility predicate between eligibility and scoring:

```
eligible → [busy filter] → [SUMO in-range-now gate] → [Stage 0: R_i ≥ T̂_i + T_comm] → score → lottery
```

`R_i = (x_exit − x_i) / v_i` from the position and speed the vehicle already reports.
`T_comm = 0` in this environment (guaranteed by `sumo_server.configure()`), so the
predicate reduces to `R_i ≥ T̂_i`. Keep it that way for comparability with Task 4, and
say so in the report — it means "still in range when training finishes", not "still in
range when the upload completes".

The in-range-now gate and Stage 0 are **not the same predicate**: the first is
instantaneous, the second is predictive. Do not merge them; log them separately.

### `T̂` source is a config knob, independent of the selector arm

Four sources, selected per run:

| `stage0_t_hat` | value used | available to a real server without hardware specs? |
|---|---|---|
| `none` | filter disabled | — |
| `fleet_mean` | 8.323077 s for every client | yes |
| `tier_spec` | the arm's own tier-mean prediction | no — needs the spec |
| `oracle` | the client's exact jittered `sleep_i` | never — upper bound only |

For `cpu`, `tier_spec` means **cpu's own (wrong) tier prediction**, not gpu's. The
point of that cell is a wrong spec passed through the filter.

### Fallback rule — this must be exactly as specified

If Stage 0 empties the candidate pool, **fall back to the unfiltered pool for that
round and log a `stage0_fallback` event.** Do not idle, and do not dispatch to a
vehicle the predicate rejected while other candidates remain.

Idling would let simulated time pass with no work, which would make a filter that
starves the pool look identical to a filter that works. The fallback keeps those two
distinguishable, and the fallback rate is itself a reported number.

### Log, per round

```
round, eligible_after_busy, eligible_after_range_gate, eligible_after_stage0,
stage0_rejected_ids, stage0_fallback, selected_id, selected_tier,
t_hat_used, R_i_selected
```

Per dispatch, also record whether the dispatch was ultimately wasted
(`raw_in_range = 0` at finish), so false positives and false negatives of the filter
can be scored against ground truth.

---

## Step 2 — the runs

Six new runs. Every control already exists in Task 4 and must not be re-run.

| # | selector | `stage0_t_hat` | what it is for |
|---|---|---|---|
| — | baseline | none | control — Task 4 |
| 1 | baseline | fleet_mean | what a server with no specs gets |
| 2 | **baseline** | **tier_spec** | **what the spec buys, with the selector held constant** |
| 3 | baseline | oracle | upper bound |
| 4 | **gpu** | **tier_spec** | the proposed system |
| 5 | cpu | tier_spec (cpu's own) | a wrong spec through the filter |
| 6 | observed | fleet_mean | the strongest reactive competitor |
| — | gpu | none | control — Task 4 |
| — | observed | none | control — Task 4 |
| — | cpu | none | control — Task 4 |

Runs 1–3 hold the selector at random on purpose. A filter measured on top of `gpu`
alone cannot be separated from `gpu`'s own scoring effect; measured on `baseline` it
can, because `baseline` has no scoring at all.

Parallelism: Task 4 established that the reconstructed clock removes host contention,
so the runs may share a machine. Report which ran together anyway.

---

## Step 3 — what to record per run

Everything Task 4 recorded, plus:

- **filter accounting**: dispatches attempted, rejected by Stage 0, `stage0_fallback`
  count and rate, per tier and per window (1–250 / 250–500 / 500–1000 / 1000–2000 /
  2000–4000)
- **filter quality against ground truth**: true positives (rejected and would have been
  wasted), false positives (rejected but would have uploaded), false negatives
  (admitted and wasted), per tier
- **pool size after each gate**: median / p10 / p90 of `eligible_after_busy`,
  `eligible_after_range_gate`, `eligible_after_stage0`, per window
- cumulative `sim_time_reconstructed` at rounds 100 / 250 / 500 / 1,000 / 2,000 / 4,000
- useful dispatches per reconstructed second, per window
- the conservation-law check: measured `T` against
  `rounds × mean_sleep(mix) / (C × (1 − waste))`
- dispatch tier share per window
- plateau, noise, TTA and rounds-to-target at 70 / 75 / 78 / 80%

---

## Step 4 — reporting

Emit the data and the tables below. Do not rank the arms, do not declare a winner, do
not explain away a result that looks wrong.

**Table A — the information ladder.** Runs 1, 2, 3 against the `baseline / none`
control: total time at round 4,000, waste, filter true/false positives, and useful work
rate in the last window. Selector is constant across this table, so every difference is
the filter.

**Table B — does the filter effect decay?** For every run, the gap against its own
control at rounds 250 / 1,000 / 4,000. The prediction is that the filter gap is flat
across those three while the scoring gap is not. **This is the table the stage exists
for.**

**Table C — score and filter together.** Runs 4 and 2 against their controls: does
`gpu + tier_spec` equal `baseline + tier_spec` at round 4,000 while beating it early?

**Table D — the wrong spec.** Run 5 against run 4 and against `cpu / none`: false
positives, waste, mix, total time.

**Table E — the reactive competitor.** Run 6 against run 1 and against
`observed / none`.

**Table F — filter mechanics.** Fallback rate, pool sizes after each gate, and filter
confusion matrix per tier, all per window.

**Table G — conservation law.** Measured against predicted `T` for all six runs. If the
law now fails, that is a finding, not an error to correct.

---

## Step 5 — written down before the runs start

| # | prediction |
|---|---|
| 1 | `baseline + tier_spec` is **12–18% faster at round 4,000** than `baseline / none` (7,525 s), from waste falling 13.2% → ~1% plus a lighter mix, since the filter rejects xavier more often than the other tiers |
| 2 | **The filter gap does not decay.** The round-4,000 gap for run 2 is no smaller than its round-1,000 gap. This is the central claim of the stage |
| 3 | `baseline + fleet_mean` captures **less than half** of run 2's gain, and kills **≈15% of viable thor work**, because 8.32 s is far longer than thor's true 1.0 s |
| 4 | `gpu + tier_spec` ≈ `baseline + tier_spec` at round 4,000 but ahead of it early — the score contributes a transient, the filter a permanent factor, and the two add roughly independently |
| 5 | `cpu + tier_spec` kills the most viable work and does not recover it, since its mix penalty is already permanent |
| 6 | `observed + fleet_mean` ≈ `baseline + fleet_mean` — the filter does not care which selector it sits in front of |

Report each as held or not held. A prediction that fails is a result; it is not
something to reconcile.

Prediction 2 is the one that matters. If the filter gap decays like the scoring gap
did, then **nothing in this system produces a durable advantage**, and that has to be
said plainly rather than absorbed into the other tables.

---

## Stop conditions

Stop and report rather than continuing if:

- Step 0's pre-check clears its threshold (fleet-mean already catches ≥ 90% of waste at
  < 3% viable work killed)
- `stage0_fallback` fires in more than 10% of rounds in any run — the filter is
  starving the pool and the cell is measuring pool size, not filter quality
- median `eligible_after_stage0` falls below `C = 4` in any run
- the conservation law's error exceeds 3% in any run — something changed that we do not
  understand
- any run's plateau falls more than 2 points below the Task 4 range (82.7–83.5%) — the
  filter would be removing enough clients to damage the model, which is a different and
  more serious finding

---

## Do not

- change sigma, the fleet mix, the tier files, the training anchor, the trace, the
  radius, the dataset, the model, the optimiser, or any selector's scoring or
  selection logic
- change the selection rule: it stays a `V_i`-weighted lottery. The top-K variant is a
  separate question and is not part of this stage
- merge the in-range-now gate with the Stage 0 predicate
- idle a round when Stage 0 empties the pool — fall back and log it
- re-run any Task 4 control
- launch any run beyond the six listed
- tune `T̂`, the predicate, or the fallback rule to make a prediction come true
