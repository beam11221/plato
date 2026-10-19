# Stage 5 — progress log

Running record against `prompts/STAGE5_stage0_filter.md`. Appended to as each
step completes; not rewritten in place.

---

## Step 0 — pre-check (no runs). CLEARED — stage proceeds.

Replayed the Stage 0 counterfactual on the existing Task 4 logs (all 8 runs:
4 arms × 2 upload policies) with `T̂_i = 8.323077 s` for every client,
using the same method as Task 4 Step D.4 (`analyze_stage4_task4.
stage0_counterfactual`, `R_i = window_end(client_id, t_dispatch) -
t_dispatch`).

`8.323077` verified: `(10×15 + 7×1.923077 + 3×1.0) / 20 = 8.323077`, the
composition-weighted mean of `T_tier_s` from
`gpu_tiers_dataset/fleets/n20_x10-o7-t3_xa15_p2500_j0p149_s4126/manifest.json`.
(Note: this is the noise-free tier-mean-of-means, not the mean of the fleet's
own *jittered* `sleep_times.csv`, which is 7.970595 s — a ~4.4% difference.
Used 8.323077 as specified.)

| T̂ source | waste caught | viable work killed | waste remaining |
|---|---|---|---|
| fleet mean (8.323 s) | 58.7–66.0% | 8.1–11.0% | 4.6–6.9% |
| tier mean (Task 4 D.4) | 92.8–96.0% | 1.1–1.8% | 0.6–1.0% |
| per-client oracle (Task 4 D.4) | 95.5–98.3% | 0.0% | — |

**Stop condition** (≥90% waste caught AND <3% viable killed): **not met** —
fleet-mean is far short (58.7–66.0% vs 90%) and kills far more viable work
(8.1–11.0% vs 3%). The hardware-spec signal is not redundant with the
fleet-only baseline. **Stage proceeds to the six runs.**

Per-tier breakdown (aggregate hides two distinct failure modes, as the prompt
anticipated):

| tier | true T | waste caught | viable killed | waste remaining | failure mode |
|---|---|---|---|---|---|
| xavier | 15.0 s | 54.5–60.3% | 0.0% | 11.3–13.4% | under-protected (8.32s is shorter than xavier needs) |
| orin | 1.92 s | 100% | 12.9–15.4% | 0.0% | over-filtered |
| thor | 1.0 s | 100% | 14.8–18.4% | 0.0% | over-filtered (worst) |

thor's 14.8–18.4% viable-killed range brackets Step 5's Prediction 3 (≈15%)
reasonably well — noted here, to be checked formally against the real Stage 5
run once it finishes, not against this replay (which reuses Task 4's dispatch
sequence, not a live run with the filter actually installed).

---

## Step 1 — implementation. DONE.

### R_i: not `(x_exit − x_i)/v_i` from live telemetry

Two things ruled out first, both confirmed against the actual codebase before
writing anything:

- **No live position/speed exists anywhere in this pipeline.** Only
  `coverage_windows.csv` (entry/exit *times*) is loaded
  (`sumo_mobility.CoverageOracle`); the raw per-second `trace.csv` with
  `x_m`/`speed_mps`/`dist_to_rsu_m` is 35× larger and was never wired in
  (see `sumo_mobility.py`'s own module docstring).
- **`window_end()` (Task 4's own D.4 formula) is oracle-exact**, i.e. it reads
  the vehicle's true future exit time — not something a real dispatcher could
  know, and not what the user asked to model.

Decision (confirmed with the user): fix `v_i` to the simulation's own hard
speed cap rather than either extreme.

`sumo_trace/e1_closed_seed4126_range400/config.yaml` gives
`v_max_kmh = 70.0` / `v_max_mps = 19.444444`, matching the SUMO network's
`speed="19.44"` on every edge and the one `vType`'s `maxSpeed`. Coverage is a
fixed 1-D span `[rsu_x - radius, rsu_x + radius]` = `[200, 1000]` m on a
1200 m road, so **every** crossing covers exactly `2 x radius = 800` m,
regardless of direction.

```
R_i(t) = window_start(client_id, t) + 2*radius_m/v_max_mps - t
       = window_start(client_id, t) + 41.142857 s - t
```

`window_start()` (new, mirrors the existing `window_end()`) needs only the
entry time of the client's *current* window — already-observed information,
never the vehicle's true exit.

**Guaranteed conservative**: SUMO enforces `speed <= v_max_mps` everywhere, so
the true crossing time is always >= 41.142857 s — R_i can never overestimate
remaining dwell. Verified against the live `input/coverage_n20r400.csv`
(10,180 windows): holds for 10,166/10,180 (99.86%). The 14 exceptions are all
`window_no` at a run's `t=0` clock start (`t_enter_s` clipped to 0 by
`remap_fleet.py`'s `write_coverage()`) — argued in `window_start()`'s
docstring to be *more* realistic, not less, since a real RSU starting at t=0
couldn't have seen an earlier entry either. Bounded to a handful of clients,
once, near round 1 of each run; not engineered around, logged if it matters.

### Code changed

| file | change |
|---|---|
| `sumo_close_loop/sumo_mobility.py` | `CoverageOracle.window_start()` (new), `_current_window_index()` helper shared with `window_end()` |
| `sumo_close_loop/stage0_filter.py` | new — `Stage0FeasibilityGate`, `Stage0Config`, `resolve_stage0_config`, `resolve_t_hat_source`, `FLEET_MEAN_T_HAT_S = 8.323077` |
| `sumo_close_loop/sumo_server.py` | `install_sumo(..., stage0=None)` wires the gate between `RsuSelectionGate` and the arm's own strategy; `server_will_close()` logs the Stage 0 tally |
| `sumo_close_loop/sumo_gates.py` | `RsuSelectionGate`'s per-round log line is now unconditional (was: only when coverage filtered someone) and carries `round=`, so `eligible_after_busy` exists for every round, joinable with Stage 0's own line |
| `sumo_close_loop/sumo_fedasync.py` | reads `[sumo] stage0_t_hat` / `coverage_radius_m` / `v_max_mps`, passes a `Stage0Config` into the server |
| `sumo_close_loop/make_stage5_configs.py` | new — generates the 6 run configs (mirrors `make_stage4_task4_configs.py`) |
| `sumo_close_loop/run_stage5.sh` | new — parallel runner + checkpoint cleaner + `.done` markers (mirrors `run_sumo.sh`) |

`stage0=None` (every config written before this stage existed, including all
8 Task 4 controls) leaves the selection pipeline byte-identical to before —
verified by `TestStage0Wiring.test_absent_by_default` /
`test_none_source_is_also_absent`.

### T̂ sourcing reuses existing machinery, not new code

- `fleet_mean` -> the constant above.
- `tier_spec` / `oracle` -> **the same existing call**,
  `gpu_tier_sleep.gpu_tops_predict_seconds(client_id, dataset_size, epochs)`.
  Which belief it returns is entirely `[gpu_tier] tiers_csv` /
  `oracle_sleep_csv` in that run's own config — already how the `gpu`/`cpu`/
  `observed` arms get their *scoring* belief, so Stage 0's belief and the
  arm's own belief are guaranteed to agree without any new plumbing. This is
  what makes "cpu's own (wrong) tier prediction" (run #5) automatic: cpu's
  config already points `tiers_csv` at `tiers_cpu.csv`.

### Tests

31 new tests (`test_stage0_filter.py`, plus additions to
`test_sumo_mobility.py`, `test_sumo_gates.py`, `test_sumo_server.py`),
covering: `window_start()` against a toy trace and the real n20r400 trace;
config resolution and its fail-loud guards; R_i as a lower bound; T̂
sourcing per mode; rejection, acceptance, and — separately — the fallback
rule (predicate must never idle a round it could fill); counter accumulation;
log line format; wiring order and backward compatibility.

```
130 passed in ~10s   (sumo_close_loop/tests/)
```

Also smoke-tested all 6 real configs against the live PLATO server for
30-45 s each (not part of the pytest suite — a real-infra check): every run
boots, the `[Stage0]` log lines carry the right per-`stage0_t_hat` values
(verified against `sleep_times.csv` for `oracle`, against `manifest.json`
`T_tier_s` for `tier_spec`, against `FLEET_MEAN_T_HAT_S` for `fleet_mean`),
and `cpu_tier_spec`'s belief collapses xavier/orin to the same predicted time
— expected, since `tiers_cpu.csv`'s xavier/orin `gpu_tflops_fp16` are
identical (0.002) before anchoring.

---

## Step 2 — the six runs. DONE.

All 6 run in parallel on one GPU (Task 4's own precedent — "the reconstructed
clock removes host contention" — plus headroom: ~20 GB free, ~2.2 GB/arm
measured in Task 4's HANDOFF.md). Each writes to
`runs/sumo_close_loop/stage5_<name>/`, logs to
`sumo_close_loop/logs/stage5_<name>.log`, with its own checkpoint cleaner —
`run_stage5.sh all`.

Task 4's own 8 runs at this cell (N=20, C=4, 4000 rounds) took **~9h50m**
wall-clock each (measured from `logs/gpu_n20r400_coverage_gated.log`'s first
to last timestamp). Expect a similar order of magnitude here; this section
will be updated as runs finish (`.done` markers /
`grep '\[Stage0\] summary'`).

Launched 2026-09-19 10:07 UTC, all 6 sharing one GPU (`run_stage5.sh all`).

**Status as of 2026-09-19 13:57 UTC** (~3h50m elapsed, no errors/tracebacks in
any log):

| run | round / 4000 | stage0_fallback rate | note |
|---|---|---|---|
| baseline_fleet_mean | 2220 (55%) | 0.45% | |
| baseline_tier_spec | 2267 (57%) | 0.22% | |
| baseline_oracle | 2265 (57%) | 0.26% | |
| gpu_tier_spec | 2260 (57%) | 0.49% | |
| cpu_tier_spec | 2239 (56%) | **2.37%** | highest, as expected — cpu's belief barely distinguishes xavier/orin (`tiers_cpu.csv` gives both ~equal `gpu_tflops_fp16`), so the predicate empties the pool more often |
| observed_fleet_mean | 2237 (56%) | 0.49% | |

**Final: all 6 reached round 4000 cleanly, 2026-09-19 16:50–16:57 UTC
(~6h45m–6h50m each). No errors or tracebacks in any log.**

---

## Steps 3-5 — analysis. Preliminary pass done (`analyze_stage5.py`).

New analyzer, same shape as `analyze_stage4_task4.py` but self-contained (see
its own module docstring for why) — parses all 6 server logs (dispatch/merge/
clock reconstruction identical to Task 4's method, plus the new `[Stage0]`
round- and selection-level lines), rebuilds `sim_time_reconstructed` via the
same C-slot queue replay, and reuses Task 4's own `task4_stepD.json` for the
four `/none` controls rather than re-deriving them (Step 2 forbids re-running
them anyway). Output: `stage5_analysis.json`.

Validation pass: every dispatch's `t_hat_used`/`R_i_selected` matched to its
`[Stage0]` log line (4111/4111, 4019/4019, ... — 100% across all 6), no
unreported dispatches, no parser problems.

### Stop conditions — all clear

| condition | threshold | worst observed | run |
|---|---|---|---|
| `stage0_fallback` rate | < 10% | 2.02% | cpu_tier_spec |
| conservation law error | < 3% | 1.08% | observed_fleet_mean |
| median `eligible_after_stage0` | ≥ C = 4 | 4.0 (exactly at the floor, one window) | cpu_tier_spec, rounds 1000-2000 |
| plateau vs Task 4 range (82.7-83.5%) | ≤ 2 pts below | 82.10% (gpu_tier_spec, 0.6 pt below the low end) | gpu_tier_spec |

None trip. **Six runs are usable as designed.**

### Table G — conservation law

| run | measured T | predicted T | rel. error |
|---|---|---|---|
| baseline_fleet_mean | 5987.0 s | 6032.7 s | +0.76% |
| baseline_tier_spec | 4388.2 s | 4407.4 s | +0.44% |
| baseline_oracle | 4401.4 s | 4441.8 s | +0.92% |
| gpu_tier_spec | 4911.1 s | 4922.3 s | +0.23% |
| cpu_tier_spec | 6275.3 s | 6304.1 s | +0.46% |
| observed_fleet_mean | 5958.2 s | 6022.7 s | +1.08% |

Law holds with the filter installed, at the same tightness Task 4 measured
without it.

### Table A — the information ladder (runs 1/2/3 vs `baseline/none`, 7525.2 s)

| run | T @ 4000 | vs control | waste | useful/s, last window |
|---|---|---|---|---|
| control (`baseline/none`) | 7525.2 s | — | 13.2% | — |
| 1 fleet_mean | 5987.0 s | **-20.4%** | 2.6% | 0.68/s |
| 2 tier_spec | 4388.2 s | **-41.7%** | 0.3% | 0.92/s |
| 3 oracle | 4401.4 s | -41.5% | 0.3% | 0.92/s |

Selector held constant (`baseline`, random) across this table, so every
difference is the filter alone. `tier_spec` and `oracle` land almost on top
of each other — the spec-derived belief is already close to the oracle's
upper bound here.

### Table B — does the filter effect decay? (THE table this stage exists for)

Gap vs each run's own `/none` control, at rounds 250 / 1000 / 4000:

| run | @250 | @1000 | @4000 | decaying? |
|---|---|---|---|---|
| baseline_fleet_mean | -16.2% | -14.8% | -20.4% | **no — grows** |
| baseline_tier_spec | -41.0% | -36.2% | -41.7% | **no — grows** |
| baseline_oracle | -36.3% | -36.3% | -41.5% | **no — grows** |
| gpu_tier_spec | -24.4% | -34.1% | -38.6% | **no — grows** |
| cpu_tier_spec | -19.8% | -26.0% | -35.7% | **no — grows** |
| observed_fleet_mean | -10.5% | -18.3% | -27.1% | **no — grows** |

**Prediction 2 holds, unambiguously, for all six runs**: the round-4000 gap
is never smaller than the round-1000 gap — every single run's advantage
*grows* over the course of it, the opposite of the scoring gap's decay in
Task 2/Stage 4. This is the strongest single result of the stage.

### Table D — the wrong spec (run 5 vs run 4 vs `cpu/none`)

| | cpu_tier_spec | gpu_tier_spec | cpu/none control |
|---|---|---|---|
| T @ 4000 | 6275.3 s | 4911.1 s | 9758.5 s |
| waste | 1.3% | 0.5% | 16.7% |
| stage0_fallback rate | 2.02% (highest of all 6) | 0.50% | — |
| xavier/orin dispatch share (2000-4000) | 36.2% / 39.4% | 28.0% / 48.6% | — |

The wrong spec still cuts waste hard (16.7%→1.3%) but stays the slowest of
the three `tier_spec` runs throughout, and dispatches far more xavier / far
less orin than `gpu` — its predicate can't tell xavier from orin apart
(`tiers_cpu.csv` gives both ~equal `gpu_tflops_fp16`), so it under-protects
against exactly the pairing it can't distinguish. Consistent with Prediction
5 ("kills the most viable work and does not recover").

### Table E — the reactive competitor (run 6 vs run 1 vs `observed/none`)

| | observed_fleet_mean | baseline_fleet_mean |
|---|---|---|
| T @ 4000 | 5958.2 s | 5987.0 s |
| waste | 2.0% | 2.6% |

0.5% apart at round 4000 — Prediction 6 holds cleanly: the filter's effect
does not depend on which selector sits in front of it.

### Prediction scorecard (Step 5)

| # | prediction | result |
|---|---|---|
| 1 | `baseline+tier_spec` 12-18% faster than control | **partially — direction right, magnitude far exceeds prediction**: measured -41.7%, more than double the predicted range |
| 2 | filter gap does not decay | **holds, all 6 runs, unambiguously** — the central claim |
| 3 | `baseline+fleet_mean` captures <50% of run 2's gain; kills ≈15% of viable thor work | gain-capture: **holds, right at the edge** (20.4/41.7 = 49.0%). thor viable-killed: **methodology mismatch, see note below** — cannot confirm/deny on a like-for-like basis yet |
| 4 | `gpu+tier_spec` ≈ `baseline+tier_spec` @ 4000, ahead early | **does not hold as stated**: gpu is *slower* at round 100 (+21.6%) and round 4000 (+11.9%); it is only briefly ahead in the 500-1000 round window (-3 to -6%) |
| 5 | `cpu+tier_spec` kills the most viable work, does not recover | **holds** — highest fallback rate, stays the slowest of the three tier_spec runs throughout |
| 6 | `observed+fleet_mean` ≈ `baseline+fleet_mean` | **holds cleanly** — 0.5% apart at round 4000 |

Reported as-is, per the prompt's own rule: a failed prediction is a result,
not something to reconcile. Prediction 4 in particular is a genuine surprise
worth sitting with, not explaining away: the `gpu` selector's own scoring,
on top of the filter, is *not* free — it is measurably slower than doing
nothing but randomly selecting among the filtered pool, both early and late.

### Methodological note: Table F's confusion matrix is NOT the same measurement as Step 0's pre-check

Per-tier `killed_viable` fractions from the live runs' own rejection events
(reconstructed the same way Task 4's Step D.4 did: from `CoverageOracle` plus
each client's true `sleep_times.csv` value) come out at **50-80%**, far above
Step 0's pre-check range (8-18%). This is **not a contradiction to resolve**
— it is two different denominators:

- **Step 0's pre-check** asked, for every dispatch that ACTUALLY HAPPENED in
  Task 4's unfiltered log, "would the filter have rejected this one?" — one
  judgment per real dispatch (~4,000-4,800 per run).
- **This live measurement** asks, for every CANDIDATE the predicate looked at
  each round (most of whom were never going to get the one open slot that
  round regardless of the filter, since only ~1 slot opens most rounds against
  5-15 eligible candidates), "would this one have finished in range if it HAD
  been dispatched?" — one judgment per candidate-round (~14,000-20,000 per
  run), the only confusion matrix a live run's own log can support, since
  there is no unfiltered twin dispatch sequence to replay against.

Both are legitimate; they answer different questions ("how good is the
predicate's judgment on the actual dispatch slot" vs "how good is its
judgment on everyone it's ever asked to judge"). Full per-tier table:

| run | xavier viable_killed | orin | thor |
|---|---|---|---|
| baseline_fleet_mean | 29.8% | 70.4% | 70.0% |
| baseline_tier_spec | 71.7% | 52.3% | 48.3% |
| baseline_oracle | 69.3% | 50.6% | 49.3% |
| gpu_tier_spec | 66.7% | 52.4% | 50.2% |
| cpu_tier_spec | 57.6% | 81.4% | 73.8% |
| observed_fleet_mean | 28.4% | 71.4% | 70.9% |

waste_caught is >98% for every tier in every run regardless of source — the
filter essentially never misses real waste at the candidate-pool level; its
entire cost is in over-rejection (the killed_viable side), consistent with
Table A/B's picture of a strong, non-decaying effect bought at a real but
bounded viable-work cost.

### Time-to-accuracy and waste, all six runs (plus Task 4 controls for reference)

TTA in reconstructed hours (`sim_time_reconstructed`, not PLATO's raw
`elapsed_time` — see Step 1 caveat on host contention under 6-way GPU
sharing).

| run | waste | plateau | TTA→70% | TTA→75% | TTA→78% | TTA→80% |
|---|---|---|---|---|---|---|
| baseline_fleet_mean | 2.60% | 83.07% | 0.085h | 0.166h | 0.243h | 0.426h |
| baseline_tier_spec | 0.32% | 82.70% | 0.049h | 0.108h | 0.229h | 0.322h |
| baseline_oracle | 0.35% | 82.89% | 0.096h | 0.121h | 0.225h | 0.345h |
| gpu_tier_spec | 0.50% | 82.10% | 0.072h | 0.135h | 0.195h | 0.296h |
| cpu_tier_spec | 1.26% | 83.71% | 0.094h | 0.154h | 0.235h | 0.405h |
| observed_fleet_mean | 1.96% | 83.08% | 0.090h | 0.146h | 0.249h | 0.344h |
| *baseline/none (control)* | *13.17%* | *82.88%* | *0.114h* | *0.189h* | *0.368h* | *0.491h* |
| *gpu/none (control)* | *14.76%* | *82.98%* | *0.093h* | *0.148h* | *0.242h* | *0.406h* |
| *cpu/none (control)* | *16.71%* | *82.68%* | *0.099h* | *0.226h* | *0.384h* | *0.603h* |
| *observed/none (control)* | *13.95%* | *83.51%* | *0.081h* | *0.160h* | *0.253h* | *0.418h* |

Waste drops by an order of magnitude in every filtered run; TTA→80% improves
against each run's own control in 5 of 6 cases (`baseline_oracle`'s 0.345h is
the one exception, and only barely, against `baseline/none`'s 0.491h — still
faster, just the smallest margin of the six). Plateau accuracy is unmoved by
the filter either way (82.1-83.7%, same band as the controls) — the filter is
buying time, not accuracy, exactly as the stage's design intends.

### Crossover rounds — the full picture behind Prediction 4

`crossover_round()` (Task 4 Step D's own definition: first round after which
an arm is slower than its comparison and stays slower for 200 rounds) gives
a single number that hides more than it shows here, so both are reported:
the single crossover number, and the full sequence of sign-changes in
`run.clock - comparison.clock` across all 4,000 rounds.

| pair | `crossover_round` | full picture |
|---|---|---|
| gpu_tier_spec vs baseline_tier_spec | 50 | slower rounds 3-267, **ahead rounds 267-1152** (874 rounds, its only lead), permanently slower again from round 1153 to 4000 |
| cpu_tier_spec vs baseline_tier_spec | 50 | slower for essentially the entire run (1-4000) — never ahead |
| cpu_tier_spec vs gpu_tier_spec | 50 | slower rounds 1-12, a 2-round blip ahead (12-14), slower again 14-4000 |
| observed_fleet_mean vs baseline_fleet_mean | *(none — never holds 200 rounds early)* | noisy flips through round 76 (small-sample), then **consistently faster from round 76 to 4000** |

`crossover_round=50` for the three `tier_spec` pairs is technically correct
by its own definition but easy to over-read: it is *not* the point where
`gpu_tier_spec` permanently falls behind `baseline_tier_spec` — that is round
1153. Between rounds 267 and 1152, `gpu_tier_spec` genuinely was ahead (this
is where Table B/Prediction 4's "ahead of it early" partially came from), it
just did not last. `cpu_tier_spec` never leads `baseline_tier_spec` at any
point in 4,000 rounds — the cleanest, starkest confirmation of Prediction 5
in the whole dataset. `observed_fleet_mean`'s small, consistent lead over
`baseline_fleet_mean` from round 76 onward is a real structural signal, not
noise, but it is small (0.5% at round 4000, Table E) and does not contradict
Prediction 6's "≈".

### Redispatch gap (Task 4 Step D's feedback-loop check, STATUS.md §5.4)

Cycles until the same client is dispatched again, after a wasted dispatch vs
after a useful one. A large `ratio` would mean a client whose uploads keep
failing is not being avoided (or is even being *sought out* by a scoring
feedback loop) — the concern Task 4 checked for `baseline`/`gpu`/`cpu`/
`observed` without a filter.

| run | after waste | after useful | ratio | n (wasted) |
|---|---|---|---|---|
| baseline_fleet_mean | 35.0 | 19.5 | 1.79 | 106 |
| baseline_tier_spec | 51.5 | 19.8 | 2.61 | 13 |
| baseline_oracle | 34.9 | 19.9 | 1.76 | 14 |
| gpu_tier_spec | 38.0 | 19.9 | 1.91 | 20 |
| cpu_tier_spec | 28.3 | 19.8 | 1.43 | 51 |
| observed_fleet_mean | 33.9 | 19.6 | 1.73 | 80 |

Every ratio is > 1 (a wasted client is redispatched *less* often, not more) —
no sign of the feedback loop Task 4 checked for, with or without the filter.
`n` is small for the `tier_spec`/`oracle` runs (13-20 wasted dispatches out
of ~4,000), so their ratios carry more sampling noise than `baseline_fleet_
mean`/`observed_fleet_mean`'s (80-106 wasted dispatches) — read the exact
ratio value with that in mind, the direction (>1, no loop) is what's solid.

### Table F — filter mechanics, per window

**Fallback rate per window** (all six comfortably under the 10% stop
condition in every window, not just on average):

| run | 1-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 |
|---|---|---|---|---|---|
| baseline_fleet_mean | 0.00% | 0.00% | 0.40% | 0.70% | 0.70% |
| baseline_tier_spec | 0.00% | 0.00% | 0.00% | 0.50% | 0.50% |
| baseline_oracle | 0.00% | 0.00% | 0.00% | 0.60% | 0.80% |
| gpu_tier_spec | 0.00% | 0.00% | 0.00% | 1.00% | 0.50% |
| cpu_tier_spec | 1.20% | 1.20% | 2.80% | 3.20% | 1.45% |
| observed_fleet_mean | 0.00% | 0.00% | 0.20% | 0.60% | 1.00% |

Fallback is essentially zero in the first 500 rounds for every run except
`cpu_tier_spec` (which is never zero, consistent with it having the
predicate that discriminates tiers worst) and rises somewhat in the back
half for everyone — plausibly the mix drifting toward more xavier as the
run's own G_i feedback plays out (Task 2's mechanism), shrinking the
in-range-now-eligible pool the filter has to work with, though this dataset
doesn't yet isolate that cause from others.

**Pool-size funnel, median clients at each gate, per window** (busy → in
range now → passes Stage 0):

| run | window | after busy filter | after range gate | after Stage 0 |
|---|---|---|---|---|
| baseline_fleet_mean | 1-250 → 2000-4000 | 17 (flat) | 10 (flat) | 7 → 6-7 |
| baseline_tier_spec | 1-250 → 2000-4000 | 17 (flat) | 10 → 9-10 | 7 → 5-6 |
| baseline_oracle | 1-250 → 2000-4000 | 17 (flat) | 9-11 | 6-7 |
| gpu_tier_spec | 1-250 → 2000-4000 | 17 (flat) | 9-11 | 6-7 |
| cpu_tier_spec | 1-250 → 2000-4000 | 17 (flat) | 10 (flat) | **5 → 4-5** |
| observed_fleet_mean | 1-250 → 2000-4000 | 17 (flat) | 10-11 | 6-8 |

`eligible_after_busy` is flat at 17 for every run in every window (C=4 slots
busy out of 20 minus a small pending-merge margin — mechanical, not a filter
effect). The range gate removes another ~30% (17→~10), and Stage 0 removes a
further ~30-40% on top of that. `cpu_tier_spec` runs the tightest funnel of
the six, ending at a median of exactly 4 — the concurrency floor — in the
1000-2000 window, which is why it also has this dataset's highest fallback
rate; still never below the floor, so the stop condition never trips, but it
is the run with the least headroom.

**Confusion matrix per tier per window** is in `stage5_analysis.json`
(`stage0_confusion_by_window`) for all 6 runs × 5 windows × 3 tiers — too
large to usefully inline here. The aggregate version (all rounds pooled) is
already in the Table F methodological note above. Checked the per-window
breakdown before writing this up rather than assuming it matches the
aggregate: waste_caught is NOT uniformly >98% in every individual cell — the
single worst is `baseline_tier_spec`, window 1-250, thor: 80.0% (8 killed_
waste / 2 missed_waste, n=10). That window-tier cell is a small-sample
corner (only 10 thor dispatches existed in the first 250 rounds of that run)
rather than a real early-run weakness; the aggregate >98% figure is not
being carried by hiding a large early or late shortfall, but it is an
average over cells with very different sample sizes, and the smallest of
them are noisier than the headline number suggests.
