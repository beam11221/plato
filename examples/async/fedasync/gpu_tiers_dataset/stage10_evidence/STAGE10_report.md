# Stage 10: Stage 0 filter + scoring in the open population

Spec: `prompts/STAGE10_filter_open.md`. Seed ledger as in Stages 8–9: physics
seed pinned at 4126; run seed 4126 / 4127 / 4128. Nothing under `plato/` is
touched.

## Status

| step | state |
|---|---|
| Step 0: audit of the open-loop Stage 0 path | **done**: filter absent; the closed-loop gate would be an H3 sibling; positions available (§0) |
| Step 0 fixes + Step 1 (`observed_per_vehicle`) + tests | **done**, 70/70 tests (§1) |
| Step 2: pre-flight | **PASSED, every check 100%**: f1 (8 runs) and f10 (6 runs), final code (§2) |
| Wave A (30 runs) | **done** 2026-09-29 12:05 UTC, 0 failures, no stop; `STAGE10A_report.md` written automatically and reviewed |
| Wave B (18 runs) | **done** 2026-09-30 12:25 UTC, 0 failures, no stop (§5) |
| Final analysis (48 runs) | **done**: `STAGE10_final_report.md`, all 90 runs in `stage10_all_runs.csv`. **Stage 10 complete** |

---

## 0. Step 0 audit (2026-09-27)

### 1. Does Stage 0 exist on the open-loop path?

**No.** `sumo_open_server.py` states it is absent on purpose ("This cell tests
the scoring claim"). `make_stage7_configs.py` writes no `stage0_*` key, and
`sumo_open_fedasync.py` never builds a gate. The only implementation is the
closed loop's `sumo_close_loop/stage0_filter.py::Stage0FeasibilityGate`, wired
in `sumo_server.install_sumo()`.

### 2. Every lookup that gate makes, and what it keys on

| lookup | closed-loop code | keyed on | in the open population this would be |
|---|---|---|---|
| position `x_i` | `PositionOracle.x_at(client_id, t)` | `client_id` | **the slot**, not the vehicle in it |
| direction | `PositionOracle.direction_at(client_id, t)` | `client_id` | the slot |
| `T̂` (tier_spec) | `gpu_tier_sleep.gpu_tops_predict_seconds(client_id, …)` → `tier_for(client_id)` → `clients_n20r400.csv` | `client_id` | **H3 exactly**: the slot's fleet-file row, which matches the occupant ~37% of the time |
| tier in logs | `gpu_tier_sleep.tier_for(client_id)` | `client_id` | H3 (the log would mislabel) |
| `fleet_mean` `T̂` | constant 8.323077 | — | not keyed |
| coverage boundary | `rsu_x_m ± radius_m`, by direction | config | fine |

Porting the gate unchanged would reproduce H3 in the filter. **It is not
reused.** The open-loop gate is a new class that reads everything through the
7b occupant abstraction:

- **tier and `T̂`** from `IdentityResolver.of_slot(k)`: the trace vehicle at
  `p_recur = 0`, the slot's fixed fleet identity at `p_recur = 1`;
- **position and direction** from the **trace vehicle physically in the slot**
  (`timeline.vehicle_in_slot(k)`), in both regimes. At `p_recur = 1` the trace
  supplies coverage timing, and the physical vehicle's position is what decides
  how long that slot stays in range. The fleet identity has no position of its
  own.

### 3. Where `R_i` gets `x_i`: positions are available at Stage 5b's resolution

The open trace carries `trace.csv` (731,142 rows) with
`t, veh_id, x_m, direction, in_coverage, dist_to_rsu_m, …`. Checked:

- **1 Hz** (`trace_dt_s: 1.0`): all 474,925 consecutive in-coverage sample pairs
  are exactly 1.0 s apart. That is the same resolution as Stage 5b's
  `positions_n20r400.csv`.
- 485,443 in-coverage rows over 10,518 vehicles. Each vehicle's
  in-coverage span matches its `coverage_windows.csv` window to within 1 s,
  for all vehicles.
- 1-D geometry: `|x_m − 600| == dist_to_rsu_m` on every row. `rsu_x_m = 600`,
  `coverage_radius_m = 400` and `v_max_mps = 19.444444` (70 km/h) come from
  the trace's `config.yaml`, the same values as Stages 5b / 6.
- Direction: every `fEB.*` row is +1 (exit at x = 1,000), every `fWB.*` row
  is −1 (exit at x = 200).

So `R_i = |x_exit(direction) − x_i(t)| / v_max` is computed from the actual
position, as in Stage 5b. **No stop.** No elapsed-time substitute is used.

### 4. Gate order

The closed loop runs `eligible → busy → in-range-now (RsuSelectionGate) →
Stage 0 → arm score → lottery`.

In the open loop, `OpenSlotSelectionGate` is the in-range-now gate: a slot is
occupied exactly while its vehicle is in coverage. It already takes the pool
PLATO offers (busy clients and reported-not-merged clients excluded) and
passes the eligible slots and the open dispatch count inward. Stage 0 is
inserted **between** it and the arm's strategy, the same position as in the
closed loop. `T_comm = 0` (the predicate is `R_i ≥ T̂_i`, with no comm term).
The fallback is Stage 5's rule: if nothing passes, the unfiltered pool is used
for that round and `stage0_fallback=1` is logged.

A **second sibling** found in the audit: `OpenPopulationMixin._polaris_selector()`
looks for the Polaris strategy only one level inside the slot gate. With Stage
0 in between, it would find the gate instead. The predictor install, the
unseen-τ rule and the `tau_hat` logging would then be skipped **silently**. It
is fixed to walk the `.inner` chain.

---

## 1. Step 0 fixes and Step 1: what was built

**New module `sumo_open_loop/open_stage0.py`**, the open-population gate. It
does **not** reuse the closed-loop class (see §0).

- `resolve_open_stage0([sumo_open])`: `stage0_t_hat` ∈ {`none` (default),
  `tier_spec`, `observed_per_vehicle`}. With the gate on, it requires
  `stage0_r_i = "vmax_distance"`, `coverage_radius_m`, `v_max_mps`, `rsu_x_m`
  and `positions_csv`. Anything missing or different raises.
- `load_trace_positions(trace.csv)`: the trace's 1 Hz in-coverage samples,
  keyed on `veh_id`, loaded into the **same** `sumo_position.PositionOracle`
  class Stage 5b used (linear interpolation, no extrapolation across gaps).
- `OpenStage0Gate`: per candidate slot,
  - `identity = resolver.of_slot(k)`, the tier and `T̂` source;
  - `veh = timeline.vehicle_in_slot(k)`, the position;
  - `R_i = |x_exit(dir) − x_veh(t)| / v_max`;
  - pass iff `R_i ≥ T̂`. With no passes, it falls back to the unfiltered pool.
  It logs one `cand` line per candidate (t, slot, veh, identity, tier, `T̂`,
  basis, `R_i`, pass), the Stage 5 round line
  (`eligible_after_range_gate`, `eligible_after_stage0`, rejected ids,
  `stage0_fallback`), a `selected` line per dispatch, and a summary.

**`T̂` sources:**
- `tier_spec`: `predict_seconds_for_tier(occupant tier)`, the same function
  the scoring uses. It is the GPU spec for `gpu` / `baseline` (15 / 1.923077 /
  1.0 s) and `tiers_cpu.csv` for `cpu`.
- `observed_per_vehicle` (Step 1): the identity's own mean realised duration,
  or the fleet running mean for a never-seen identity (8.323077 s before the
  first report). It never touches spec. The durations
  (`report arrival − dispatch` on the server clock, `T_comm = 0`) are
  recorded in `process_client_info` **only for delivered reports**, so a
  wasted dispatch contributes nothing. The report line now logs `duration=`
  so the check can be recomputed.

**Server wiring (`sumo_open_server.py`):**
- `install_open_population(…, stage0, positions)` builds
  `OpenSlotSelectionGate(OpenStage0Gate(arm strategy))`. That is Stage 5's order.
- `stage0=None` or a disabled gate leaves the chain exactly as in Stages 7–9.
- `_polaris_selector()` now walks the `.inner` chain. This fixes the audit's
  second sibling.
- `server_will_close` logs the gate summary.
- The entry point reads the config and loads positions only when the gate is on.

**Tests: 17 new in `tests/test_stage10_filter.py`, 70/70 in total.** They use
the real `PositionOracle`, selector, resolver and mixin methods, and pin:
- absent key → no gate; missing geometry, a non-`vmax_distance` R_i or an
  unknown source → raises;
- R_i for EB (exit 1,000) and WB (exit 200);
- `tier_spec` reads the **occupant's** tier, not the slot's CSV row (the H3
  regression);
- at `p_recur = 1`, the fleet identity's tier and the trace vehicle's position;
- `observed_per_vehicle` covers never-seen → 8.323077, own mean and fleet
  mean, and **no spec is ever read** (the spec function is patched to raise);
- the predicate and the fallback;
- `_polaris_selector` finds the selector two levels down, and nothing for
  baseline.

**Configs** (`make_stage7_configs.py --stage 10 --variant {f1,f10,nf} --seed N`):

| variant | configs | contents |
|---|---|---|
| f1 | `configs/stage10f1_s<seed>/` | filter on, α = 1 + running-mean τ̂ for the Polaris arms; baseline / gpu / cpu / observed |
| f10 | `configs/stage10f10_s<seed>/` | filter on, α = 10 (Polaris default package); gpu / cpu / observed |
| nf | `configs/stage10nf_s<seed>/` | pure random baseline |

- Each differs from its Stage 9 / 8 / 7b partner only in run dir, port and
  the Stage 0 block.
- The 7b and 9 configs regenerate byte-identically.
- No port is duplicated.

---

## 2. Step 2: pre-flight (seed 4126, 100 rounds)

`preflight_stage10.py` runs the Stage 7b and Stage 9 checks unchanged and
then checks **every candidate the gate considered**, recomputed from outside
the server:

- tier;
- `T̂` against the spec, or against the running means rebuilt from the
  `duration=` fields;
- `R_i` from the trace position;
- pass;
- the round-line counts and fallback flag;
- selected slots passed and carry the same `T̂` / `R_i`;
- plus the stop thresholds: fallback < 10%, median `eligible_after_stage0` ≥ 4.

### First pass (8 f1 runs)

Every check was 100% (Stage 7b / 9 all 100%; tier, `T̂`, pass, round lines,
selection, fallback and pool size all 100%) **except** "R_i recomputed ==
logged", at 20–27%. Diagnosis:

- **Log precision, not the gate.** The largest disagreement over 9,663
  candidates was 0.000498 s. That is exactly the bound from logging `t` at 3
  decimals (R_i changes by at most 1 s per s, so ±0.0005 s). The line now logs
  `t` at 6 decimals. This is a logging change only.
- **R_i undefined in 256 / 9,663 candidates (2.6%)**, always in the **last
  second** before the vehicle's window closes, after its last 1 Hz in-coverage
  sample. The oracle never extrapolates, so the gate rejects. The true R_i
  there is < 1 s, below every `T̂`, so the outcome is the one a position
  would give. This is the closed-loop gate's behaviour too. The checker now
  requires both sides to agree it is undefined, and requires it to happen only
  in a window's last second.

Pool sizes from that pass (median [p10, p90]):

| arm | after range gate | after Stage 0 | rejected | fallback |
|---|---|---|---|---|
| baseline+F open | 12 [11, 14] | 10 [8, 12] | 18.7% | 0% |
| gpu+F open | 12 [11, 14] | 10 [8, 12] | 16.3% | 0% |
| observed+F open | 12 [11, 14] | 10 [8, 13] | 17.4% | 0% |
| cpu+F open | 12 [10, 14] | **7** [5, 8] | 45.1% | 0% |
| baseline+F closed | 12 [11, 14] | 10 [7, 12] | 20.6% | 0% |
| gpu+F closed | 12 [11, 14] | 9 [7, 11] | 25.6% | 0% |
| observed+F closed | 12 [10, 14] | 10 [8, 11] | 17.2% | 0% |
| cpu+F closed | 12 [11, 14] | **7** [5, 9] | 42.6% | 0% |

No fallback anywhere, and every median ≥ C = 4. `cpu`'s spec over-predicts
the fast tiers, so it rejects the most.

The first-pass logs are in `logs/superseded_stage10/`. The second pass (f1
and f10, final code) is below.

### Safeguards in the Wave driver (`run_stage10.sh`)

- A four-slot queue in the spec's launch order: open baseline+F / gpu+F, then
  the rest of open, then closed.
- Every job runs in its own process group, with its own cleaner.
- Checkpoints are deleted as soon as a run finishes.
- After every run, `stage10_runcheck.py` checks the stop conditions:
  - conservation ≤ 3%;
  - plateau ≥ 80.1%;
  - fallback ≤ 10% and median pool ≥ 4;
  - predictor agreement 100%;
  - occupancy identical to 7b over the common prefix;
  - for `baseline / none`, T within 3% of 7b / 8.
  A stop kills every running job group. A baseline drift blocks Wave B.
- Wave B starts only after `STAGE10A_report.md` is written: the driver waits
  for `logs/stage10_waveB.go`, which is written once the report is done.

### Second pass (final code): f1 × 8 and f10 × 6, seed 4126, 100 rounds

**PRE-FLIGHT PASSED, every check 100% in all 14 runs**
(`stage10f1_s4126_preflight.json`, `stage10f10_s4126_preflight.json`).

- Every Stage 7b and Stage 9 check passes, in every run.
- Every Stage 10 check passes:
  - filter tier == occupant tier;
  - `T̂` == spec value, or == `observed_per_vehicle` recomputed from the
    delivered durations;
  - `R_i` == `vmax_distance` from the trace position (to 1e-5 s);
  - pass == (`R_i ≥ T̂`);
  - round-line counts and fallback flag;
  - selected slots passed and carry the candidate's `T̂` / `R_i`;
  - undefined `R_i` only in a window's last second (32–41 per run);
  - fallback < 10%;
  - median pool ≥ 4.

**One checker adjustment.** The f10 run showed the Stage 9 "logged V_i ==
G_i / √(1 + τ̂)" row at 92–99% in `gpu`. The worst mismatch was a relative
1.16e-6. V_i, G_i and τ̂ are each logged to 6 decimals, so at α = 10 (G ≈ 35)
the propagated rounding reaches ~4–7e-6, above the check's fixed 2e-6.

The check now uses the per-dispatch propagated rounding bound
`5e-7 + 5e-7/√(1+τ) + V·5e-7/(2(1+τ))`, with a floor of 2e-6, so it is never
looser than before. With it, f10 passes 100%, and Stage 9's own pre-flight
and f1 still pass 100%.

Pool sizes and filter behaviour (second pass; median [p10, p90]):

| arm | α | regime | after range gate | after Stage 0 | rejected | fallback |
|---|---|---|---|---|---|---|
| baseline+F | — | open | 12 [11, 14] | 10 [8, 12] | 18.2% | 0% |
| gpu+F | 1 | open | 12 [11, 14] | 10 [8, 12] | 18.1% | 0% |
| observed+F | 1 | open | 12 [11, 14] | 10 [9, 12] | 17.4% | 0% |
| cpu+F | 1 | open | 12 [11, 14] | 7 [5, 8] | 45.7% | 0% |
| baseline+F | — | closed | 12 [11, 14] | 10 [7, 11] | 20.2% | 0% |
| gpu+F | 1 | closed | 12 [11, 14] | 9 [7, 11] | 25.0% | 0% |
| observed+F | 1 | closed | 12 [11, 14] | 10 [8, 12] | 18.4% | 0% |
| cpu+F | 1 | closed | 12 [11, 14] | 6 [5, 9] | 44.3% | 0% |
| gpu+F | 10 | open | 12 [11, 14] | 10 [8, 12] | 21.3% | 0% |
| observed+F | 10 | open | 12 [11, 14] | 9 [7, 12] | 24.3% | 0% |
| cpu+F | 10 | open | 12 [10, 14] | 7 [5, 8] | 44.6% | 0% |
| gpu+F | 10 | closed | 12 [10, 14] | 10 [7, 12] | 21.3% | 0% |
| observed+F | 10 | closed | 12 [10, 14] | 10 [8, 11] | 20.8% | 0% |
| cpu+F | 10 | closed | 12 [10, 15] | 7 [5, 9] | 43.4% | 0% |

The fallback rate, the spec's first concern given ~13 vehicles in coverage, is
**0% everywhere**. The smallest median pool after Stage 0 is 6 (cpu), above
C = 4. The p10 is ≥ 5 in every arm.

---

## 3. Waves

The driver is `sumo_open_loop/run_stage10.sh` (log `logs/run_stage10.log`):
a four-slot queue in the spec's launch order.

- **Wave A** (30 runs):
  1. open `baseline+F`, `gpu+F`;
  2. open `observed+F`, `cpu+F`, `baseline/none`;
  3. closed, all five arms.
- **Between waves:** `stage10_waveA_report.py` writes `STAGE10A_report.md`
  and, if nothing is stopped or blocked, the go-file. Wave B then starts
  without waiting.
- **Wave B** (18 runs): f10 gpu / observed / cpu × both regimes × 3 seeds.

Wave A launched **2026-09-27 19:22 UTC**. At start, the first four runs (open
`baseline+F` / `gpu+F`, seeds 4126–4127) were dispatching and gating, with no
tracebacks, the correct run seeds and the gate installed (`tier_spec`), each
with its own cleaner.

### Analysis tooling, ready before the runs finish

- `analyze_stage10.py --wave A|B` builds the Stage 9 per-run set (through
  `analyze_stage9.per_run`, unchanged) and adds the filter set:
  - fallback, rejections;
  - pools per window;
  - rejection accuracy per tier, computed from each occupant's true sleep
    (independent of the log);
  - admitted waste per tier;
  - candidate pass rate per tier;
  - waste / mix split vs `baseline / none`;
  - `T̂` basis per dispatch and the fleet running mean over time;
  - `G_i` x : t vs the no-filter partner;
  - comparisons A–I with per-seed sign checks;
  - the occupancy invariant over every analysed run.

  It was exercised on the pre-flight logs: conservation residual 0.07%, and
  admitted waste 0% in 100 rounds.
- `stage10_waveA_report.py` is called by the driver when Wave A ends. It
  writes `STAGE10A_report.md` and writes the Wave B go-file only if:
  - all 30 runs are present;
  - no run check stopped or blocked;
  - occupancy is identical;
  - `baseline / none` is within 3% of 7b / 8.

  Dry-run into scratch with no Stage 10 data: no errors, and correctly no
  go-file.

---

## 4. STOP (Wave B only): `baseline / none` s4127 differs from Stage 8 by −3.51%

2026-09-28 11:05 UTC. `stage10_runcheck.py` reported the following, and the
driver recorded the block (`logs/stage10_waveB.blocked`):

| re-run of `baseline_open` | original | re-run | gap |
|---|---|---|---|
| Stage 9 invariance, s4126 | 7,238.0 (7b) | 7,144.9 | −1.29% |
| Stage 10 nf, s4126 | 7,238.0 (7b) | 7,138.9 | −1.37% |
| **Stage 10 nf, s4127** | **7,251.2 (Stage 8)** | **6,996.5** | **−3.51% > 3%** |
| Stage 10 nf, s4128 | 7,110.3 (Stage 8) | running | |

The spec says "stop Wave B and report". **Wave A continues**, because the
condition stops only Wave B. When Wave A ends, the driver will write
`STAGE10A_report.md` but no go-file, and exit with `STAGE10_WAVE_B_BLOCKED`.

### What the gap is made of (conservation law, empirical tier sleeps)

| pair | total | mix | waste | xavier share | waste | host processing time, mean / median |
|---|---|---|---|---|---|---|
| s4126, Stage 9 | −1.29% | −0.45 | −0.83 | 34.7 → 34.5% | 13.38 → 12.66% | 0.113 → 0.111 / 0.100 → 0.099 s |
| s4126, Stage 10 | −1.37% | −0.54 | −0.91 | 34.7 → 34.5% | 13.38 → 12.58% | 0.113 → 0.113 / 0.100 → 0.099 s |
| **s4127, Stage 10** | **−3.51%** | **−2.93** | −0.63 | **35.0 → 33.6%** | 13.06 → 12.51% | 0.113 → 0.114 / 0.100 → 0.100 s |

**Evidence against harness drift:**
- The host processing time injected into PLATO's clock is unchanged (0.113 s
  mean, 0.100 s median in every run).
- In s4127, most of the gap is the **mix** channel: random selection happened
  to draw 1.4 points less xavier. For a random selector that is a chance
  outcome. Across seeds, `baseline_open`'s xavier share has SD 0.5 points
  (Stage 8), so 1.4 points is about 2.8 SD, which is unusual but possible.
- Nothing on the `baseline / none` path has changed since 7b in a way that
  reaches selection. The Stage 9 knobs are not read, and the Stage 10 gate is
  not installed. The only additions are counters and log fields.

**Evidence that cannot be dismissed:**
- All three re-runs are **faster** than their originals, and waste is lower
  in all three (by 0.6–0.9 points).
- Three same-sign gaps in three tries is weak evidence of a small systematic
  shift, but it is evidence.
- This cannot be decided from single re-runs: the spread of same-config,
  same-code replicates has never been measured beyond one pair.

### Decision needed before Wave B

1. **Treat it as execution noise and release Wave B.** State that the
   `baseline / none` spread is at least ±3.5%. Every Wave A / B comparison
   against `baseline` then has to clear that noise.
2. **Measure the replicate spread first.** Re-run `baseline_open` s4127 twice
   more under the current code (2 runs, ~5.5 h, alongside Wave A's closed
   runs). If same-code replicates spread about ±3%, it is noise. If they
   cluster near 7,000 s while Stage 8 sits at 7,251 s, the harness has drifted
   and that needs finding.
3. **Hold everything and investigate the code path** between the Stage 8 and
   Stage 10 `baseline / none` runs.

**Recommendation: option 2.** It is cheap, it runs in parallel with Wave A,
and it is the only option that separates noise from drift instead of assuming
one.

---

## 5. Decision (user, 2026-09-29): continue with Wave B after Wave A

The −3.51% `baseline / none` gap (s4127 open) is **accepted as execution
noise**. By the time of the decision, the re-runs spanned **−3.51% to +1.06%**:

| re-run | gap |
|---|---|
| s4126 open | −1.37% |
| s4126 closed | −1.60% |
| s4127 open | −3.51% |
| s4128 open | +1.06% |

- s4128 broke the "all re-runs faster" pattern that had looked systematic.
- The host processing time injected into PLATO's clock is unchanged.
- The gap is mostly the mix channel of a random selector.

**Consequence carried into every result:** `baseline / none` carries at least
**±3.5%** of execution noise, and any comparison against a baseline has to
clear it.

### How it is applied (no change to the running driver)

- `logs/stage10_waveB.accepted` records the accepted run
  (`stage10nf_s4127_baseline_open`) and the reason.
- `logs/stage10_waveB.blocked` was renamed to
  `…blocked.accepted_by_user`, so the driver does not exit at the end of
  Wave A.
- `stage10_waveA_report.py` now reads the acceptance file. An accepted block
  no longer holds Wave B and is listed as such in `STAGE10A_report.md`.
  **Every other condition still gates Wave B:**
  - all 30 runs present;
  - no run-check stop;
  - no *new* unaccepted block, for example if the remaining `baseline / none`
    closed re-runs drift > 3%;
  - occupancy identical.
- Dry run: blocks 0, H ok. It is held only by the 7 Wave A runs still going.

So when Wave A ends, the driver writes `STAGE10A_report.md`, then the go-file,
and starts Wave B without waiting, in the spec's order.
