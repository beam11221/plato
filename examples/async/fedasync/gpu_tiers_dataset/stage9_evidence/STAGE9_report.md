# Stage 9: neutral priors for never-seen vehicles

Spec: `prompts/STAGE9_neutral_prior.md`. Seed ledger as in Stage 8
(`stage8_evidence/STAGE8_report.md`): physics seed pinned at 4126, run seed
4126 / 4127 / 4128.

## Status

| step | state |
|---|---|
| Step 0: log-only check on 7b / 8 | **done**: hypothesis **refuted** (§0) |
| Step 1: knobs + tests | **done**, 53/53 tests (§1) |
| Step 2: pre-flight + baseline invariance run | pre-flight **PASSED 100%**; **invariance FAILED → stop condition, queue stopped 2026-09-25 22:57 UTC** (§4) |
| Step 3: 22 runs | **done**: `STAGE9_ALL_DONE` 2026-09-27 10:47 UTC, 0 failures, all checkpoints deleted |
| Step 4: report | **done** (§7 interim + §8 final) |

---

## 0. Step 0: what `gpu − observed` is made of (Stage 7b / 8 logs, α = 10)

Script: `sumo_open_loop/stage9_step0.py` → `step0_log_check.json`. It reads
the runs through `analyze_stage7` unchanged.

**Hypothesis tested:** `gpu`'s edge is signal precision among the fast tiers.
`observed` cannot separate orin from thor, while the spec can.

### thor : orin dispatch ratio

| cell | gpu | observed | baseline | gpu > observed? | ≥ 1 obs: gpu / observed |
|---|---|---|---|---|---|
| 4126 open | 0.504 | 0.466 | 0.474 | yes | 0.515 / 0.457 |
| 4126 closed | 0.450 | 0.451 | 0.450 | **no** | 0.450 / 0.451 |
| 4127 open | 0.544 | 0.516 | 0.499 | yes | 0.575 / 0.530 |
| 4127 closed | 0.470 | 0.446 | 0.427 | yes | 0.470 / 0.446 |
| 4128 open | 0.545 | 0.501 | 0.505 | yes | 0.576 / 0.508 |
| 4128 closed | 0.386 | 0.380 | 0.416 | yes | 0.386 / 0.380 |

`gpu` leans toward thor more than `observed` in 5 of 6 cells, but not in every
seed. The lean is clear in the open regime (+0.03 to +0.04) and negligible in
the closed one.

### `observed`'s τ̂ at dispatch, seen vehicles

| cell | orin median [IQR] | thor median [IQR] | P(same τ̂), random orin × thor pair | orin × thor dispatched in the same round with the same τ̂ |
|---|---|---|---|---|
| 4126 open | 1 [0, 1] | 0 [0, 1] | 0.376 | 8 / 18 |
| 4126 closed | 1 [0, 1] | 0 [0, 1] | 0.424 | 35 / 67 |
| 4127 open | 1 [0, 2] | 0 [0, 1] | 0.354 | 13 / 28 |
| 4127 closed | 0 [0, 1] | 0 [0, 1] | 0.436 | 42 / 79 |
| 4128 open | 1 [0, 1] | 0 [0, 1] | 0.367 | 6 / 18 |
| 4128 closed | 0 [0, 1] | 0 [0, 1] | 0.418 | 47 / 85 |

Realised staleness is an integer round count that sits mostly at 0–1 for
both fast tiers. A random orin and a random thor carry the same τ̂ 35–44% of
the time, so `observed` separates them only weakly. That part of the
hypothesis is right. The candidate set is not logged, so "same round" means
pairs **dispatched** in the same round, a lower bound on co-candidates.

### `gpu − observed` through the conservation law (log-points ≈ % of T)

`T = R × mean_sleep / (C × (1 − waste))`. The mix channel is split by a Shapley
average over the three factors, so the parts sum exactly.

| cell | measured | xavier share | orin / thor split | within-tier | waste | residual |
|---|---|---|---|---|---|---|
| 4126 open | −1.93 | **−1.77** | −0.16 | −0.28 | +0.27 | −0.00 |
| 4126 closed | −2.73 | **−1.87** | +0.00 | +0.11 | −0.99 | −0.01 |
| 4127 open | −0.24 | **−0.57** | −0.12 | +0.31 | +0.07 | +0.06 |
| 4127 closed | −1.69 | **−1.77** | −0.07 | +0.44 | −0.23 | −0.07 |
| 4128 open | −1.16 | **−2.04** | −0.18 | −0.18 | +1.30 | −0.07 |
| 4128 closed | −2.49 | **−1.93** | −0.02 | −0.03 | −0.61 | +0.07 |

The residual is ≤ 0.07 points: the conservation law closes in every cell.

### Verdict: **refuted**

The orin/thor split contributes −0.18 to +0.00 points in every cell, a
negligible part of the gap. **The gap comes from xavier share**, −0.57 to
−2.04 points, and it is the largest term in every one of the six cells. The
waste channel varies in sign and partly offsets it in the open regime
(+0.07 to +1.30). `gpu` does lean toward thor slightly more (5 / 6 cells), but
that is worth almost nothing, because orin and thor differ by only ~0.9 s of
sleep while xavier differs by ~13 s.

So `gpu`'s consistent edge over `observed` is that it dispatches **slightly
less xavier**: 0.26–0.99 points of dispatch share less than `observed`, depending on cell. Signal precision
among the fast tiers is not the mechanism.

---

## 1. Step 1: the two knobs

| knob | where | default (= 7b / 8) | Stage 9 |
|---|---|---|---|
| `g_prior_alpha` | **`[server] polaris_alpha`**, a key that already existed and is read only for the unexplored proxy (`fedasync_polaris_vi_server.py:372`, used at `:295`) | 10.0 | **1.0** |
| `observed_unseen_tau` | `[server] observed_unseen_tau` (new) | `const_0.01` | **`fleet_running_mean`** |

**Note on "mean of measured `G_i`".** The code's proxy is α × the mean norm of
the **latest aggregation batch**, not a mean over all identities' `G_i`. With
`minimum_clients_aggregated = 1`, that batch is usually one report. α = 10 is
defined this way, and Stage 9 changes only α, as specified. So under α = 1 a
never-seen vehicle gets exactly the batch mean norm (the "basis"). The
aggregation strategy now records `last_proxy_basis` so this can be checked;
nothing reads it.

**`fleet_running_mean`**
(`sumo_open_server.OpenPopulationMixin.install_unseen_tau_rule`):
- It applies to the reactive arm only.
- A vehicle is never-seen while its identity has no **delivered** report,
  which is the same predicate that updates its observed τ.
- Its τ̂ is the mean of every delivered staleness so far, across all vehicles,
  evaluated **at scoring time**. It is C − 1 = 3 before the first report.
- The running sum is fed from `OpenSlotSelectionGate.on_reports_received`,
  from delivered reports only.
- `gpu` / `cpu` keep their spec τ̂ for every vehicle, and `baseline` reads
  neither knob.
- An unknown value raises.

Dispatch lines gain `G_proxy_basis=` and `tau_unseen=` (appended; the existing
parsers are unaffected).

**Tests:** `tests/test_stage9_priors.py`, 8 new, 53/53 in total. They use the
real `PolarisViClientSelectionStrategy` and the real
`FedAsyncPolarisAggregationStrategy`:
- default and explicit `const_0.01` leave the observed predictor as the
  original function object;
- `fleet_running_mean` gives 3.0 → the running mean for unseen vehicles and
  the observed τ for seen ones;
- proactive arms ignore the rule;
- an unknown value raises;
- the default α is 10; the α = 10 proxy is 10 × the batch mean; the α = 1
  proxy is the batch mean.

The shared closed-loop tests (`test_predictor_strictness`,
`test_staleness_predictor`) pass.

**Configs:**
- `configs/stage9_s<seed>/`: gpu / cpu / observed, plus `baseline` at 4126
  for invariance;
- `configs/stage9_s4126_preflight/`;
- `configs/stage8_s<seed>/cpu_{open,closed}.toml` for the α = 10 cpu
  partners.

Against 7b / 8, the Stage 9 configs differ only in run dir, port,
`polaris_alpha = 1.0`, `observed_unseen_tau` and `physics_seed = 4126`. The
baseline config carries the knobs too, which is what the invariance run tests.

While generating configs, a port-numbering change briefly **overwrote the 7b
config files' ports**. They were restored from a backup taken before the
change and verified byte-identical, and the generator now reproduces 7b
byte-identically. No run was affected.

---

## 2. Step 2: pre-flight (seed 4126, 100 rounds, all 8 arm × regime)

`preflight_stage9.py` → `stage9_s4126_preflight.json`. It runs every Stage 7b
check unchanged. The one exception is 7b's "τ̂ == τ_obs (observed)", which is
split into a seen and a never-seen row. **PRE-FLIGHT PASSED, every check 100%:**

| Stage 9 check | gpu_o | cpu_o | obs_o | gpu_c | cpu_c | obs_c |
|---|---|---|---|---|---|---|
| never-seen `G_i` == proxy == 1.0 × basis | 42/42 | 49/49 | 42/42 | — | — | — |
| observed never-seen: τ̂ == logged running mean | — | — | 46/46 | — | — | 23/23 |
| logged running mean == running mean recomputed from the log (delivered stalenesses, start 3) | — | — | 115/115 | — | — | 115/115 |
| observed seen: τ̂ == its own τ_obs | — | — | 69/69 | — | — | 92/92 |
| logged V_i == G_i / √(1 + τ̂) of the dispatch line | 117/117 | 116/116 | 115/115 | 112/112 | 117/117 | 115/115 |

Every Stage 7b check (identity, tier, predictor tier, sleep, partition,
seed, resets, one identity per slot, startup `T_round`, `gpu` τ̂ = Stage 3–6
values) is also 100% in all eight, `baseline` included.

---

## 3. Run ledger

Driver: `sumo_open_loop/run_stage9.sh` (log `logs/run_stage9.log`). It is a
**four-slot queue**: a new run starts as soon as one finishes, so the slowest
run of a wave never leaves slots idle. Each run's checkpoints are deleted when
it finishes.

The **invariance run is first**. When it finishes, `stage9_invariance.py`
compares it with 7b `baseline_open` (dispatch sequence, recycle sequence,
T @ 4,000). On any difference the driver kills the running jobs and stops
(stop condition). The 18 neutral-prior runs start alongside it rather than
after it, to keep four slots busy. If invariance fails, they are stopped and
discarded.

| # | run | config |
|---|---|---|
| 1 | invariance: `stage9_s4126_baseline_open` | `configs/stage9_s4126/` |
| 2–7 | `stage9_s4126_{gpu,cpu,observed}_{open,closed}` | α = 1 |
| 8–13 | `stage9_s4127_…` | α = 1 |
| 14–19 | `stage9_s4128_…` | α = 1 |
| 20–23 | `stage8_s4127_cpu_{open,closed}`, `stage8_s4128_cpu_{open,closed}` | α = 10, Stage 8 code and seed split |

The first launch at 17:25:57 started nothing: the driver aborted on `set -u`,
because bash treats an empty associative array as unbound. It was fixed
(`set -o pipefail` only) and relaunched at 17:26:09. The aborted log is kept as
`logs/run_stage9.aborted_setu.log`.

---

## 4. STOP: the baseline invariance run did not reproduce Stage 7b

### What happened

The invariance run (`stage9_s4126_baseline_open`) finished at 22:57 UTC.
`stage9_invariance.py` → `baseline_invariance.json`:

| | Stage 7b | Stage 9 |
|---|---|---|
| dispatches | 4,620 | 4,583 |
| dispatch sequence | identical for the first **25** dispatches, then diverges | |
| T @ 4,000 (reconstructed) | 7,238.0 s | 7,144.9 s (**−1.29%**) |

The driver stopped the queue, as the stop condition requires.

### Cause: host wall-clock time inside PLATO's simulated clock, not the knobs

The two runs differ from the **very first report**. Same vehicle, same
`sleep`, but `finish_t` = 1.74 vs 1.80 s. PLATO adds each client's
`processing_time` to the simulated finish time (`plato/servers/base.py:1028`),
and that `processing_time` is measured with the host clock
(`time.perf_counter() - tic`, `plato/clients/strategies/defaults.py:252`).

Measured as `finish − dispatch − sleep` over every report, it is on average
0.11 s per job (5–95%: 0.085–0.17 s; max 0.8–4.7 s), and it differs from one
execution to the next. It is the same in magnitude in every run from 7b and 8.
Whenever that jitter swaps the order of two events, which here happened at
dispatch 25, the runs diverge from then on.

This is not something the knobs cause:

- `baseline` reads neither knob, and the tests show the defaults leave the
  code paths untouched.
- The sequences are identical until the first timing swap.
- The mechanism predicts divergence for **any** two executions of the same
  config.

Consequences:

1. **No run in this programme is bit-reproducible.** Exact dispatch-sequence
   invariance, as Step 2 asks for it, cannot be met by any re-run, including
   a re-run of 7b's own config.
2. The analysis clock (`sim_time_reconstructed`, sleeps only) excludes the host
   time. But selection happens on PLATO's clock, which includes it, so the
   host jitter propagates into the dispatch decisions.
3. **The spread from execution alone is about 1.3% on T @ 4,000** (one pair,
   `baseline_open`, seed 4126). That is the same size as `baseline`'s seed SD
   (1.08% open) and as the C2 gaps (`gpu − observed`: 0.24–2.73%). Part of
   what has been reported as seed noise is execution noise, and C2-sized
   differences are within it.

### A second problem found while checking: two runs survived the stop without cleaners

The driver killed the job **wrappers**. The python servers of `gpu_closed` and
`cpu_closed` kept training for about 8 h (to rounds 1,530 / 1,487). Their
cleaners had been stopped with the wrappers, so they accumulated **67 GB and
65 GB of checkpoints**. Both were killed on 2026-09-26 around 06:47 UTC, and
all their checkpoints were deleted.

The driver now starts every job in its own process group (`setsid`) and stops
jobs with `kill -- -pgid`. This was tested: the pid equals the pgid, and a
group kill terminates the job.

### State of the runs

| run | state |
|---|---|
| `stage9_s4126_baseline_open` | complete (invariance run) |
| `stage9_s4126_gpu_open` | complete, `.done` |
| `stage9_s4126_observed_open` | complete, `.done` |
| `stage9_s4126_cpu_open` | *reached round 4,000 and wrote its summary and full `results/`; killed during shutdown, so no `.done`. Its data are complete |
| `stage9_s4126_gpu_closed`, `cpu_closed` | killed mid-run (rounds 1,530 / 1,487): **invalid**, to be re-run |
| the other 17 | not started |

### Decision needed

1. **Accept invariance in distribution, and reuse the 7b / 8 baselines**
   (recommended). The knobs provably do not reach `baseline` (tests plus
   identical sequences up to the first host-timing swap). Record the ~1.3%
   execution spread and carry it into every comparison. C2 in particular has
   to clear it. Resume the queue as it is.
2. **Also run fresh Stage 9 baselines for all three seeds** (6 extra runs), so
   every Stage 9 comparison is between runs of the same code. This does not
   remove the execution noise; it only avoids reusing older runs.
3. **Make the simulation deterministic** by zeroing `processing_time` on the
   open server before PLATO uses it. Runs would then be exactly reproducible,
   but the simulated clock would change (about −0.11 s per job), so every
   α = 10 partner cell from 7b / 8 would have to be re-run too for paired
   comparisons (18 or more runs), on top of Stage 9.

---

## 5. Decision and resumption (2026-09-26)

**User decision: option 1.** Invariance is accepted **in distribution**; the
7b / 8 `baseline` cells are reused. The ~1.3% same-config execution spread
is recorded and carried into every comparison.

The decision is on file as `logs/stage9_invariance.accepted`. The driver
reads it instead of re-running the exact comparison.

Housekeeping before resuming:
- `stage9_s4126_cpu_open` was marked `.done` by hand, with a note in the
  marker. It had reached round 4,000, written its summary and complete
  `results/`, and was killed only during shutdown.
- The killed `stage9_s4126_gpu_closed` / `cpu_closed`: logs were moved to
  `logs/superseded_stage9/`, run directories removed, and both re-queued from
  scratch.
- The stopped driver log is kept as
  `logs/superseded_stage9/run_stage9.stopped_2026-09-25.log`.

Resumed at 06:53:40 UTC with the fixed driver (`setsid` per job, group kill).
The queue skipped the four complete runs and started `s4126` gpu_closed,
cpu_closed, observed_closed and `s4127` gpu_open. On check, all four were
dispatching with no tracebacks, the correct seeds (run 4126 / 4127, physics
4126) and α = 1.0, one cleaner each, and every python server's pgid equal to
the job id the driver holds. 19 runs remain.

---

## 6. PRELIMINARY results, 2026-09-26 ~17:00 UTC (8 of 23 runs done)

`sumo_open_loop/analyze_stage9.py` → `stage9_analysis.json`. The script works
on whatever runs exist and will be re-run in full at the end.

**Available α = 1 runs:** seed 4126, all six cells; seed 4127, `gpu_open`
only. Every comparison below is **one seed** except where a second is shown.
Same-config execution noise is ~1.3% (§4). No verdict here is final.

Soundness of the 7 α = 1 runs:
- conservation error ≤ 0.04%;
- plateau 82.13 – 83.07%, all inside the band, so dropping newcomers is not
  damaging the model;
- predictor tier agreement 100% (`gpu`, `cpu`).

### T @ 4,000, seed 4126

| cell | α = 1 | α = 10 | α = 1 vs α = 10 | waste α = 1 / α = 10 | xavier share α = 1 / α = 10 |
|---|---|---|---|---|---|
| gpu open | **5,768.3** | 6,768.6 | **−14.78%** (4127: −15.06%) | 10.89 / 8.94% | **26.2** / 34.0% |
| observed open | 6,678.8 | 6,901.5 | −3.23% | 11.93 / 8.69% | 32.0 / 34.8% |
| cpu open | 7,638.7 | 7,723.3 | −1.10% | 13.99 / 10.49% | 37.0 / 39.5% |
| baseline open | — | 7,238.0 | | 13.38% | 34.7% |
| gpu closed | 7,610.2 | 7,659.3 | −0.64% | 13.45 / 13.41% | 39.6 / 39.9% |
| observed closed | 7,799.0 | 7,874.3 | −0.96% | 13.51 / 14.27% | 41.0 / 40.9% |
| cpu closed | 9,490.5 | 9,445.7 | +0.47% | 16.40 / 15.73% | 49.9 / 50.1% |
| baseline closed | — | 7,260.4 | | 13.56% | 37.1% |

### Membership and fairness, open regime, seed 4126

| arm | ever dispatched, xavier / orin / thor | first-dispatch mix, x / o / t | median wait to 1st dispatch, x / o / t (s) | never dispatched | Jain |
|---|---|---|---|---|---|
| **gpu α = 1** | **85.3** / 98.1 / 98.1% | **44.7** / 37.6 / 17.7 | **11.9** / 7.4 / 7.5 | 8.1% | 0.644 |
| observed α = 1 | 91.3 / 90.7 / 93.9% | 48.5 / 34.3 / 17.2 | 10.0 / 10.5 / 11.2 | 8.5% | 0.637 |
| cpu α = 1 | 90.8 / 90.3 / 93.9% | 48.3 / 34.2 / 17.5 | 11.5 / 10.3 / 9.8 | 8.8% | 0.665 |
| baseline | 91.6 / 92.1 / 91.0% | 48.7 / 35.0 / 16.3 | 11.2 / 10.1 / 10.1 | 8.3% | 0.628 |
| gpu α = 10 | 99.5 / 99.8 / 100.0% | 48.5 / 34.7 / 16.9 | 4.5 / 2.9 / 2.5 | 0.3% | 0.695 |
| observed α = 10 | 99.8 / 99.8 / 100.0% | 48.8 / 34.5 / 16.7 | 2.8 / 2.9 / 2.6 | 0.2% | 0.702 |

What this shows (seed 4126):
- Under α = 1, every arm leaves about 8–9% of admitted vehicles undispatched,
  **the same as `baseline`**. Under α = 10 the figure is 0.2–0.4%.
- Only `gpu` makes that loss **tier-selective**: 14.7% of xavier are never
  dispatched, against 1.9% of orin / thor. `observed` and `cpu` lose all tiers
  about equally, like `baseline`.
- `gpu`'s first dispatches are less xavier-heavy (44.7% vs ~48.5%), and it
  makes xavier wait longest.

### `G_i`

- Never-seen `G_i` at dispatch is ~0.85 under α = 1 (the batch-mean proxy),
  against ~8 under α = 10.
- Seen xavier : thor, rounds > 2,000:

| | open | closed |
|---|---|---|
| gpu α = 1 | 1.81 | 2.86 |
| gpu α = 10 | 1.37 | 3.06 |
| observed α = 1 | 1.52 | 2.83 |
| observed α = 10 | 1.55 | 2.90 |

- `observed`'s running-mean τ̂ for never-seen vehicles stays at 2.27–2.32
  (open) and 2.06–2.21 (closed) throughout.

### Comparisons (seed 4126 unless noted)

| # | result |
|---|---|
| A: gpu vs observed, α = 1, open | **−13.63%** (the α = 10 value was −1.93%). Far beyond the 1.3% execution noise |
| B: gpu vs observed, α = 1, closed | −2.42% (α = 10: −2.73%) |
| C: α = 1 vs α = 10 | open: gpu −14.78% (4127: −15.06%), observed −3.23%, cpu −1.10%. closed: gpu −0.64%, observed −0.96%, cpu +0.47% |
| D: vs baseline, α = 1 | open: gpu −20.31% (4127: −21.63%), observed −7.73%. closed: gpu +4.82%, observed +7.42%. **The C3 sign flip persists** |
| E: cpu vs baseline, α = 1 | open +5.54%, closed +30.72%. cpu is the worst arm |

### Predictions, provisional (mostly one seed)

| # | status so far |
|---|---|
| 1 membership moves, most for xavier under gpu | **consistent**: 99.5 → 85.3% for xavier under gpu. All arms drop to ~91% |
| 2 C2 appears in open, gpu beats observed by more than 1.93% | **consistent, strongly**: −13.6% (one seed). Mechanism as predicted: fewer xavier first dispatches and fewer xavier overall |
| 3 closed almost unaffected | **consistent**: α = 1 vs α = 10 within ±1% in closed |
| 4 cpu worst | **consistent** in both regimes |
| 5 fairness cost | **present, and concentrated on xavier under gpu** (14.7% never dispatched); plateau intact. The overall never-dispatched share (8%) is no worse than `baseline`'s |
| 6 C3 flip persists | **consistent** (seed 4126) |

This is provisional. Seeds 4127 and 4128 are needed for every sign check,
and the α = 10 cpu partners for 4127 / 4128 are still queued.

---

## 7. Interim results: all 18 α = 1 runs (2026-09-27 ~06:30 UTC)

Still missing: three α = 10 `cpu` partners (`stage8_s4127_cpu_closed`,
`stage8_s4128_cpu_{open,closed}`). They affect only comparison C for `cpu`.

`analyze_stage9.py` fix: T @ 4,000 is now taken with `clock_at` (the nearest
round ≤ 4,000), as in every other analysis. `stage9_s4128_gpu_closed` logs its
last merge as round 3,999, which crashed the first version.

**Soundness (18 α = 1 runs):**
- conservation error ≤ 0.12%;
- predictor tier agreement 100%;
- plateau 81.77 – 83.18%. `stage9_s4128_cpu_closed` is at 81.77%, **0.33
  points below** the 82.1% band edge. The stop condition is "more than 2
  points below", so this is noted, not a stop.

### T @ 4,000, α = 1

| cell | 4126 | 4127 | 4128 |
|---|---|---|---|
| gpu open | 5,768.3 | 5,682.7 | 5,693.2 |
| observed open | 6,678.8 | 6,571.4 | 6,937.2 |
| cpu open | 7,638.7 | 7,459.4 | 7,697.4 |
| gpu closed | 7,610.2 | 7,998.6 | 7,409.7 |
| observed closed | 7,799.0 | 8,115.1 | 7,394.6 |
| cpu closed | 9,490.5 | 9,902.0 | 9,044.0 |

### Comparisons, per seed (4126 / 4127 / 4128)

| # | comparison | per seed | verdict |
|---|---|---|---|
| **A** | gpu vs observed, α = 1, open | **−13.63 / −13.52 / −17.93%** | **held in every seed; every seed beyond the 1.29% execution noise** |
| B | gpu vs observed, α = 1, closed | −2.42 / −1.44 / +0.20% | **undecided** (4128 flips) |
| C | α = 1 vs α = 10, gpu open | −14.78 / −15.06 / −15.30% | held in every seed |
| C | gpu closed | −0.64 / −0.29 / +4.24% | mixed, inside the closed-regime seed spread (Stage 8 SD ≈ 6%) |
| C | observed open | −3.23 / −2.01 / +2.01% | undecided |
| C | observed closed | −0.96 / −0.55 / +1.43% | undecided, small |
| C | cpu open | −1.10 / −2.46 / (pending) | |
| C | cpu closed | +0.47 / (pending) / (pending) | |
| **D** | gpu vs baseline, open | **−20.31 / −21.63 / −19.93%** | held |
| D | observed vs baseline, open | −7.73 / −9.38 / −2.44% | held |
| **D** | gpu vs baseline, closed | **+4.82 / +8.98 / +1.99%** | held: behind in closed |
| D | observed vs baseline, closed | +7.42 / +10.57 / +1.78% | held: behind in closed |
| E | cpu vs baseline, open / closed | +5.54 / +2.87 / +8.26%; +30.72 / +34.92 / +24.49% | cpu is worst, every seed |

### Membership, open regime, α = 1

| seed | arm | ever dispatched, x / o / t | never dispatched | Jain | first-dispatch xavier share |
|---|---|---|---|---|---|
| 4126 | gpu | **85.3** / 98.1 / 98.1% | 8.1% | 0.644 | 44.7% |
| 4126 | observed | 91.3 / 90.7 / 93.9% | 8.5% | 0.637 | 48.5% |
| 4126 | baseline (α = 10 run) | 91.6 / 92.1 / 91.0% | 8.3% | | |
| 4127 | gpu | **87.7** / 97.9 / 98.1% | 7.0% | 0.637 | 45.3% |
| 4127 | observed | 91.8 / 91.0 / 90.9% | 8.6% | 0.635 | 48.9% |
| 4127 | baseline | 91.4 / 91.8 / 92.5% | 8.3% | | |
| 4128 | gpu | **87.2** / 97.7 / 97.4% | 7.4% | 0.633 | 45.2% |
| 4128 | observed | 90.8 / 89.1 / 86.4% | 10.5% | 0.626 | 49.6% |
| 4128 | baseline | 90.5 / 92.3 / 90.0% | 8.9% | | |

Overall xavier dispatch share, gpu / observed: 26.2 / 32.0, 25.6 / 31.6,
26.1 / 33.3%. Seen `G_i` xavier : thor for gpu, open / closed:
1.81 / 2.86, 1.54 / 2.81, 1.76 / 2.74.

### Predictions, interim (all three seeds; only the cpu α = 10 partners pending)

| # | status |
|---|---|
| 1 membership moves, most for xavier under gpu | **held**: gpu leaves 12–15% of xavier undispatched vs ~2% of orin / thor. Every arm ends at ~7–10% overall, which is baseline's level |
| 2 C2 appears: gpu beats observed in open, beyond −1.93% | **held in every seed**, −13.5 to −17.9%. The mechanism matches the prediction: gpu's first dispatches are 45% xavier vs 48.5–49.6%, and 26% of its dispatches overall vs 32–33% |
| 3 closed almost unaffected | mostly: gpu closed −0.6 / −0.3 / +4.2%, observed closed −1.0 / −0.6 / +1.4%, all within the closed regime's known seed spread |
| 4 cpu worst | **held** in every seed and both regimes |
| 5 fairness cost | **present, tier-selective**: under gpu, 12–15% of xavier are never dispatched. The overall never-dispatched share and Jain's index are at baseline's level. Plateau is intact apart from one run 0.33 points under the band edge (cpu closed, not gpu) |
| 6 C3 flip persists under α = 1 | **held in every seed**: gpu and observed are ahead of baseline in open and behind it in closed |

---

## 8. Final (2026-09-27 ~11:00 UTC): all 22 runs, 42 analysed with partners

The queue finished at 10:47 UTC (`STAGE9_ALL_DONE`) with no failures. Every
run's checkpoints are deleted, and no `checkpoints/` directory remains under
`runs/sumo_open_loop/`.

**Occupancy invariant:** one digest across **all 42 runs** (18 α = 1, 18
α = 10 partners, 6 baselines) over the common prefix of 1,616 recycles.

A first version of `analyze_stage9.py` hashed a fixed 1,800-recycle cut and
reported 4 digests. That was an analysis bug, not a stop: α = 1 `gpu_open`
runs end at t ≈ 5,700 s with only ~1,620 recycles, so their cut was shorter by
construction. The script now uses the true common prefix.

**cpu α = 10 partners (4127 / 4128):**
- T @ 4,000 open 7,647.6 / 7,844.1 s, closed 9,957.7 / 8,852.4 s;
- conservation error ≤ 0.09%, plateau 81.91 – 83.62%, predictor agreement
  100%.

This completes comparison C for `cpu`:

| C: α = 1 vs α = 10 | 4126 | 4127 | 4128 | verdict |
|---|---|---|---|---|
| cpu open | −1.10% | −2.46% | −1.87% | held (small) |
| cpu closed | +0.47% | −0.56% | +2.16% | undecided, ≈ 0 |

All other numbers in §7 stand unchanged: A, B, D, E, membership, and the
prediction table.

### Summary

**Prediction 2, the point of the stage, holds in every seed.** With a neutral
prior, newcomers have to compete. `gpu` then beats `observed` in the open
regime by **13.5–17.9%** (α = 10: 0.2–1.9%). It is **15% faster than itself
under α = 10** in every seed, and **~20% faster than random selection**. Each
seed's gap is 10× or more the execution noise.

**The mechanism is spec-ranked admission.** `gpu` lets 12–15% of xavier go
undispatched, against ~2% of the faster tiers. Its overall xavier dispatch share
falls to ~26% (observed 32–33%).

**Prediction 5, the price.**
- The loss is **tier-selective**: xavier vehicles bear it.
- In aggregate it is not a fairness regression against random selection. The
  never-dispatched share is 7–10% for every α = 1 arm and for `baseline`, and
  Jain's index is 0.63–0.67 for all of them. What α = 10 bought was universal
  admission (0.2–0.4% never dispatched).
- The model does not suffer: plateau is in the band apart from one `cpu`
  closed run at 0.33 points under the edge.

**Closed regime (Prediction 3):** the prior barely matters, as expected, since
only a slot's first use is never-seen. C3's sign flip against `baseline`
persists in every seed (Prediction 6), and `cpu` stays worst (Prediction 4).

**C2 is therefore not a null.** The earlier null (Stages 7b / 8) came from the
optimistic prior: it admitted every newcomer regardless of spec, which left
the proactive signal nothing to decide. Under a neutral prior the spec decides
who trains, and that is worth ~14–18% of wall time in the open population, at
the cost of excluding a minority of slow-tier vehicles.
