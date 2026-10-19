# Project status — SUMO closed-loop FedAsync

Written 2026-09-07, after all eight runs finished. This is the "where things
stand" file: what was built, what the runs found, what is still open. Start here
when picking the project up cold, then read `README.md` for the design and
`HANDOFF.md` for how to run it.

---

## 1. Where this sits in the project

| step | what it adds | state |
|---|---|---|
| `../surgical/` | four selection arms, clients stand still | done earlier |
| **this step** | the clients are **vehicles that move**; an RSU can only reach them inside its range | **done** |
| next step | **location prediction** — predict where a vehicle will be when it finishes, and filter on that before scoring | not started |

The four arms are unchanged from the surgical experiment and must stay that way
(`fedasync-four-arm-benchmark`): `baseline` (random), `gpu` (correct spec),
`cpu` (deliberately wrong spec), `observed` (Polaris's reactive estimator).

**The selection formula was deliberately not touched**: still
`V_i = G_i / sqrt(1 + tau_hat_i)`, client speed and dataset quality only.
Mobility restricts *who may be asked*, never *who looks best*. Adding a mobility
term is the next step's job.

---

## 2. What was built

| file | what it does |
|---|---|
| `remap_fleet.py` | maps SUMO vehicles onto client IDs, tier-preserving; drops the warm-up |
| `sumo_mobility.py` | `CoverageOracle` — is client *i* in RSU range at time *t*? |
| `sumo_gates.py` | the two places coverage changes FedAsync: selection, and upload |
| `sumo_server.py` | the waiting clock, the upload verdict, the round rule, waste counters |
| `sumo_fedasync.py` | entry point for all four arms |
| `make_sumo_configs.py` | renders the four configs and asserts they differ in ~4 keys |
| `preflight.py` | checks the inputs agree; prints the numbers the run must reproduce |
| `run_sumo.sh` | runs arms in parallel, one checkpoint cleaner each, `.done` markers |
| `health_check.py` | one health pass; exits non-zero on trouble. Drove the monitoring loop |
| `analyze.py` | per-cell report vs the no-mobility control → `RESULTS_<cell>.md` |
| `compare_experiments.py` | the cross-experiment 2x2, with debiased MAE |
| `parse_waste_log.py` | server log → per-report CSV + waste tally |
| `make_handoff_bundle.sh` | packs the untracked files for a machine not on this mount |
| `tests/` | 93 tests, no GPU needed, ~10 s |

Nothing in PLATO's core was modified. Everything is wrappers, a mixin and
subclasses, per the repo's strategy pattern.

---

## 3. What the runs found

Eight runs, all complete, 10 000 rounds each, zero tracebacks, zero deadlocks,
no trace wrap: `{n20, n100} x {baseline, gpu, cpu, observed}`.
Reports: `RESULTS_n20.md`, `RESULTS_n100.md`.

### 3.1 Waste computing — the headline, and the cleanest result

| arm | n20 total | n100 total | xavier n20 | xavier n100 |
|---|---|---|---|---|
| baseline | 17.2 % | 19.7 % | 40.1 % | 37.6 % |
| gpu | 18.2 % | 20.0 % | 39.6 % | 38.9 % |
| observed | 18.4 % | 20.0 % | 39.9 % | 37.9 % |
| cpu | **21.2 %** | **24.0 %** | 39.1 % | 37.9 % |

**The waste rate per tier is the same in all four arms.** No selector avoids it,
because none of them knows where the vehicles are. What differs between arms is
only how often each *asks* the slow tier: `cpu` sends the most work to xavier
(46.8 % of dispatches at n20 against baseline's 34.4 %) and therefore wastes the
most.

The measured rates match what the coverage trace predicts on its own
(43.5 / 5.6 / 2.9 % for n20), which is the strongest evidence the mobility
pipeline is wired correctly.

**This is the target for the next step.** Location prediction has ~38-40 % of
xavier's dispatches to reclaim, and every arm has the same room to improve.

### 3.2 Mobility erodes the GPU-spec predictor's advantage

MAE of `tau_hat` in rounds, `gpu` vs `observed`, across the 2x2. Regenerate with
`python3 compare_experiments.py`.

| cell | condition | raw gap (observed − gpu) | verdict | debiased gap |
|---|---|---|---|---|
| n20 | still | +0.254 ± 0.023 | 11.3 SE, gpu better | +0.338 |
| n20 | moving | +0.044 ± 0.024 | 1.8 SE, **tie** | +0.130 |
| n100 | still | +0.149 ± 0.022 | 6.9 SE, gpu better | +0.256 |
| n100 | moving | −0.088 ± 0.024 | 3.7 SE, **observed better** | +0.044 |

Read the two right-hand columns together, because they say different things:

* **Raw**: with the clients standing still the GPU spec beats the reactive
  estimator decisively. Under mobility the advantage shrinks to nothing at n20
  and *reverses* at n100.
* **Debiased** (each arm's constant offset removed first): **`gpu` is better in
  all four conditions**. The reversal is an artifact.

The artifact is the bias column: `observed` measures tau directly so its bias is
~0 everywhere (+0.01 to +0.03), while `gpu`'s deepens as waste grows
(−0.52 → −0.58 → −0.68 → **−1.00**). `gpu` predicts `T_i / T_round` from a speed
model blind to mobility; under mobility a round happens only when an upload
succeeds, so rounds arrive `(1 − waste)` times slower than reports while the
prediction stays put.

**The defensible claim** is the debiased one: mobility genuinely narrows the gap
(0.338 → 0.130 at n20; 0.256 → 0.044 at n100) but does not reverse it. Anyone
reporting only the raw numbers would conclude that a reactive estimator beats a
correct hardware spec under mobility, and that conclusion is not supported.

### 3.3 Mobility changes who gets asked — but by adding work, not moving it

Dispatch counts, n20, still → moving:

| arm | total dispatches | xavier share | xavier count Δ |
|---|---|---|---|
| baseline | 10 003 → 12 080 | 43.5 % → 34.4 % | **−197** |
| gpu | 10 003 → 12 235 | 42.6 % → 37.9 % | +381 |
| cpu | 10 003 → 12 689 | 53.7 % → 46.8 % | +561 |
| observed | 10 003 → 12 263 | 41.8 % → 38.1 % | +489 |

Every wasted upload frees a slot that is refilled, so mobility *adds* ~2 000-2 700
dispatch cycles. The fast tiers absorb most of them, which is why xavier's share
falls while its absolute count rises in three of the four arms. `baseline` is the
exception — it has no speed signal holding xavier in the mix.

### 3.4 On accuracy and time, the arms are not separable

Round-budget gaps sit inside ±1.5 accuracy points; time-to-accuracy ratios run
1.05-1.80× but the project's measured noise floor for accuracy-crossing times is
±10-15 % (`../gpu_tiers/HANDOFF.md` § 4.13). The mechanism is visible in §3.3:
`gpu` and `observed` differ by 0.2 points of dispatch share, so they pick nearly
the same vehicles and the clock cannot tell them apart. The surgical experiment
reached the same null at n20 for the same reason.

---

## 4. Decisions taken along the way

| decision | where it is recorded |
|---|---|
| `anchor_sleep = 15`, and anchored seconds ARE physical seconds | **ADR-0004** |
| `eta_nominal = 1.0` is a placeholder; implied efficiency is 1.83 % of peak | ADR-0004 |
| A round exists only when an aggregation happens | `README.md` § "What a round is" |
| Refill the pipeline by the deficit, counting both withheld groups | `README.md` landmines |
| The idle-clock guard must ask about the *selectable* set | `HANDOFF.md` § 7 |

Two bugs were found and fixed mid-project, both pinned by tests:

1. **Pipeline inflation** — counting only `training_clients` as busy re-dispatched
   vehicles that were still training in simulated time, running the pipeline at 4
   dispatches per merge instead of 1. Caught by the first smoke run.
2. **Deadlock at round 719** — the idle-clock guard asked "is any vehicle in
   range" when it had to ask "is any *selectable* vehicle in range". Caught by
   the monitoring loop; cost one restart. The fix fired 34-40 times per arm in
   the n20 runs, so without it every arm would have died.

---

## 5. Open questions, in priority order

1. **Multi-seed.** Every number above is a single seed (4126). Nothing here
   should go into a write-up without at least three seeds on whichever cell is
   reported.
2. **How to report MAE.** §3.2 shows the raw and debiased numbers disagree about
   who wins. A write-up needs to state both and explain the bias, or it will
   mislead. This is the single most important presentational decision.
3. **The anchor is the least-evidenced number in the setup.** Waste rates are a
   direct function of "a Jetson Xavier trains 2500 CIFAR-10 images in 15 s". If a
   real Jetson benchmark exists, adopt it — ADR-0004 § 3 has the migration notes
   and lists what has to be re-derived.
4. **The exploration-bonus feedback loop.** A vehicle whose uploads keep failing
   is never marked explored, so it keeps its 10× `G_i` proxy and keeps being
   picked. Honest (the server learned nothing) but it did not exist in the
   surgical experiment. Not yet measured; worth checking before the next step
   builds on it.
5. **The §3.3 mechanism is a hypothesis.** "Mobility gives the fast tiers more
   dispatches, so the reactive estimator sees fresher data" fits the numbers but
   was not tested directly. Measuring re-dispatch interval per tier would settle it.

---

## 6. Operational notes

* Interpreter: the bundled `.venv` is broken — use `/usr/bin/python3.11`
  (torch 2.4.1+cu124). `HANDOFF.md` § 2.2 lists the packages that had to be added.
* Two machines share one 200 GB mount. This box ran `cpu gpu`, the other ran
  `baseline observed`. **Do not run `build_inputs.sh` on the second machine while
  a run is live** — it would rewrite live inputs.
* Disk: n20 holds ~4 GB per arm, n100 ~12 GB (8 GB of that is the per-client
  payload files, which the cleaner deliberately leaves alone — see `README.md`).
  Clear finished runs' `checkpoints/` only; `results/`, `*.log` and `models/` are
  the analysis data.
* Throughput: two arms sharing one GPU ran at ~13 rounds/minute, so 10 000 rounds
  took 12-13 h per arm.
* `health_check.py` is the watchdog; the failure it exists to catch is a log that
  stops growing while the process is alive.
