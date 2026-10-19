# Stage 7b: occupant identity, pre-flight, re-runs

Spec: `prompts/STAGE7B_rerun.md`. Step 1 findings:
`STAGE7B_step1_siblings.md`. Decision (user, 2026-09-23): **re-run all
eight**; no Stage 7 run is reused.

Status is kept current at the top; sections below are appended as each step
lands.

---

## Status

| step | state |
|---|---|
| Step 1: sibling hunt | done, reported |
| Step 0: occupant identity abstraction | **done**; 45/45 Stage 7/7b tests pass |
| Step 2: pre-flight smoke runs (8 × 100 rounds) | **PASSED: every check 100% in all 8 runs** (§3) |
| Startup `T_round` | **decided (user): keep fleet-file value** 2.0808 s (gpu) / 3.6202 s (cpu) in both regimes. §4 |
| Wave A (4 open runs) | **done** 2026-09-23 23:41–23:58 UTC, 0 failures. Reported in §5 |
| Wave B (4 closed runs) | **done** 2026-09-24 09:40–09:46 UTC, 0 failures. Reported in §6 (full 8-run results) |

Housekeeping done: `cpu_open` / `cpu_closed` from Stage 7 killed, and all
eight Stage 7 `checkpoints/` deleted (~34 GB). Every 7b run starts its own
checkpoint cleaner (keep 50 rounds, every 3 min, 12 GB ceiling).

---

## 1. Step 0: one abstraction

New module `sumo_open_loop/occupant_identity.py`. Every slot has, at every
moment, an **occupant identity** `(identity_id, tier, sleep_s, partition)`:

| | `p_recur = 0` | `p_recur = 1` |
|---|---|---|
| identity_id | trace `veh_id` | slot number `k` |
| tier | trace tier | `clients_n20r400.csv` row `k` |
| sleep_s | `vehicle_sleep_s(4126, veh_id, tier)` | `client_sleep_times_n20r400.csv` row `k` |
| partition | `blake2b(veh_id) mod 20` | `k − 1` (sampler id `k`, as in Stages 3–6) |
| coverage window (upload?) | trace vehicle | trace vehicle (trace supplies timing only) |

`IdentityResolver.of_slot(k)` is the only way any Stage 7b code learns who is in
a slot. There is **one code path**. The two regimes differ only inside the
resolver, never at the call sites.

Everything from the Step 1 inventory that keyed on `client_id` now reads the
identity:

| inventory row | now reads | where |
|---|---|---|
| 1 predictor tier (candidates) | occupant identity at scoring time | `OpenPopulationMixin._occupant_predict_seconds` |
| 2 busy-pool terms | identity **frozen at dispatch** | `_dispatched_predict_seconds`, via the new `busy_predict_seconds` hook |
| 3 startup `T_round` | **fleet-file value kept** (decision §4); the initial-occupant value is logged for the record only | `_set_startup_t_round` |
| 4 data partition | identity partition; sampler rebuilt **after** `configure` | `OccupantLifecycle.configure` |
| 5 training seed | `identity_seed(base, identity, round)` | `OccupantSeededTrainer` |
| 6 `G_i` prior on recycle | the aggregation strategy's current proxy, and the slot re-added to *its* `unexplored_clients` | `stage7_population.reset_slot_state` / `unexplored_owner` |
| 8 observation count | per identity, **delivered** reports only | `process_client_info` |
| 9, 10 sleep, dispatch tier | dispatched identity | `process_client_info`, `record_dispatches` |
| 12, 13 audit fields | dispatch line carries `identity_id`, `pred_tier`, the `tau_hat` V_i **used**, `tau_obs`, `partition`, `sleep_s`, `G_proxy` | `record_dispatches` |

Rows that deliberately stay slot-keyed: PLATO transport state, occupancy,
`UPLOADED_KEY` (per dispatch), and the τ / `G_i` arrays. At `p_recur = 0` a
slot's array entry is reset whenever the identity changes. At `p_recur = 1`
slot = identity. So in both regimes the arrays are effectively
identity-indexed, and rewriting them keyed on identity would change nothing.

At `p_recur = 1`, `identity_seed(base, "k", r)` is **byte-identical** to the
closed-loop `derive_seed(base, k, r)` (tested). The sampler id is `k`, as in
Stages 3–6. So the closed control is seeded and partitioned exactly like the
closed-loop cell.

### How each fix avoids the Stage 7 failure

- **Partition:** the sampler is rebuilt after `configure`. It is never done by
  swapping `client_id`: the trainer and algorithm name their per-client model
  files by `client_id`, and two vehicles with the same partition can be in
  flight at once.
- **Prior:** `reset_slot_state` now **raises** if a Polaris server's
  `unexplored_clients` cannot be found. The Stage 7 defect was a silent skip.
- **Shared modules** (`fedasync_polaris_vi_selection.py`,
  `fedasync_polaris_vi_server.py`): the only additions are the
  `busy_predict_seconds` hook (default `None` → the historical call),
  `last_tau_hats` (read-only exposure) and `last_proxy` (recorded, not read).
  With the defaults, Stages 3–6 behave identically. `test_predictor_strictness.py`
  and `test_staleness_predictor.py` pass. `test_t_round.py` shows 2 failures,
  both inside `gpu_tier_sleep.py`, which this stage does not touch: a
  "sane range < 10 s" bound and a pre-anchoring expected value (0.055 s vs the
  anchored 20 s). They pre-date this work.

### Tests: 45 / 45

`tests/test_stage7_population.py` (31) and `tests/test_stage7b_identity.py`
(14, new).

The new file imports the **real** `FedAsyncPolarisAggregationStrategy` wrapped
in the **real** `UploadGate`, the real selector and the real predictor. Every
Stage 7 defect survived because it was tested against a fake that put things
where the real code does not. Stage 7's own reset test had placed
`unexplored_clients` on the server, so it has been corrected to the real
location, and a test now requires the old layout to raise.

Pinned:
- slot 1 is scored as its thor occupant, not the CSV's xavier (H3 regression);
- busy terms use the dispatched identity after the slot recycles;
- busy terms never touch the fleet file;
- the `p_recur = 1` identity is stable across different trace vehicles;
- the `p_recur = 1` fleet has per-tier sleep means 14.37 / 1.78 / 1.07;
- `identity_seed` reproduces `derive_seed` for slot identities;
- a new identity carries the current proxy.

---

## 2. Pre-existing behaviour found while wiring the predictor (not changed)

Stage 7 and Stage 6 `gpu` logs both show exactly **three** predicted values
for the entire run (Stage 6 s4126: 0.9242 / 7.2089 / 0.4806). The proactive
arms always take the fallback path `τ̂ = T_pred / T_round(startup)`. The live
busy-pool branch ("T_round recomputed each round from live busy pool", as the
log claims) never produces a value in practice, because `busy_residuals` is
empty at every selection.

That is how the `gpu` arm has behaved since Stage 3, and it is what 7b's
Prediction 1 numbers (0.822 / 0.721 / 0.349) are computed from. Changing it
would change the estimator, which 7b forbids, so it is **left as is**.

Two consequences:

- The busy-pool fix (row 2) is correct but dormant.
- The startup `T_round` sets the scale of **every** proactive prediction in the
  run. That is why §4 matters.

---

## 3. Step 2: pre-flight

Checker: `sumo_open_loop/preflight_stage7b.py`. It reads the smoke-run logs and
recomputes every expected value independently: from the trace `veh_id`, the
fleet files and the shared derivations. A check passes only if the log agrees
with the **specification**, not merely with itself.

Smoke runs: `configs/stage7b_preflight/*.toml`, 100 rounds, otherwise
identical to the Wave configs. Result (`stage7b_preflight.json`):
**PRE-FLIGHT PASSED: every applicable check is 100% in all eight runs.**

| check | base_o | gpu_o | cpu_o | obs_o | base_c | gpu_c | cpu_c | obs_c |
|---|---|---|---|---|---|---|---|---|
| identity == spec | 115/115 | 112/112 | 109/109 | 109/109 | 117/117 | 120/120 | 118/118 | 114/114 |
| tier == occupant tier | 115/115 | 112/112 | 109/109 | 109/109 | 117/117 | 120/120 | 118/118 | 114/114 |
| **predictor tier == occupant tier** | — | **112/112** | **109/109** | — | — | **120/120** | **118/118** | — |
| sleep == occupant sleep (dispatch) | 115/115 | 112/112 | 109/109 | 109/109 | 117/117 | 120/120 | 118/118 | 114/114 |
| sleep == occupant sleep (report) | 115/115 | 112/112 | 109/109 | 109/109 | 117/117 | 120/120 | 118/118 | 114/114 |
| partition == occupant (dispatch) | 115/115 | 112/112 | 109/109 | 109/109 | 117/117 | 120/120 | 118/118 | 114/114 |
| partition == occupant (client) | 115/115 | 112/112 | 109/109 | 109/109 | 117/117 | 120/120 | 118/118 | 114/114 |
| sampler_id == partition + 1 | 115/115 | 112/112 | 109/109 | 109/109 | 117/117 | 120/120 | 118/118 | 114/114 |
| one fingerprint per partition | 18/18 | 19/19 | 19/19 | 18/18 | 20/20 | 20/20 | 20/20 | 20/20 |
| one partition per fingerprint | 18/18 | 19/19 | 19/19 | 18/18 | 20/20 | 20/20 | 20/20 | 20/20 |
| seed == identity_seed(identity, round) | 115/115 | 112/112 | 109/109 | 109/109 | 117/117 | 120/120 | 118/118 | 114/114 |
| **tau_hat logged == tau_hat used** | — | **100/100** | **100/100** | **100/100** | — | **100/100** | **100/100** | **100/100** |
| tau_hat == tau_obs (observed arm) | — | — | — | 109/109 | — | — | — | 114/114 |
| observations == earlier delivered reports | — | 112/112 | 109/109 | 109/109 | — | 120/120 | 118/118 | 114/114 |
| **never-seen G_i == current proxy** | — | **52/52** | **54/54** | **46/46** | — | — | — | — |
| p=0: state_reset == recycles | 55/55 | 58/58 | 60/60 | 54/54 | — | — | — | — |
| p=1: slot keeps one identity/tier/partition (per slot) | — | — | — | — | 20/20 | 20/20 | 20/20 | 20/20 |
| p=1: resets only on a slot's first use | — | — | — | — | 20 | 20 | 20 | 20 |

"—" means the check does not apply to that arm or regime. `baseline` has no
predictor, no `tau_hat` and no `G_i`. At `p_recur = 1` there are no resets,
so no identity is ever "never seen" after its first dispatch. The
`tau_hat used` check covers every dispatch whose report was delivered (the
prediction-accuracy line is emitted only then).

What the fingerprint rows show: each partition produced exactly one label
distribution, and no label distribution was produced by two partitions. If
the sampler were still slot-bound, as in Stage 7, fingerprints would group by
slot, and at `p_recur = 0` every partition would show several. The `p=0`
open runs see 18–19 of the 20 partitions within 100 rounds, which is expected
for ~110 dispatches over `hash(veh_id) mod 20`.

The same smoke runs measured the startup `T_round` for the `cpu` arm:
fleet file **3.6202 s**, initial occupants 3.5769 s (`p_recur = 0`, −1.2%) and
3.7500 s (`p_recur = 1`, +3.6%). CPU tiers are far less heterogeneous, so §4's
problem is smaller for `cpu` than for `gpu`, but it has the same sign
structure.

Pre-flight checkpoints (36 GB across 8 × 100 rounds) were deleted after the
check. Logs and `results/` are kept.

---

## 4. Open decision: startup `T_round` (7b requirement 3)

7b adopted the audit's requirement 3: "startup pool median from actual initial
occupants". Implemented as specified, the smoke runs show:

| | startup `T_round` | initial occupants |
|---|---|---|
| Stages 3–6 (fleet file, all 20) | **2.0808 s** | 10 xavier / 7 orin / 3 thor |
| 7b `p_recur = 0` | **1.7423 s** (−16%) | 15 trace vehicles: 6 / 6 / 3 |
| 7b `p_recur = 1` | **2.6603 s** (+28%) | slots 1–15: 10 / 5 / 0 |

The `p_recur = 1` value is an **artefact of allocation order**. The allocator
hands out slots 1→20 FIFO, so the first 15 arrivals take slots 1–15, and slots
1–10 are the fleet's xavier. Over the run, round-robin reuse evens this out.
Measured over t ∈ [0, 7500) s, every slot is occupied 63–64% of the time, and
occupied slot-time by tier is:

| | xavier | orin | thor |
|---|---|---|---|
| `p_recur = 1` | 49.9% | 35.1% | 15.0% |
| `p_recur = 0` | 48.4% | 34.9% | 16.7% |

That is the nominal 50 / 35 / 15. Only the t = 0 snapshot is skewed.

Because §2 shows the startup `T_round` sets the scale of every proactive
prediction, this rule would:

- give `gpu` / `cpu` a **different τ̂ scale in each regime** (1.74 vs 2.66), a
  difference between `p_recur = 0` and `p_recur = 1` that has nothing to do
  with identity churn, which the closed control exists to isolate;
- move both regimes away from Stages 3–6 (2.08).

The effect on `s(τ̂)` is modest but real. The xavier : thor ratio of
`s(τ̂)` is 0.425 at 2.08, 0.405 at 1.74 and 0.456 at 2.66.

The queue-model `T_round` over the **long-run** occupied mix is 2.025 s at
`p_recur = 0` and 2.078 s at `p_recur = 1`, both within 2.7% of the fleet-file
value. This is the "or shown not to matter" branch of requirement 3, and it
favours the fleet-file value rather than the initial-occupant rule.

**Recommendation: keep the fleet-file startup `T_round` (2.0808 s) in both
regimes**, i.e. drop requirement 3. It is exact for `p_recur = 1`, within 2.7%
of the long-run pool at `p_recur = 0`, identical across regimes, and identical
to Stages 3–6. `baseline_open` and `observed_open` do not use `T_round`, so
they are unaffected by this decision and can start as soon as the pre-flight
passes.

### Decision (user, 2026-09-23 18:25 UTC): keep the fleet-file value

`_set_startup_t_round` no longer overrides anything. It logs the value in use
and, for the record, what the initial occupants would have given:

```
[Stage7b] startup T_round=2.080769 s (fleet file, kept; not overridden).
  For the record, 15 initial occupant(s) {'orin': 6, 'thor': 3, 'xavier': 6} would give 1.742308 s.
```

Requirement 3 is therefore **dropped**. The reason is the evidence above,
not convenience.

`baseline_open` and `observed_open` had already started. Neither arm reads
`T_round`, and the change touches only that method, so they are unaffected.

**Re-smoke of `gpu_open` and `cpu_open` with the final code** (100 rounds,
`stage7b_preflight_rerun_open_proactive.json`): every check 100%. That
includes a new, stronger check: **the `gpu` arm's `tau_hat` equals the
Stage 3–6 value for the occupant's tier on 110 / 110 dispatches**
(7.208872 / 0.924214 / 0.480591). With the fleet-file `T_round`, the `gpu`
arm is now exactly the Stage 3–6 estimator, fed the right vehicle. The
superseded smoke logs are in `logs/superseded_preflight/`.

`gpu_closed` / `cpu_closed` will be re-smoked with the final code before
Wave B. They are not re-smoked now, because six concurrent processes would
repeat the Stage 6 GPU-OOM failure.

---

## 5. Wave A: results (`p_recur = 0`, seed 4126, 4000 rounds)

Analysis: `analyze_stage7.py --stage 7b` → `stage7b_analysis.json`. Each run's
`checkpoints/` was deleted after it finished (~17 GB).

### Soundness: every stop condition clear

| check | threshold | measured |
|---|---|---|
| conservation-law error | < 3% | 0.011 – 0.049% |
| plateau | 82.1–83.7% | 82.69 – 83.62% |
| occupancy sequence identical across the 4 runs | stop if not | identical over the common 1,916-recycle prefix |
| state reset at `p_recur = 0` | `resets == recycles` | yes (all four) |
| predictor tier == occupant tier (`gpu`, `cpu`) | 100% | **100.0%** on all 4,396 / 4,472 dispatches (Stage 7: 37%) |
| measured κ | ≪ k\* (30–70) | 2.06 – 2.47 (median 2) |

### Cumulative time

| round | gpu | observed | baseline | cpu | gpu − obs | gpu − base | obs − base |
|---|---|---|---|---|---|---|---|
| 100 | 116.7 | 133.9 | 135.6 | 135.5 | −12.81% | −13.90% | −1.26% |
| 250 | 389.8 | 416.6 | 460.1 | 459.2 | −6.44% | −15.29% | −9.46% |
| 500 | 821.3 | 818.0 | 954.0 | 959.4 | +0.40% | −13.91% | −14.25% |
| 1000 | 1626.0 | 1626.0 | 1748.6 | 1852.2 | 0.00% | −7.01% | −7.01% |
| 2000 | 3398.3 | 3425.7 | 3658.5 | 3841.1 | −0.80% | −7.11% | −6.36% |
| **4000** | **6768.6** | **6901.5** | **7238.0** | **7723.3** | **−1.93%** | **−6.49%** | **−4.65%** |

Waste: `gpu` 8.94%, `observed` 8.69%, `baseline` 13.38%, `cpu` 10.49%.
Time to accuracy (h), for 70 / 75 / 78 / 80%:

| arm | 70% | 75% | 78% | 80% |
|---|---|---|---|---|
| gpu | 0.077 | 0.185 | 0.302 | 0.397 |
| observed | 0.103 | 0.174 | 0.271 | **0.358** |
| baseline | 0.098 | 0.180 | 0.317 | 0.520 |
| cpu | 0.099 | 0.179 | 0.305 | 0.470 |

The TTA ordering between `gpu` and `observed` is not the T@4000 ordering.
The accuracy curve has noise SD ≈ 0.046 per evaluation, which is large
relative to these gaps.

### Prediction scoring

| # | verdict | evidence |
|---|---|---|
| 1 | **NOT HELD as specified**; mechanism visible in a different form (below) | xavier share of zero-observation dispatches: `gpu` **48.5%**, `observed` **48.9%**, `baseline` 48.6%, `cpu` 48.3%. That is the trace population mix, in every arm |
| 2 | **HELD** | ≥ 1-observation dispatches: `gpu` 22.9 / 50.9 / 26.2, `observed` 23.6 / 52.4 / 24.0 (xavier / orin / thor, %). **But** `baseline` is 25.2 / 50.6 / 24.2, so most of the shift away from xavier is mechanical (xavier finishes fewer cycles before exit), not knowledge |
| 3 | **HELD in direction; not beyond seed noise** | `gpu_open` 6768.6 s vs `observed_open` 6901.5 s = **−1.93%**. Stage 6's `gpu` seed spread was 4.88%, so one seed cannot separate this from noise. The gap also crosses zero mid-run (+0.40% at 500, 0.00% at 1000) |
| 4 | **HELD** | `cpu_open` is slowest: 7723.3 s, +6.7% vs `baseline`. It dispatches the most xavier overall (39.5% vs 34–35%) |

### Why Prediction 1 cannot show up as specified: the first dispatch is saturated

The prediction assumed the selector chooses **which** never-seen vehicles get
dispatched. In this fleet almost every admitted vehicle gets its first
dispatch in every Polaris arm. The 10× `G_i` prior puts never-seen candidates
far ahead of everyone else, so they all get taken; only their **order** is a
choice. The measurement below is post-hoc: it was not named in the prompt,
and is reported alongside the specified metric, not in place of it.

Admitted vehicles entering in (0, 6768) s:

| arm | tier | ever dispatched | entry → 1st dispatch, mean / median (s) | 1st dispatch wasted |
|---|---|---|---|---|
| gpu | xavier | 99.4% | **7.41 / 4.49** | **3.4%** |
| gpu | orin | 99.8% | 4.79 / 2.85 | 0.0% |
| gpu | thor | 100.0% | 4.36 / 2.42 | 0.0% |
| observed | xavier | 99.8% | 5.19 / 2.75 | 1.0% |
| observed | orin | 99.8% | 4.99 / 2.81 | 0.0% |
| observed | thor | 100.0% | 5.04 / 2.47 | 0.0% |
| cpu | xavier / orin / thor | 99.5–99.7% | 6.39 / 6.18 / 6.31 (medians 4.07 / 4.00 / 3.71) | 1.7 / 0 / 0% |
| baseline | xavier / orin / thor | 91.1–92.2% | 13.72 / 13.48 / 13.65 | 10.1 / 0.7 / 0.4% |

This is the mechanism 7b describes, visible in **timing**. Among never-seen
candidates, `gpu` ranks by `s(τ̂)` and **defers xavier**: median wait 4.49 s
against 2.42–2.85 s for the faster tiers. `observed` gives every never-seen
vehicle τ̂ = 0.01 and shows **no tier preference** (2.47–2.81 s): in effect it
picks at random, exactly as the prompt said.

What the deferral buys is not what the prediction expected. Deferring a xavier
does not avoid dispatching it (99.4% still are). It dispatches it **later in
its coverage window**, and its first dispatch is then lost to mobility 3.4% of
the time against 1.0% under `observed`. The proactive signal changes when the
slow tier trains, not whether it does.

Where `gpu`'s −1.93% then comes from is not established by this analysis.
Overall `gpu` dispatches slightly less xavier (33.99% vs 34.81%), which through
the conservation law is worth about 1.8% of mean sleep, and slightly more
waste (+0.25 pt). Together that is roughly the measured gap. Which dispatches
carry the difference has not been isolated.

In the terms the prompt set out in advance, this is the reverse of the "1
holds, 3 does not" case: **the specified mechanism metric does not move,
because the channel it was meant to read is about order, not membership; the
outcome moves in the predicted direction but by less than one seed's noise.**

### Against Stage 7 (the invalid runs), for orientation only

| arm | Stage 7 | 7b | change | what changed |
|---|---|---|---|---|
| gpu | 7335.2 | 6768.6 | −7.7% | predictor now scores the right vehicle (37% → 100%) |
| observed | 6686.9 | 6901.5 | +3.2% | never-seen vehicles now get the 10× prior (Stage 7 gave them 0.5, pushing them down the ranking) |
| baseline | 7136.6 | 7238.0 | +1.4% | partition and seed now follow the vehicle |

Stage 7's headline "`gpu` 9.69% behind `observed`" came from the defect. With
the predictor fixed, the sign flips.

---

## 6. Wave B and the full picture: all eight runs

All eight runs completed 4,000 rounds with no failures. Each run's
`checkpoints/` was deleted after it finished (Wave B ~16 GB).
`stage7b_analysis.json` holds all eight.

### Soundness (all eight)

| check | measured |
|---|---|
| conservation-law error | −0.036% … +0.049% (limit 3%) |
| plateau | 82.29 – 83.61% (band 82.1–83.7%) |
| occupancy sequence identical across all 8 | **yes**, common prefix of 1,916 recycles |
| predictor tier == occupant tier (`gpu`, `cpu`, both regimes) | **100%** in all four runs |
| `p_recur = 1`: identity never lost | 0.5% zero-observation dispatches (each slot's first use); observations climb 7 → 160 per window |

### Headline table

| arm | regime | T@4000 (s) | waste | xavier / orin / thor dispatched | TTA 80% (h) |
|---|---|---|---|---|---|
| baseline | open | 7238.0 | 13.38% | 34.7 / 44.3 / 21.0 | 0.520 |
| **gpu** | **open** | **6768.6** | 8.94% | 34.0 / 43.9 / 22.1 | 0.397 |
| cpu | open | 7723.3 | 10.49% | 39.5 / 40.4 / 20.1 | 0.470 |
| observed | open | 6901.5 | 8.69% | 34.8 / 44.5 / 20.7 | **0.358** |
| **baseline** | **closed** | **7260.4** | 13.56% | 37.1 / 43.4 / 19.5 | 0.448 |
| gpu | closed | 7659.3 | 13.41% | 39.9 / 41.5 / 18.7 | 0.490 |
| cpu | closed | 9445.7 | 15.73% | 50.1 / 34.4 / 15.5 | 0.484 |
| observed | closed | 7874.3 | 14.27% | 40.9 / 40.8 / 18.4 | 0.449 |

### Relative cumulative time (gpu − obs / gpu − base / obs − base, %)

| round | open | closed |
|---|---|---|
| 100 | −12.81 / −13.90 / −1.26 | −5.49 / −24.19 / −19.79 |
| 250 | −6.44 / −15.29 / −9.46 | −1.93 / −19.65 / −18.07 |
| 500 | +0.40 / −13.91 / −14.25 | −2.97 / −18.78 / −16.29 |
| 1000 | 0.00 / −7.01 / −7.01 | −2.25 / −8.91 / −6.81 |
| 2000 | −0.80 / −7.11 / −6.36 | −2.01 / **+1.06** / **+3.14** |
| **4000** | **−1.93 / −6.49 / −4.65** | **−2.73 / +5.49 / +8.46** |

### Findings

**1. The `gpu` − `observed` gap is not produced by identity churn.** It is
−1.93% at `p_recur = 0` and −2.73% at `p_recur = 1`. It does not widen when
the reactive estimator is starved; if anything it is slightly smaller. Both
are within the single-seed noise measured in Stage 6 (`gpu` spread 4.88%). The
answer to the question this cell was built to ask is therefore **no**: in this
fleet, making the reactive estimator blind to 44% of its dispatches does not
open a proactive advantage beyond seed noise. This matches Wave A's mechanism
reading (§5). The proactive signal only changes the *order* of first
dispatches, and that channel is too narrow to move the outcome.

**2. Both Polaris arms do better in the open population than in the closed
one.** `gpu` is 13.2% faster open than closed; `observed` is 14.1% faster.
`baseline`, which carries no identity state, is unchanged (7238.0 vs 7260.4,
0.3%), so the trace and the fleet swap are not what moves the Polaris arms. The
cause is the persistent state.

**3. In the closed population, both Polaris arms lose to plain FedAsync after
round ≈2,000**, finishing +5.49% (`gpu`) and +8.46% (`observed`) behind
`baseline`. This is the closed-pool failure Stages 3–6 documented (the
cumulative-advantage crossing at round ≈2,400), reproduced here on the
open-trace timing. The mechanism is visible directly in `G_i`. Median `G_i`
at dispatch, rounds 2001–4000, for dispatches with ≥ 1 observation:

| run | xavier | orin | thor | xavier : thor |
|---|---|---|---|---|
| gpu_open | 1.09 | 0.72 | 0.80 | 1.4× |
| observed_open | 1.09 | 0.72 | 0.70 | 1.6× |
| gpu_closed | 1.32 | 0.47 | 0.43 | **3.1×** |
| observed_closed | 1.15 | 0.47 | 0.40 | **2.9×** |

In a closed pool, the slow tier's `G_i` stays high while the fast tiers' decays,
so `V_i ∝ G_i` steers dispatches toward xavier: 39.9% / 40.9% of dispatches,
against 37.1% for `baseline` and 34.0% / 34.8% open. The `s(τ̂)` discount
(`gpu`) and the observed staleness (`observed`) are not enough to offset it.
In the open population, resets and short identities cap the divergence, and
the arms stay ahead of `baseline` to round 4,000.

**4. `cpu` is the slowest arm in both regimes** (+6.7% open, +30.1% closed vs
`baseline`). It dispatches the most xavier in both (39.5% / 50.1%).

### Stage 7's original predictions, re-scored on the corrected runs

| # | prediction | verdict on 7b data |
|---|---|---|
| 1 | at `p_recur = 0`, `gpu` beats `observed` by more than the noise floor | **direction held, magnitude not** (−1.93%, within seed noise) |
| 2 | at `p_recur = 1`, `gpu` and `observed` indistinguishable | **held at noise level** (−2.73%, within 4.88%). The gap is not larger open than closed, so it is not a churn effect |
| 3 | at `p_recur = 0`, `observed` degrades toward `baseline` | **not held.** `observed_open` is 4.65% *ahead* of `baseline`, and 12.4% faster than `observed_closed`. The information-state starvation is real (44% zero-observation dispatches) and costs nothing |
| 4 | the closed-pool `s(τ̂)`/`G_i` divergence does not appear at `p_recur = 0`, so `gpu`'s advantage over `baseline` does not cross zero | **held.** Open: `gpu` stays ahead of `baseline` throughout (−6.49% at 4000). Closed: it crosses at ≈2,000. `G_i` xavier : thor 1.4× open vs 3.1× closed |
| 5 | `cpu` worst in both regimes | **held** |
| 6 | κ at `p_recur = 0` in the low single digits | **held** (2.06–2.47) |

### Caveats

- One seed (4126) per cell. The two gaps that matter (−1.93%, −2.73%) are
  inside Stage 6's measured single-seed spread. The open-vs-closed
  differences (13–14%) and the closed-pool crossing are far outside it.
- The TTA-80% ordering does not always match the T@4000 ordering (for
  example, `observed_open` reaches 80% before `gpu_open`). The accuracy noise
  per evaluation is SD ≈ 0.046.
- §5's first-dispatch timing analysis is post-hoc.
