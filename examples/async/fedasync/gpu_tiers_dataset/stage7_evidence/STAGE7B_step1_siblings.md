# Stage 7b, Step 1: siblings of H3

Date: 2026-09-23. Spec: `prompts/STAGE7B_rerun.md`. Report only. **Nothing has
been changed except the actions the prompt ordered:** `cpu_open` and
`cpu_closed` were killed (together with their launcher and cleaners), and all
eight Stage 7 `checkpoints/` directories were deleted (~34 GB). Logs,
`results/` and `models/` (47 MB per run) are kept.

Every claim below is checked against the run logs, not only the code. Unless
a row says otherwise, the log evidence comes from `gpu_open`; the
partition, prior and leak checks were repeated on `baseline_open` and
`observed_open`.

## Headline

1. **Data partitions are slot-bound.** In every dispatch where the two can be
   told apart, the sampler was built from the slot id, never from the
   vehicle's partition. `baseline_open` and `observed_open` therefore deviate
   from spec. **Decision needed.**
2. **A new sibling that is not about `client_id` keying but is just as silent:
   the `G_i` prior is never restored on recycle.** At `p_recur = 0`, a
   never-seen vehicle carries `G_i = 0.5` instead of the unexplored proxy
   (`α × mean ≈ 10×`). That is **below** the median measured `G_i` (0.90), so
   new vehicles were ranked *down*, not up. This contradicts the premise of
   7b Prediction 1 and affects `observed_open` and `gpu_open`.
3. The per-dispatch torch seed is slot-keyed: `derive_seed(4126, slot, round)`
   on 4684 of 4684 dispatches.

## 1. Data partitions: **slot-bound** (evidence from logs)

Design intent: `VehicleShardLifecycle.load_data` swaps `client_id` for
`partition + 1`. PLATO's per-dispatch order is
`process_server_response → load_data → configure → allocate_data`
(`plato/clients/composable.py:310-319`). `load_data` only loads the
*datasource* (the whole dataset), and only when none is cached yet. The
**sampler**, which is what chooses the partition, is built in `configure`
(`plato/clients/strategies/defaults.py:121`). By then the swap has already
been undone, so the sampler gets the **slot** id.

Log evidence. `plato/samplers/registry.py:56` logs
`[Client #N] Sampler: noniid`, where `N` is the id the sampler was built
with. Pairing each one with the dispatch's own
`[Stage7] client slot=S ... -> partition=P` line gives:

| run | sampler built with slot id | with partition id (P+1) | ambiguous (S = P+1) |
|---|---|---|---|
| `baseline_open` | **4401** | **0** | 193 |
| `observed_open` | **4357** | **0** | 174 |
| `gpu_open` | **4481** | **0** | 203 |
| `gpu_closed` | **4454** | **0** | 205 |

The ambiguous ~4% is the chance rate of `S = P+1` (1/20). The log also shows
`Loading its data source` only **4 times** per run (once per worker), against
`Sampler:` **4684** times. The swap has no effect on which data is used.

At `p_recur = 0` every vehicle in slot *k* trained on partition *k−1*,
regardless of `hash(veh_id)`. At `p_recur = 1` under the new 7b semantics,
slot-bound is *correct*.

## 2. Torch seeding

- **Process level: correct.** `seeding.seed_everything()` runs at module level
  in `sumo_open_fedasync.py` and seeds `random`, numpy, torch and CUDA, with
  `cudnn.deterministic = True`. Log:
  `[Seed] ... seed=4126 ... numpy=True torch=True cudnn_deterministic=True`.
- **Per dispatch: slot-keyed.** `ClientSeededTrainer` reseeds from
  `derive_seed(base_seed, client_id, round)`. Recomputing it from the log gives
  a match with `client_id = slot` on **4684 / 4684** dispatches, with 20
  distinct ids against 1730 distinct vehicles. The spec wants *identity*.

## 3. Full inventory: state and lookups on the Stage 7 path

"Should follow" uses the 7b abstraction:
**occupant identity** = trace vehicle at `p_recur = 0`, fixed slot identity
at `p_recur = 1`.
"Dispatch identity" = the occupant identity frozen at dispatch.

| # | item | where | should follow | currently | realised in logs? |
|---|---|---|---|---|---|
| 1 | predictor tier, candidate scoring | `gpu_tier_sleep.tier_for(cid)` ← closed-loop CSV | occupant | slot (CSV) | **yes**: 62.8% wrong (H3) |
| 2 | predictor busy-pool terms (live `T_round`, residuals) | `fedasync_polaris_vi_selection.py:222,231` | dispatch identity | slot (CSV) | **yes** (H3) |
| 3 | startup `T_round`, `pool_mean_T` | `compute_pool_T_round_seconds` over CSV | actual initial occupants | CSV `n=10/7/3` | **yes** (H3) |
| 4 | **data partition (sampler)** | `defaults.py:121` in `configure` | occupant | **slot** | **yes, 100%**: §1 |
| 5 | **per-dispatch training seed** | `client_seeded_trainer.derive_seed` | dispatch identity | **slot** | **yes, 4684/4684**: §2 |
| 6 | **`G_i` unexplored prior on recycle** | `stage7_population.reset_slot_state` | occupant (proxy restored) | **fixed 0.5, proxy never restored** | **yes**: §4 |
| 7 | observed τ update (`local_stalenesses[idx]`) | `on_reports_received`, **gated** by `OpenSlotSelectionGate` (`delivered()`) | dispatch identity | slot | **correct** (corrected 2026-09-23, see §5) |
| 8 | observation count `_observations[slot] += 1` | `sumo_open_server.py:308`, on every report | dispatch identity | slot | **latent, 0 occurrences** (§5) |
| 9 | sleep (physics) | `sumo_open_server.py:295`, `vehicle_sleep_s(veh_id, trace tier)` | dispatch identity | trace vehicle | correct at `p=0`; **wrong at `p=1` under the new semantics** (design change, not a bug in the old runs) |
| 10 | `_dispatch_vehicle`, `_dispatch_tier` | `record_dispatches`, `timeline.*_in_slot` | occupant | trace vehicle | same as 9 |
| 11 | client-side `[GpuTierSleep] ... tier_id= sleep_time=` log | `GpuTierLogCallback`, CSV by slot | dispatch identity | slot | 4684/4684 match the CSV row, so it **mislabels ~63%**. Log only, since the value is overwritten server-side, but misleading in any audit |
| 12 | `staleness_predictions[cid]` + accuracy log | selection strategy | dispatch identity | slot, popped on reset | correct; the log needs `identity_id` + tier (audit requirement 4) |
| 13 | `tau_hat=` field in the dispatch log | `record_dispatches` reads `local_stalenesses` | value the selector used | observed-τ array | **yes** (audit check 3) |
| 14 | `G_i` update (`local_gradient_bounds[idx]`) | inside `aggregate_weights`, **gated** by `UploadGate` | dispatch identity | slot | **correct**: only delivered reports update it, and a delivered report means the vehicle was still in coverage, so it still holds its slot |
| 15 | `aggregation_weights` | `fedasync_polaris_vi_server.py:204` | — | slot | **inert**: initialised, never read |
| 16 | PLATO `training_clients`, `reported_clients`, comm times | `plato/servers/base.py` | slot (transport) | slot | correct: per dispatch, or uniform payload size |
| 17 | `partition_sizes` | server state | — | slot | harmless: uniform 2,500 |
| 18 | occupancy, `_selectable_slots` | `sumo_open_server` | slot, by definition | slot | correct |
| 19 | `UPLOADED_KEY[reporting_id]` | set in `process_client_info` | dispatch | slot | correct: at most one dispatch in flight per slot |

## 4. The `G_i` prior is never restored (row 6)

`reset_slot_state` resets `local_gradient_bounds[idx] = 0.5` and then tries
`server.unexplored_clients.add(idx)`, so that the next aggregation overwrites
0.5 with the unexplored proxy `α × mean` (α = 10). But `unexplored_clients`
is an attribute of the **aggregation strategy**
(`fedasync_polaris_vi_server.py:187`), which `sumo_open_server.py:128` wraps
in `UploadGate`. The server object has no such attribute, so the
`getattr(server, "unexplored_clients", None)` guard returns `None` and the
re-add is **silently skipped**. The unit test passed because it supplies a
fake server that does carry the attribute.

Log evidence: `G_i` on zero-observation dispatches.

| run | zero-obs dispatches | `G_i = 0.500000` | after round 500 |
|---|---|---|---|
| `observed_open` | 1686 | **1670** | **1467 / 1467** |
| `gpu_open` | 1730 | **1714** | **1537 / 1537** |

The 16 exceptions are all in the first rounds, before the first recycle,
when the initial unexplored set still applies (values 24–38).

Scale: the median measured `G_i` of already-observed vehicles after round 500
is **0.90** (`observed_open`; p10 0.35, p90 1.89). The intended proxy is
≈ 10 × mean ≈ 9. What new vehicles actually got was 0.5, which is **below
the median**. The intended design gives never-seen vehicles a strong
*exploration* bonus. The runs gave them a *penalty*.

Consequences:

- **7b Prediction 1's premise** ("every never-seen candidate carries the same
  10× `G_i` prior") does not hold in the current code.
- **Stage 7's `observed_open` result** (6.30% ahead of `baseline`) was produced
  with new vehicles pushed down the ranking, so the arm leaned on
  re-dispatching vehicles it already knew. That result is confounded and
  should not be quoted as it stands.
- `baseline_*` is unaffected (it has no `G_i`). At `p_recur = 1` there are no
  resets, so row 6 does not arise there.

## 5. Latent: the observation count (row 8). Row 7 was a false alarm

> **Correction (2026-09-23, during Step 0).** An earlier version of this
> section said the observed-τ update was ungated. That was wrong. `UploadGate`
> gates only `aggregate_weights`, but the selection side has its own gate:
> `OpenSlotSelectionGate.on_reports_received` passes only `delivered(updates)`
> to the Polaris strategy. A delivered report means the vehicle was still in
> coverage, so it still held its slot, and the τ update lands on the right
> occupant. Only the observation count (row 8) was ungated.

`UploadGate` gates only `aggregate_weights`. The observed-τ update
(`on_reports_received`) and `_observations[slot] += 1` run on **every** report,
including one from a vehicle that left coverage. If the slot had already been
recycled to a new vehicle, that vehicle would inherit a measurement it never
produced.

Checked in the logs. At `p_recur = 0`, a vehicle's **first-ever** dispatch
must show `observations = 0` and `tau_hat = 0.01`, and on every dispatch the
logged `observations` must equal that vehicle's own prior reports:

| run | first dispatches with inherited obs / τ | logged obs ≠ own reports |
|---|---|---|
| `observed_open` | 0 / 1686, 0 / 1686 | 0 / 4531 |
| `gpu_open` | 0 / 1730, 0 / 1730 | 0 / 4684 |
| `baseline_open` | 0 / 1842, 0 / 1842 | 0 / 4594 |

**This never happened in these runs.** It is still a code path with no
guard, so the fix should key both on the dispatch identity, and the
pre-flight should keep this check.

## 6. Cleared

- **Tier mix under the new `p_recur = 1` semantics.** The fleet is 10/7/3 =
  50/35/15%. The trace's windows are 49.1 / 34.9 / 16.0%. Binding identity to
  the slot does not change the tier composition between the two regimes.
- `G_i` update (row 14), PLATO transport state (row 16), occupancy (row 18).

## Decisions that come back to you

**A. Partitions (§1).** `baseline_open` and `observed_open` used slot-bound
partitions (and slot-keyed seeds, §2), so both deviate from spec. By the 7b
prompt, they cannot be reused unless you decide otherwise.

**B. The `G_i` prior (§4).** This is independent of A. `observed_open` is
affected regardless of the partition decision, and `gpu_open` is being re-run
anyway. Restoring the proxy restores the specified reset. It is not a change
to `V_i` or to the selection rule. It does change what "never seen" looks
like to the selector, which is the premise of 7b Prediction 1.

**Recommendation:** re-run **all four open arms**, so Wave A becomes 4 runs
instead of 2. `observed_open` is affected by B in any case, and
`baseline_open` differs from spec by both partition and seed. Reusing it
would pair a spec-deviating baseline with three corrected arms.

Everything else (rows 1–3, 5, 7–13) goes into the single occupant-identity
abstraction of Step 0, together with the pre-flight checks of Step 2. Those
checks should add: the `G_i` of a never-seen occupant equals the unexplored
proxy, and sampler id = occupant partition + 1.

---

## Decision (2026-09-23, user)

**Re-run everything.** All four open arms (`baseline`, `gpu`, `cpu`,
`observed` at `p_recur = 0`) and all four closed arms at `p_recur = 1` under
the 7b semantics. **No Stage 7 run is reused.**

Why the open `baseline` and `observed` are not reused either:

- `gpu` / `cpu` scored vehicles on another vehicle's TOPS (H3).
- `observed` gave never-seen vehicles `G_i = 0.5` instead of the 10× proxy
  (§4), which penalises exploration instead of rewarding it.
- `baseline` is the least affected, but its partition and seed are slot-bound
  (§1–2), so it is not the specified baseline. If it were paired with three
  corrected arms, the differences could not be attributed to the selector.

The goal is a **valid** result, not a favourable one. The fixes make the code
do what the Stage 7 / 7b specs already say. None of them touches the GPU
equation, `V_i`, the selection rule, sigma, the tier ratio, `M`, the trace,
the dataset, the model or the optimiser, and nothing is tuned toward a
prediction.

Stage 7 results that survive without a re-run: the soundness checks
(conservation, plateau, occupancy invariant), κ ≈ 2.5–2.7 ≪ k\*, and the
information-state measurement (how little `observed` knows at `p_recur = 0`).

Plan: Step 0 (one occupant-identity abstraction + tests) → Step 2 pre-flight
(all checks at 100%) → Wave A (4 open runs) → report → Wave B (4 closed runs).
Each run has its own checkpoint cleaner.
