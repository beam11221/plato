# Stage 13 — CIFAR-100 pilot. One seed, open population, 5 methods + 1 replicate.

Switch the dataset to CIFAR-100 and run. **No exploration, no tuning, no extra
diagnostics**: run the five runs below and report the standard metrics.

Do not touch anything under `plato/`.

---

## Configuration

**Identical to Stage 11 / Stage 10 open population in every setting**, except the dataset:

| | value |
|---|---|
| dataset | **CIFAR-100** (50,000 train / 10,000 test, 100 classes) |
| model | ResNet-18 (GroupNorm), unchanged except the output layer → 100 classes |
| population | open, `p_recur = 0` (no `p_return`) |
| prior | `g_prior_alpha = 1`, `observed_unseen_tau = fleet_running_mean` |
| filter (where on) | Stage 0, `stage0_r_i = vmax_distance`; `tier_spec` `T̂` for baseline / gpu, `observed_per_vehicle` for observed |
| everything else | **unchanged from Stage 11**: trace `e1_open_poisson_seed4126_range400`, physics seed 4126, `M = 20`, `C = 4`, partitioning `blake2b(veh_id) mod 20` with 2,500 images per partition and Dirichlet α = 0.8 (now over 100 labels), FedAsync, optimiser, learning rate, local epochs, batch size, evaluation every 10 rounds on the full test set, `coverage_gated`, startup `T_round` = 2.0808 s, vehicle sleep times |
| rounds | **4,000, every run** |
| target accuracy | set to **0.99** (unreachable) so no run stops early. Check that no other setting can end a run before round 4,000 |

Do not change any other value. If a setting has to change for CIFAR-100 to run at all
(e.g. a dataset path or class count), change only that and list it in the report.

## The runs — run seed 4126

The five methods compared throughout Stages 9–11, plus one replicate:

| # | arm | filter |
|---|---|---|
| 1 | baseline / none | off |
| 2 | baseline + F | on |
| 3 | gpu + F | on |
| 4 | observed + F | on |
| 5 | cpu + F (`tier_spec` from the CPU spec) | on |
| 6 | baseline + F, **replicate** of #2 (same config, same seed). Run last | on |

At most four at once. Start #2 and #3 first. If Stage 11 Phase 2 is still running, queue
behind it within the four-run limit.

**Stop conditions:** disable the plateau stop condition (plateau < 80.1%) entirely for
this stage. Every run must reach round 4,000 regardless of accuracy. Keep only the
soundness checks: conservation error ≤ 3%, fallback ≤ 10%, median pool after Stage 0
≥ 4, predictor tier == occupant tier. If one of those fails, report it and continue the
remaining runs. Do not kill the queue.

## Report — `STAGE13_pilot_report.md`

Per run, the standard Stage 10 set:

- final accuracy (plateau) and accuracy at rounds 1,000 / 2,000 / 3,000 / 4,000
- **TTA** (E0 method: 11-evaluation rolling median, first evaluation ≥ target) at
  targets 30 / 40 / 50 / 60 / 70 / 80%, "not reached" where it is not reached, with
  rounds-to-target next to each
- **waste**, and dispatch share per tier
- T @ 4,000 (backup)
- the accuracy-vs-round and accuracy-vs-sim-time series as CSV, all six runs

Then one table of the pairs #3 vs #2, #4 vs #2, #5 vs #3, #2 vs #1 and the replicate #6 vs
#2, at each target reached by both runs. Label it **one seed, not a result**.

## Do not

- tune learning rate, epochs, batch size, model or any other setting
- change sleep times, tiers, trace, `M`, `C`, the filter, the score or α
- stop any run before 4,000 rounds
- add runs, seeds or analyses beyond the above
