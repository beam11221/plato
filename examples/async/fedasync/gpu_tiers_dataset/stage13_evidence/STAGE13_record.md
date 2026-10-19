# Stage 13 (CIFAR-100 pilot): implementation record

Spec: `prompts/STAGE13_cifar100_pilot.md`. Nothing under `plato/` was touched.
The results report (`STAGE13_pilot_report.md`) is written automatically when
the six runs end.

## Configs: `sumo_open_loop/make_stage13_configs.py`

The configs are copied from the Stage 10 open configs (p_recur = 0, the
p_return = 0 end of Stage 11). Only these fields change, and each is asserted
to change exactly once:

| setting | change |
|---|---|
| `dataset_name` | CIFAR10 → **CIFAR100** |
| `[parameters.model] num_classes` | **100** (output layer) |
| `target_accuracy` | 0.9 → **0.99** |
| `base_path` | new run folders |
| `port` | 19701–19706 |

**Runs:**

| # | prefix | arm |
|---|---|---|
| #1 | `stage13nf_s4126` | baseline / none |
| #2–#5 | `stage13f_s4126` | baseline / gpu / observed / cpu + F |
| #6 | `stage13rep_s4126` | baseline + F (replicate of #2) |

**What can end a run early:** only `rounds = 4000` and `target_accuracy` (now
unreachable). The trace (36,000 s) outlasts every run; the longest T@4000 so
far is 9,958 s.

**Model.** The spec says "ResNet-18 (GroupNorm)". Stages 3–11 actually used
Plato's `resnet_18`, which uses **BatchNorm**. It was kept unchanged, as the
spec also requires, apart from the output layer.

**Data.** CIFAR-100 was downloaded to `fedasync/data/cifar-100-python`
(50,000 train / 10,000 test). The Dirichlet sampler takes the class count
from the datasource, so it covers all 100 labels.

## Tests: `tests/test_stage13_cifar100.py` (8 new; 89/89 in total)

- Each config differs from its source only in the allowed lines.
- Ports and run folders are unique and do not clash with any earlier config.
- In a fresh process, the real datasource, sampler and model build:
  - 100 classes;
  - 2,500 images per partition, with labels beyond class 9;
  - output shape (·, 100);
  - one finite loss and backward pass.

## Smoke run (CPU, 20 rounds, gpu+F, 2026-10-02)

- It ran on CPU so it would not take a GPU slot. It finished cleanly.
- The Stage 0 filter was active: pool 14 → 11 in the first rounds, fallback
  0%, predictor tier agreement 100%.
- Accuracy was 6.2% at round 10 and 9.4% at round 20 (chance is 1%).
- `stage13_runcheck.py` and `stage13_report.py` were both run on it.
- Its checkpoints were deleted.

## Queue: `sumo_open_loop/run_stage13.sh` (log `logs/run_stage13.log`)

- **Start:** launched 2026-10-02 17:25 UTC. It waits until the Stage 11
  driver has started all 34 Phase 2 runs (or has exited). It then fills free
  slots, counting **every** training process on the machine against the limit
  of four.
- **Order:** #2, #3, #1, #4, #5, #6.
- **After each run:** `stage13_runcheck.py` checks the soundness items only
  (reached round 4,000, conservation ≤ 3%, fallback ≤ 10%, median pool ≥ 4,
  predictor tier). The plateau rule is disabled. A failure is logged and the
  queue continues.
- **Housekeeping:** each run has its own process group and cleaner, and its
  checkpoints are deleted at the end.
- **At the end:** `stage13_report.py` writes `STAGE13_pilot_report.md` and
  `series_<#>_*.csv`.

## Confirmed with the user (2026-10-02)

- **Model:** BatchNorm ResNet-18, as in Stages 3–11.
- **Configs:** taken from the Stage 10 open configs.
- **G_i prior `α = 1`:** `polaris_alpha = 1.0` is set in all three scored
  arms (gpu, observed, cpu). Also present: `polaris_initial_gradient_bound =
  0.5`, `observed_unseen_tau = "fleet_running_mean"`. The baseline arms read
  none of these.
- **Clients:** 20 slots, C = 4.

**Partitions, measured with the real sampler from the Stage 13 config.**
Each vehicle trains on partition `blake2b(veh_id) mod 20`, one of these 20.

| | value |
|---|---|
| images per partition | 2,500, no duplicates within a partition |
| average per class | 25 |
| classes present | 91–97 of 100 (mean 94) |
| classes with ≥ 10 images | 60–70 (mean 65) |
| largest class | 100–178 images (mean 136) |
| share of the 10 largest classes | 33–40% |

Partitions are drawn independently, so they can overlap; 20 × 2,500 is not a
split of the 50,000 images.
