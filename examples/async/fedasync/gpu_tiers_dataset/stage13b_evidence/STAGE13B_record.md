# Stage 13B: CIFAR-100 at 10,000 rounds (seed 4126, open population)

**Request (2026-10-04):** run the Stage 13 CIFAR-100 experiment again, the
same in every respect, with **10,000 rounds for every run**. Record it
separately as **Stage 13B**. It was first named "Stage 12B"; the user
corrected the name a few minutes after launch (see Log).

**Why:** at 4,000 rounds, Stage 13's accuracy flattened at about 45–48%, so
no run reached any TTA target above 40%. 13B shows whether longer training
raises the plateau and separates the arms.

Nothing under `plato/` is touched.

## Configuration

The same six runs as Stage 13: same arms, run seed 4126, physics seed 4126,
trace, filter, α = 1, BatchNorm ResNet-18 with 100 outputs, and 2,500 images
per partition.

| change vs Stage 13 | value |
|---|---|
| `[trainer] rounds` | 4,000 → **10,000** |
| run folders and prefixes | `stage13bnf_s4126`, `stage13bf_s4126`, `stage13brep_s4126` |
| ports | 19721–19726 |

| # | prefix / run | arm |
|---|---|---|
| 1 | `stage13bnf_s4126 / baseline_open` | baseline / none |
| 2 | `stage13bf_s4126 / baseline_open` | baseline + F |
| 3 | `stage13bf_s4126 / gpu_open` | gpu + F |
| 4 | `stage13bf_s4126 / observed_open` | observed + F |
| 5 | `stage13bf_s4126 / cpu_open` | cpu + F |
| 6 | `stage13brep_s4126 / baseline_open` | baseline + F, replicate of #2 |

**Generation:** `python3 sumo_open_loop/make_stage13_configs.py --stage stage13b --rounds 10000`.
- `tests/test_stage13_cifar100.py` pins that each 13B config equals its Stage 13
  config except header, `base_path`, `port` and `rounds`.
- 95/95 tests pass.
- Stage 13's configs re-generate byte-identically, and its report re-renders
  identically.

**Will 10,000 rounds fit in the trace?** The trace is 36,000 s. Scaling Stage
13's T@4000 by 2.5 gives about 17,850 s for the slowest run (baseline /
none) and about 13,400 s for cpu+F. Both are well inside it. The only things
that end a run are `rounds = 10000` and `target_accuracy = 0.99`, which is
unreachable.

## Queue

```
./run_stage13.sh stage13b 10000      # log: sumo_open_loop/logs/run_stage13b.log
```

- **Order:** #2 baseline+F, #3 gpu+F, #1 baseline / none, #4 observed+F, then
  #5 cpu+F and #6 the replicate, as slots free up.
- **Limit:** at most four runs on the machine.
- **Checks after each run:** soundness only (`stage13_runcheck.py <prefix>
  <run> 10000`). It checks that the run reached round 10,000, conservation
  error ≤ 3%, fallback ≤ 10%, median pool ≥ 4, and predictor tier ==
  occupant tier. There is no plateau rule. A failure is logged and the
  queue continues.
- **Housekeeping:** each run has its own process group and checkpoint cleaner;
  checkpoints are deleted when the run ends.
- **At the end:** `stage13_report.py --stage stage13b --rounds 10000` writes
  `STAGE13B_report.md` and `series_<#>_*.csv` in this folder.
  - It reports TTA at 30–80% (E0), with rounds-to-target.
  - Accuracy at rounds 1k / 2k / 3k / 4k / 6k / 8k / 10k.
  - Waste, tier share and T@10,000.
  - The pairs #3/#2, #4/#2, #5/#3, #2/#1 and #6/#2. These are one seed, not a
    result.

**Expected duration:** each Stage 13 run took about 5 h for 4,000 rounds, so
each 13B run should take about 12–13 h. Runs #1–#4 should end around
00:30–01:30 UTC on 5 Oct. #5 and #6 start then and end around 13:30–14:30
UTC on 5 Oct.

## Log

- **12:24 UTC:** launched under the name "Stage 12B".
- **12:28 UTC:** the user renamed it to Stage 13B.
  - The four runs had reached rounds 30–45, about 4 minutes in. They were
    stopped and their folders, logs and configs deleted, so that every
    artifact carries one name.
  - The scripts were renamed and the configs regenerated.
- **Found during the rename, and fixed:** the Stage 13 dataset test
  (`test_dataset_sampler_model_build_for_100_classes`) loaded the real
  `stage13f_s4126/gpu_open` config. Each test run therefore wrote a 3-line
  `server_<pid>.log` into that **finished run's folder**.
  - There were five such stubs, from 2 to 4 Oct. The analysis then picked the
    newest one and failed.
  - Training data was never affected.
  - Fixes:
    - the test now uses a copy of the config whose `base_path` points to a
      temporary folder;
    - `analyze_stage7.server_log()` only accepts a server log that has a
      matching `results/<pid>.csv`;
    - the five stubs were deleted.
  - Afterwards the Stage 13 report re-rendered identically.
- **12:30 UTC:** Stage 13B queue started. #2, #3, #1 and #4 are running, each
  with its cleaner.
