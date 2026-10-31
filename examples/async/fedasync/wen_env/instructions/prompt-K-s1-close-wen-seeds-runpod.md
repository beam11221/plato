# Prompt K — Close Scenario 1: `wen_fresh` and `wen_fresh_gpu`, seeds 4127 / 4128 (RunPod)

Continues from Prompt H (`wen_fresh`), Prompt I (Study 2, `wen_fresh_gpu`), Prompt J (Polaris, 3 seeds) and `s1_tta_energy_4arms.md`. Same rules: do not edit `plato/` core; keep the ground-truth / reported / estimated separation; locked values only; stop and report on any conflict.

## Why

Scenario 1 is complete for every method except the two Wen variants used in the thesis tables, which have seed 4126 only:
- `wen_fresh` = "Wen et al." in the main S1 comparison (decision S2-1);
- `wen_fresh_gpu` = Wen et al. + our GPU estimator (Study 2b).

The v3 rule needs the same direction in all 3 seeds, so P22–P24 (S1) and P29–P32 have no verdict yet. **No new code, no new settings: run the existing arms at the two missing seeds, unchanged.**

## Step 0 — Pre-flight (no runs)

1. Confirm the code to be run is the committed tree used for the seed-4126 runs of both arms (source-tree hash in their manifests). If anything on the run path changed since, list the commits and run the 200-round replay for `S1_wen_fresh_s4126` and `S1_wen_fresh_gpu_s4126`: dispatch-log rows identical. Mismatch → stop.
2. Configs `S1_wen_fresh_s4127/_s4128` and `S1_wen_fresh_gpu_s4127/_s4128` differ from their seed-4126 configs only in seed, base path and port (diff them and show the diff).
3. Limits unchanged: `wen_fresh` T_max = 34.334345905647005 s, E_max = 36.423846640394224 J; `wen_fresh_gpu` uses the GPU limits computed in Prompt I Step 2.3 (print both values; they must be identical to the seed-4126 manifests).
4. Predictions v3.3 (`77e2d7b7`) and v3.4 commits are in the manifests' guard.
5. Free disk ≥ 20 GB per planned concurrent run plus margin; checkpoint cleaner running. Make sure no Study 2 or Polaris run is still active; if one is, report it.
6. **Watchdog:** the EIO hangs in Prompt I lost ~6 h. Configure the watchdog so that a run with no new merge for 30 minutes is reported immediately in the run log and in `logs/watchdog_alerts.log`; do not auto-restart — stop and report.

## Step 1 — Runs

| Arm | Scenario | Seeds | Stage 0 |
|---|---|---|---|
| `wen_fresh` | S1 | 4127, 4128 | off (no-op in S1) |
| `wen_fresh_gpu` | S1 | 4127, 4128 | off |

- 4 runs, in parallel. 4,000 rounds max, stop at 90%, identical to every S1 run.
- Manifests: seed, source-tree hash, config hash, limits, predictions v3.3 / v3.4 hashes.
- Online = replayed clock check for every run.
- If a run hangs (watchdog alert), archive it under `_failed/` with the reason and rerun it from scratch; report it as a deviation.

## Step 2 — Report

`evidence/s1_final.md`, mechanical only (no conclusions beyond the P-tables).

**2.1 Main S1 table, 3 seeds each, mean ± sd, and per-seed rows below:** random, Ours (gpu), Polaris, Wen et al. (`wen_fresh`), Wen et al. + GPU (`wen_fresh_gpu`); plus Wen stale (`wen`) as a reference row. Columns:
- TTA@80 / 85 / 90% (online clock; "not reached" with max accuracy and the count of seeds reaching it);
- energy to 80 / 85 / 90% (ground truth E_train + E_up of dispatches finished by that time — same definition as `s1_tta_energy_4arms.md`);
- rounds to 80 / 85 / 90%;
- s/round (whole run; before / after 80%);
- max accuracy;
- dispatch share per tier (rounds 1–500 and last 500), whole-run Gini, distinct vehicles, image coverage of aggregated updates at r500 / r1000 / end;
- mean |x_upload − 1000|, mean |x̂ − x| (where the arm has an x̂).

Reading B for S1 gpu seed 4126 (counted at its maximum, 0.8975) as in every earlier table; footnote it.

**2.2 Verdict tables (v3 rule, all 3 seeds):**
- P22 (S1 part only): wen_fresh Gini < wen.
- P23 (S1 part only): wen_fresh Gini > random.
- P24: gpu TTA@80% < wen_fresh.
- P29: wen_fresh_gpu mean |x̂ − x| < wen_fresh.
- P30: wen_fresh_gpu mean |x_upload − 1000| < wen_fresh.
- P31: wen_fresh_gpu s/round < wen_fresh.
- P32: wen_fresh_gpu Thor share (whole run) > wen_fresh.

For each: per-seed values, direction, held / not held. No re-wording. P25 and P33 are S2 and stay open.

**2.3 Pairwise directions (reported, not predictions)** for Ours vs each other method at 80 / 85 / 90% on TTA and on energy: per-seed sign and the count of seeds in which Ours is lower (e.g. "3/3", "2/3"). One table.

**2.4 Curves:** accuracy vs online clock, S1, mean with min–max band — one panel with random, Ours, Polaris, Wen et al., Wen et al. + GPU; and energy-to-accuracy curves (cumulative ground-truth energy vs accuracy) for the same arms.

**2.5 Deviations and run health:** watchdog alerts, reruns, wall time per run, fresh-evaluation time.

## Stop conditions

- Step 0 replay mismatch, config diff beyond seed / path / port, or limits differing from the seed-4126 manifests → stop before Step 1.
- Online ≠ replayed clock in any run → stop all runs and report.
- Free disk < 20 GB for > 15 minutes → stop launching and report.
- All 4 runs done → write the report and stop (no S2 runs).

## Deliverable message

Step 0 results, run status and wall times, the main S1 table (means), P22–P24 and P29–P32 verdicts, the pairwise-direction table, and every deviation.
