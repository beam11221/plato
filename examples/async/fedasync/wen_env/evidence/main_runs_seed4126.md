# Main runs, Phase A: seed 4126 (Prompt C, Step 6)

This file reports mechanical numbers only, with no interpretation.
- Predictions: `docs/predictions_v3.md`, locked at commit `f2bb90a1` before any full run.
- Time axis: the online simulated clock. Accuracy is evaluated every 10 rounds.
- All 11 runs pass the clock check: the online clock equals the replayed clock with a maximum difference of 0.0 s, and there are 0 merge-order residuals.

## Deviations that affect these numbers (details in `promptC_progress.md`)

- **The run budget is "stop at 90% accuracy", not 4,000 rounds.** `target_accuracy = 0.9` ends every run when it is reached; the user decided to keep it. Runs end between rounds 1,720 and 2,110. The share window for rounds 3,500–4,000 is therefore empty. "Plateau" (the mean accuracy of the last 200 rounds) is the approach to 90%, not a true plateau.
- **S1 gpu and S1 cpu are the runs that hung when the disk filled.** They hung at rounds 1,957 and 1,929, at maximum accuracies of 89.75% and 89.68%, which the user rounded to 90%. Their logs are intact up to the hang. Every other cell ran to its own 90% stop.
- The pilot-gate plateau test fails for this reason (`pilot_gate.md`). By the user's instruction, Phase A continued anyway.

## 1. Per cell (single seed)

| Cell | TTA@80% (s) | 80% crossing round | Plateau (last 200 rounds) | s / round, whole run | s / round, before / after 80% | Energy to 80% (J, ground truth) | Explored share |
|---|---|---|---|---|---|---|---|
| S1 random (ε = 1) | 1,419 | 380 | 0.8847 | 3.754 | 3.734 / 3.760 | 23,600 | 1.000 |
| S1 cpu | 1,535 | 410 | 0.8824 | 3.724 | 3.743 / 3.719 | 26,590 | 0.206 |
| S1 gpu | 1,158 | 390 | 0.8787 | 3.244 | 2.969 / 3.313 | 19,390 | 0.207 |
| S1 observed | 1,426 | 390 | 0.8801 | 3.681 | 3.656 / 3.688 | 23,090 | 0.208 |
| S1 gpu_oracle | 1,168 | 390 | 0.8818 | 3.249 | 2.994 / 3.317 | 20,340 | 0.210 |
| S1 gpu ε = 0 | 1,117 | 400 | 0.8780 | 3.145 | 2.791 / 3.228 | 18,900 | 0.000 |
| S1 gpu proportional | 1,386 | 370 | 0.8843 | 3.740 | 3.747 / 3.738 | 24,960 | — |
| S2 random (ε = 1) | 1,813 | 410 | 0.8815 | 4.425 | 4.422 / 4.427 | 34,560 | 1.000 |
| S2 gpu | 1,348 | 420 | 0.8841 | 3.573 | 3.211 / 3.673 | 24,750 | 0.211 |
| S2 gpu + Stage 0 | 1,109 | 380 | 0.8830 | 3.069 | 2.917 / 3.112 | 20,280 | 0.214 |
| S2 cpu + Stage 0 | 1,548 | 430 | 0.8844 | 3.665 | 3.599 / 3.684 | 27,460 | 0.209 |

Energy to 80% is the ground-truth energy of every dispatch that finished at or before TTA@80%.

### Dispatch share per tier (Xavier / Orin / Thor)

| Cell | Rounds 1–500 | Rounds 1,500–2,000 | Rounds 3,500–4,000 |
|---|---|---|---|
| S1 random | 0.43 / 0.39 / 0.18 | 0.45 / 0.36 / 0.19 | — |
| S1 cpu | 0.49 / 0.25 / 0.26 | 0.53 / 0.31 / 0.16 | — |
| S1 gpu | 0.32 / 0.38 / 0.30 | 0.51 / 0.31 / 0.18 | — |
| S1 observed | 0.42 / 0.32 / 0.25 | 0.47 / 0.31 / 0.22 | — |
| S1 gpu_oracle | 0.35 / 0.35 / 0.30 | 0.42 / 0.36 / 0.21 | — |
| S1 gpu ε = 0 | 0.26 / 0.43 / 0.31 | 0.55 / 0.30 / 0.15 | — |
| S1 gpu proportional | 0.51 / 0.34 / 0.16 | 0.50 / 0.35 / 0.15 | — |
| S2 random | 0.46 / 0.35 / 0.19 | 0.47 / 0.35 / 0.18 | — |
| S2 gpu | 0.34 / 0.42 / 0.24 | 0.48 / 0.31 / 0.21 | — |
| S2 gpu + Stage 0 | 0.34 / 0.37 / 0.28 | 0.46 / 0.35 / 0.20 | — |
| S2 cpu + Stage 0 | 0.45 / 0.27 / 0.28 | 0.48 / 0.34 / 0.18 | — |

For reference, the fleet is 0.50 / 0.33 / 0.17.

### S2: waste, Stage 0 and dispatch rate

| Cell | Waste (dispatches) | Waste (GPU-s) | Waste (J) | Stage 0 rejected | Stage 0 fallback cycles | Dispatches / s | Accepted / s |
|---|---|---|---|---|---|---|---|
| S2 random | 23.1% (533 / 2,305) | 819.1 | 37,480 | — | — | 0.294 | 0.226 |
| S2 gpu | 12.3% (273 / 2,214) | 444.0 | 20,120 | — | — | 0.319 | 0.280 |
| S2 gpu + Stage 0 | 0.81% (14 / 1,736) | 17.3 | 859.5 | 24.0% | 0 of 1,734 | 0.328 | 0.326 |
| S2 cpu + Stage 0 | 0.47% (9 / 1,901) | 17.1 | 751.6 | 73.4% | 33 of 1,899 | 0.274 | 0.273 |

## 2. Predictions v3: direction for seed 4126

**Status of every row: single seed, not evaluated.** The held / not-held verdict under the v3 rule (same direction in all 3 seeds) is filled in only after Phase B. Prediction texts are as locked; the comparisons below are mechanical.

| P | Comparison (v3 wording, abbreviated) | Seed 4126 | Direction as predicted? | Status |
|---|---|---|---|---|
| P1 | cpu's s/round gain over random is less than half of gpu's | cpu −0.8%, gpu −13.6% against random (whole run) | yes | single seed — not evaluated |
| P2 | gpu beats cpu on s/round and TTA@80% | s/round 3.244 vs 3.724; TTA 1,158 vs 1,535 s | yes | single seed — not evaluated |
| P3 | gpu beats random on s/round before 80%, over the whole run, and on TTA@80% | before 2.969 vs 3.734; whole 3.244 vs 3.754; TTA 1,158 vs 1,419 s | yes | single seed — not evaluated |
| P4 | gpu plateau within ±1 point of random | 0.8787 vs 0.8847 (−0.60 points) | yes | single seed — not evaluated (plateau definition affected by the 90% stop) |
| P5 | energy to reach 80%: gpu < random | 19,390 vs 23,600 J | yes | single seed — not evaluated |
| P6 | Wen + GPU vs Wen-faithful | — | — | not run (Wen selector is a later prompt) |
| P7 | gpu ≈ gpu_oracle | TTA 1,158 vs 1,168 s; s/round 3.244 vs 3.249; plateau 0.8787 vs 0.8818 | (no direction predicted) | single seed — not evaluated |
| P8 | observed does not beat gpu in any phase | before 80%: 3.656 vs 2.969; after: 3.688 vs 3.313; TTA 1,426 vs 1,158 s | yes (gpu ahead in every phase) | single seed — not evaluated |
| P9 | Stage 0 reduces waste (gpu; Wen-faithful) | gpu: 12.3% → 0.81% of dispatches | yes, for gpu; Wen-faithful not run | single seed — not evaluated (gpu direction was disclosed as already seen) |
| P10 | gpu + Stage 0 waste near zero; cpu + Stage 0 admits fewer dispatches per unit time than gpu + Stage 0 | waste 0.81%; dispatches/s 0.274 (cpu) vs 0.328 (gpu); accepted/s 0.273 vs 0.326 | yes / yes | single seed — not evaluated |
| P11 | Stage 0 does not lower the plateau | S2 gpu + Stage 0 0.8830 vs S2 gpu 0.8841 (−0.11 points) | **no** (lower by 0.11 points; magnitude not predicted) | single seed — not evaluated (plateau definition affected by the 90% stop) |
| P12 | location-only | — | — | not run |
| P13 | proportional ≈ random on s/round; ε = 0 gain ≥ ε = 0.2 gain; ε = 0 plateau ≤ ε = 0.2 plateau | proportional 3.740 vs random 3.754 (−0.4%; seed spread unknown with one seed); gain −16.2% vs −13.6%; plateau 0.8780 vs 0.8787 | ≈ (spread not available) / yes / yes | single seed — not evaluated |

## 3. Curves

`main_runs_seed4126_curves.png` plots accuracy against the online clock, one panel per scenario. With one seed there is no min–max band.

## 4. Run inventory

**Runs used.** Every run below is linked from `runs/wen_env/_phaseA/`.

| Cell | Directory | Last round | Final accuracy |
|---|---|---|---|
| S1 random | `S1_random_s4126` | 1,770 | 90.29% |
| S1 cpu | `_failed/S1_cpu_s4126_hung_diskfull_0740` | 1,929 | 89.68% max (rounded to 90%) |
| S1 gpu | `_failed/S1_gpu_s4126_hung_diskfull_0740` | 1,957 | 89.75% max (rounded to 90%) |
| S1 observed | `S1_observed_s4126` | 1,880 | 90.22% |
| S1 gpu_oracle | `S1_gpu_oracle_s4126` | 1,830 | 90.08% |
| S1 gpu ε = 0 | `S1_gpu_eps0_s4126` | 2,110 | 90.39% |
| S1 gpu proportional | `S1_gpu_prop_s4126` | 1,920 | 90.06% |
| S2 random | `S2_random_s4126` | 1,770 | 90.40% |
| S2 gpu | `S2_gpu_s4126` | 1,940 | 90.21% |
| S2 gpu + Stage 0 | `S2_gpu_stage0_s4126` | 1,720 | 90.11% |
| S2 cpu + Stage 0 | `S2_cpu_stage0_s4126` | 1,890 | 90.10% |

**Not used**, archived under `runs/wen_env/_failed/`:
- `S1_gpu_oracle_s4126_hung_r1501`: cleaner payload race.
- `*_hung_diskfull_0740` for gpu ε = 0, gpu_oracle and proportional: hung at the disk-full moment.
- `S1_gpu_s4126_rerun_stopped_by_user_r133` and `S1_cpu_s4126_rerun_stopped_by_user_r122`.
