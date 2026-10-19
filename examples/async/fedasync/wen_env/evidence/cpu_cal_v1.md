# cpu_cal v1: CPU-favourable calibration ablation (Prompt D, Step 5)

This file is mechanical: no conclusions beyond the P14 table.

## Predictions and runs

- **Predictions:** `docs/predictions_v3_1_addendum.md`, commit `098e1dd1`, locked before any `cpu_cal` run. v3 is commit `f2bb90a1`.
- **Runs:** S1 `cpu_cal` × seeds 4126, 4127 and 4128. Every setting is identical to the main runs, including the stop rule (`target_accuracy = 0.9`). The runs ended at rounds 2,160, 2,620 and 2,290.
- **Clock check:** all 3 runs pass (online clock equals the replayed clock, maximum difference 0.0 s).

## Calibration

- ρ_cal = mean_i T_train_tier(i) / mean_i (b·|S_i| / f_tier(i)) = **84.98715511665667** cycles/bit. It is computed from the fleet by `wen_estimators.rho_cal` and written into each config and manifest. The entry point recomputes it at start-up and refuses any mismatch.

| Tier | T̂_cpu_cal_train |
|---|---|
| Xavier, \|S\| = 1,667 | 1.5826227674331694 s |
| Orin, \|S\| = 1,667 | 1.5826227674331694 s |
| Orin, \|S\| = 1,666 | 1.5816733836494663 s |
| Thor, \|S\| = 1,666 | 1.3383390169341638 s |

## Step 2 tests: all pass (`wen_env/tests/test_wen_cpu_cal.py`)

1. `cpu_cal` with ρ = 1,250 reproduces the `cpu` dispatch log for seed 4126, checked on 30-round runs `*_rhochk`.
2. The fleet mean of T̂_cpu_cal_train equals the mean datasheet T_train to 1e-9 relative: 1.5417505783860512 against …517.
3. Per-tier values are as in the table above.
4. No tier is blocked at E_max = 600 J. The CPU-family Ê with ρ_cal is about 1.7–2.4 J of training energy plus about 1–2.3 J of upload.

Additional checks:
- the manifest refuses any `cpu_cal` run before the addendum commit;
- the configs carry the computed ρ_cal.

## 1. Main-runs table with `cpu_cal` (S1 arms; mean ± SD over 3 seeds)

S1 gpu and S1 cpu at seed 4126 are the runs that hung when the disk filled; they are counted as reaching 90% by the user's decision (`promptC_progress.md`).

| cell | seeds | TTA@80% (s) | crossing round | plateau | s/round | s/round before / after 80% | energy to 80% (J) | explored | clock ok |
|---|---|---|---|---|---|---|---|---|---|
| S1_cpu | 3 | 1652 ± 1.9e+02 | 443.3 ± 49 | 0.8825 ± 0.0024 | 3.734 ± 0.039 | 3.726 ± 0.034 / 3.735 ± 0.043 | 2.874e+04 ± 4.9e+03 | 0.1999 ± 0.0066 | True |
| S1_cpu_cal | 3 | 1498 ± 2.5e+02 | 466.7 ± 80 | 0.886 ± 0.0026 | 3.364 ± 0.047 | 3.211 ± 0.04 / 3.401 ± 0.043 | 3.288e+04 ± 4.2e+03 | 0.1977 ± 0.0071 | True |
| S1_gpu | 3 | 1315 ± 2.1e+02 | 440 ± 70 | 0.8811 ± 0.0042 | 3.244 ± 0.0095 | 2.989 ± 0.025 / 3.308 ± 0.0051 | 2.216e+04 ± 3.3e+03 | 0.1985 ± 0.0081 | True |
| S1_gpu_eps0 | 3 | 1398 ± 2.7e+02 | 493.3 ± 90 | 0.8824 ± 0.0042 | 3.22 ± 0.077 | 2.831 ± 0.037 / 3.314 ± 0.078 | 2.331e+04 ± 4.1e+03 | 0 ± 0 | True |
| S1_gpu_oracle | 3 | 1360 ± 1.7e+02 | 450 ± 52 | 0.8855 ± 0.0037 | 3.272 ± 0.021 | 3.02 ± 0.024 / 3.333 ± 0.014 | 2.404e+04 ± 3.3e+03 | 0.2014 ± 0.0084 | True |
| S1_gpu_prop | 3 | 1535 ± 1.3e+02 | 410 ± 35 | 0.883 ± 0.0012 | 3.721 ± 0.029 | 3.743 ± 0.056 / 3.714 ± 0.025 | 2.692e+04 ± 1.9e+03 | — | True |
| S1_location_only | 3 | 1441 ± 1.1e+02 | 446.7 ± 35 | 0.8847 ± 0.0062 | 3.349 ± 0.014 | 3.227 ± 0.0074 / 3.381 ± 0.018 | 3.252e+04 ± 2.7e+03 | 0.202 ± 0.012 | True |
| S1_observed | 3 | 1622 ± 1.7e+02 | 446.7 ± 49 | 0.8813 ± 0.003 | 3.69 ± 0.015 | 3.633 ± 0.027 / 3.703 ± 0.017 | 2.77e+04 ± 4.4e+03 | 0.1994 ± 0.0087 | True |
| S1_random | 3 | 1733 ± 3.5e+02 | 463.3 ± 91 | 0.8821 ± 0.0026 | 3.773 ± 0.023 | 3.739 ± 0.014 / 3.784 ± 0.031 | 3.021e+04 ± 6.7e+03 | 1 ± 0 | True |


### Dispatch share per tier (xavier / orin / thor)

| cell | rounds 1–500 | 1,500–2,000 | 3,500–4,000 |
|---|---|---|---|
| S1_cpu | 0.4801 ± 0.043 / 0.2696 ± 0.057 / 0.2503 ± 0.015 | 0.5315 ± 0.031 / 0.3151 ± 0.0075 / 0.1534 ± 0.037 | — / — / — |
| S1_cpu_cal | 0.5611 ± 0.064 / 0.2802 ± 0.056 / 0.1587 ± 0.0092 | 0.5496 ± 0.044 / 0.2947 ± 0.011 / 0.1557 ± 0.041 | — / — / — |
| S1_gpu | 0.3041 ± 0.025 / 0.4309 ± 0.047 / 0.2649 ± 0.029 | 0.5183 ± 0.016 / 0.3143 ± 0.0063 / 0.1674 ± 0.023 | — / — / — |
| S1_gpu_eps0 | 0.251 ± 0.011 / 0.4402 ± 0.046 / 0.3088 ± 0.053 | 0.5489 ± 0.042 / 0.3007 ± 0.043 / 0.1504 ± 0.0012 | — / — / — |
| S1_gpu_oracle | 0.3499 ± 0.024 / 0.3931 ± 0.042 / 0.257 ± 0.038 | 0.5016 ± 0.07 / 0.3178 ± 0.039 / 0.1806 ± 0.035 | — / — / — |
| S1_gpu_prop | 0.4834 ± 0.03 / 0.3526 ± 0.023 / 0.164 ± 0.0081 | 0.4858 ± 0.016 / 0.3472 ± 0.0086 / 0.167 ± 0.025 | — / — / — |
| S1_location_only | 0.5604 ± 0.0094 / 0.2955 ± 0.003 / 0.1441 ± 0.0098 | 0.5647 ± 0.046 / 0.2775 ± 0.035 / 0.1578 ± 0.035 | — / — / — |
| S1_observed | 0.4542 ± 0.052 / 0.342 ± 0.039 / 0.2039 ± 0.042 | 0.5161 ± 0.041 / 0.3104 ± 0.022 / 0.1735 ± 0.052 | — / — / — |
| S1_random | 0.4641 ± 0.03 / 0.3645 ± 0.023 / 0.1713 ± 0.008 | 0.4909 ± 0.039 / 0.3486 ± 0.013 / 0.1604 ± 0.026 | — / — / — |
## 2. P14 (v3 rule: same direction in all 3 seeds)

| P | statement checked | seed 4126 | seed 4127 | seed 4128 | verdict |
|---|---|---|---|---|---|
| P14a | cpu_cal gain over random ≥ ½ × location-only gain | ✓ cpu_cal +9.6% vs ½·loc +5.4% | ✓ cpu_cal +10.8% vs ½·loc +5.7% | ✓ cpu_cal +12.1% vs ½·loc +5.7% | **held** |
| P14b | gpu s/round < cpu_cal s/round | ✓ 3.244 vs 3.393 | ✓ 3.253 vs 3.388 | ✓ 3.234 vs 3.310 | **held** |
| P14c | energy to 80%: gpu < cpu_cal | ✓ 19.4 vs 28.1 kJ | ✓ 21.2 vs 34.6 kJ | ✓ 25.9 vs 36.0 kJ | **held** |
| P14d | cpu_cal s/round < cpu s/round | ✓ 3.393 vs 3.724 | ✓ 3.388 vs 3.777 | ✓ 3.310 vs 3.700 | **held** |

## 3. Step 4: log-only mechanism tables (every S1 arm including cpu_cal, and every S2 arm)

## 1. Ground-truth energy per dispatch (J)

| arm | all | before 80% crossing | after 80% crossing |
|---|---|---|---|
| S1_random | 66.7 ± 1.9 | 65.0 ± 2.5 | 67.3 ± 1.7 |
| S1_cpu | 68.7 ± 2.1 | 64.6 ± 4.3 | 69.8 ± 1.6 |
| S1_cpu_cal | 71.5 ± 1.3 | 70.9 ± 5.0 | 71.9 ± 1.1 |
| S1_gpu | 63.9 ± 0.8 | 50.4 ± 1.2 | 67.4 ± 0.8 |
| S1_observed | 67.3 ± 0.8 | 61.8 ± 4.1 | 68.7 ± 0.2 |
| S1_gpu_oracle | 64.3 ± 0.6 | 53.3 ± 1.9 | 66.9 ± 0.4 |
| S1_gpu_eps0 | 65.5 ± 0.8 | 47.3 ± 0.7 | 70.0 ± 0.2 |
| S1_gpu_prop | 66.6 ± 1.0 | 65.7 ± 2.4 | 66.8 ± 0.8 |
| S1_location_only | 71.7 ± 1.3 | 72.8 ± 0.5 | 71.5 ± 1.8 |
| S2_random | 66.9 ± 1.0 | 66.3 ± 1.3 | 67.1 ± 1.0 |
| S2_gpu | 64.0 ± 0.9 | 51.8 ± 2.2 | 67.5 ± 1.3 |
| S2_gpu_stage0 | 63.7 ± 0.9 | 52.0 ± 0.7 | 66.8 ± 1.1 |
| S2_cpu_stage0 | 67.0 ± 0.6 | 63.6 ± 1.0 | 68.0 ± 0.6 |

## 2. Upload distance and position at training finish

| arm | mean d at training finish (m) | share finishing inside [400, 1600] |
|---|---|---|
| S1_random | 495 ± 3 | 0.609 ± 0.005 |
| S1_cpu | 470 ± 9 | 0.647 ± 0.014 |
| S1_cpu_cal | 298 ± 11 | 0.829 ± 0.012 |
| S1_gpu | 284 ± 1 | 0.844 ± 0.002 |
| S1_observed | 464 ± 2 | 0.647 ± 0.003 |
| S1_gpu_oracle | 294 ± 4 | 0.834 ± 0.003 |
| S1_gpu_eps0 | 262 ± 28 | 0.877 ± 0.027 |
| S1_gpu_prop | 478 ± 8 | 0.626 ± 0.014 |
| S1_location_only | 291 ± 10 | 0.833 ± 0.016 |
| S2_random | 301 ± 5 | 0.970 ± 0.002 |
| S2_gpu | 214 ± 2 | 0.985 ± 0.001 |
| S2_gpu_stage0 | 181 ± 3 | 1.000 ± 0.000 |
| S2_cpu_stage0 | 391 ± 4 | 0.999 ± 0.000 |

## 3. True time per dispatch (s)

| arm | mean T_train | mean T_up |
|---|---|---|
| S1_random | 1.440 ± 0.054 | 9.875 ± 0.022 |
| S1_cpu | 1.494 ± 0.059 | 9.697 ± 0.069 |
| S1_cpu_cal | 1.585 ± 0.035 | 8.499 ± 0.110 |
| S1_gpu | 1.362 ± 0.021 | 8.364 ± 0.018 |
| S1_observed | 1.457 ± 0.023 | 9.606 ± 0.024 |
| S1_gpu_oracle | 1.375 ± 0.020 | 8.437 ± 0.056 |
| S1_gpu_eps0 | 1.408 ± 0.024 | 8.247 ± 0.222 |
| S1_gpu_prop | 1.437 ± 0.030 | 9.723 ± 0.062 |
| S1_location_only | 1.593 ± 0.036 | 8.449 ± 0.053 |
| S2_random | 1.449 ± 0.028 | 8.765 ± 0.063 |
| S2_gpu | 1.363 ± 0.024 | 8.003 ± 0.028 |
| S2_gpu_stage0 | 1.357 ± 0.025 | 7.772 ± 0.033 |
| S2_cpu_stage0 | 1.446 ± 0.016 | 9.469 ± 0.014 |

## 4. Dispatch share per tier (xavier / orin / thor)

| arm | rounds 1–500 | last 500 rounds of the run |
|---|---|---|
| S1_random | 0.464 ± 0.030 / 0.365 ± 0.023 / 0.171 ± 0.008 | 0.501 ± 0.027 / 0.340 ± 0.006 / 0.159 ± 0.021 |
| S1_cpu | 0.480 ± 0.043 / 0.270 ± 0.057 / 0.250 ± 0.015 | 0.531 ± 0.033 / 0.289 ± 0.043 / 0.180 ± 0.014 |
| S1_cpu_cal | 0.561 ± 0.064 / 0.280 ± 0.056 / 0.159 ± 0.009 | 0.533 ± 0.026 / 0.275 ± 0.023 / 0.192 ± 0.020 |
| S1_gpu | 0.304 ± 0.025 / 0.431 ± 0.047 / 0.265 ± 0.029 | 0.528 ± 0.013 / 0.302 ± 0.014 / 0.170 ± 0.007 |
| S1_observed | 0.454 ± 0.052 / 0.342 ± 0.039 / 0.204 ± 0.042 | 0.490 ± 0.003 / 0.308 ± 0.007 / 0.202 ± 0.009 |
| S1_gpu_oracle | 0.350 ± 0.024 / 0.393 ± 0.042 / 0.257 ± 0.038 | 0.493 ± 0.009 / 0.320 ± 0.017 / 0.187 ± 0.013 |
| S1_gpu_eps0 | 0.251 ± 0.011 / 0.440 ± 0.046 / 0.309 ± 0.053 | 0.587 ± 0.039 / 0.260 ± 0.061 / 0.153 ± 0.031 |
| S1_gpu_prop | 0.483 ± 0.030 / 0.353 ± 0.023 / 0.164 ± 0.008 | 0.497 ± 0.006 / 0.340 ± 0.012 / 0.163 ± 0.016 |
| S1_location_only | 0.560 ± 0.009 / 0.295 ± 0.003 / 0.144 ± 0.010 | 0.537 ± 0.016 / 0.303 ± 0.033 / 0.161 ± 0.017 |
| S2_random | 0.478 ± 0.015 / 0.344 ± 0.004 / 0.178 ± 0.015 | 0.499 ± 0.014 / 0.328 ± 0.005 / 0.173 ± 0.019 |
| S2_gpu | 0.316 ± 0.031 / 0.405 ± 0.008 / 0.279 ± 0.038 | 0.515 ± 0.029 / 0.320 ± 0.018 / 0.165 ± 0.037 |
| S2_gpu_stage0 | 0.322 ± 0.019 / 0.395 ± 0.017 / 0.283 ± 0.005 | 0.481 ± 0.004 / 0.334 ± 0.014 / 0.185 ± 0.017 |
| S2_cpu_stage0 | 0.456 ± 0.017 / 0.277 ± 0.007 / 0.267 ± 0.015 | 0.501 ± 0.016 / 0.332 ± 0.021 / 0.168 ± 0.006 |

## 5. Position-estimate error: mean |x̂_finish − x_finish| (m)

| arm | error |
|---|---|
| S1_random | — |
| S1_cpu | 557.1 ± 4.9 |
| S1_cpu_cal | 29.7 ± 0.2 |
| S1_gpu | 4.7 ± 0.1 |
| S1_observed | — |
| S1_gpu_oracle | 0.0 ± 0.0 |
| S1_gpu_eps0 | 4.8 ± 0.2 |
| S1_gpu_prop | 4.9 ± 0.1 |
| S1_location_only | 30.8 ± 0.4 |
| S2_random | — |
| S2_gpu | 4.7 ± 0.1 |
| S2_gpu_stage0 | 4.6 ± 0.1 |
| S2_cpu_stage0 | 547.0 ± 2.2 |
