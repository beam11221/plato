# Wen-faithful v1: Study 1, ours vs Wen-faithful vs random (Prompt E, Step 7)

This file is mechanical: no conclusions beyond the P15–P21 table.

## Locked inputs

- **Predictions:**
  - `docs/predictions_v3_2_addendum.md`, commit `66a2ed82`, locked before any `wen` run;
  - v3 at `f2bb90a1`;
  - v3.1 at `098e1dd1`.
- **ADR:** `docs/decisions/ADR-0017-wen-faithful-async.md`.
- **Runs:** `wen` × S1 and S2 × seeds 4126, 4127 and 4128, with Stage 0 off. The stop rule is `target_accuracy = 0.9` with at most 4,000 rounds, as in the main runs.
- **Clock check:** all 6 runs pass (online clock equals the replayed clock, maximum difference 0.0 s, 0 merge-order residuals).
- **Comparators:** existing runs (W9):
  - S1: random (ε = 1), gpu (ε = 0.2), location_only, cpu;
  - S2: random and gpu, with Stage 0 off.

## Step 1: constraint limits (W5)

- **T_max = 34.334345905647005 s** and **E_max = 36.423846640394224 J**.
- These are the 90th percentiles of F1 and F2 over 30 vehicles × 30,000 trace samples (900,000 pairs).
- Qualitative check: PASS. Thor is not fully excluded, and both limits cut some candidates.
- Full table: `wen_limits.md`. Fleet total cut: 20.0% (T only 10.0%, E only 10.0%). T_max cuts only Xavier and Orin; E_max cuts only Thor (60.0%).
- x̂ reuses `Estimator.upload_hat` → `wen_formulas.wrap_x`, the gpu arm's own wrap.

## Step 3: tests (`wen_env/tests/test_wen_faithful.py`) — all pass

1. **F1 and F2** for one vehicle per tier at x = 200, 1,000 and 1,700 match a hand calculation to 1e-9 relative:

| tier | x | F1 (s) | F2 (J) | x̂ |
|---|---|---|---|---|
| Xavier / Orin | 200 | 31.330682 | 26.396416 | 781.9 |
| Xavier / Orin | 1000 | 33.234330 | 26.777146 | 1581.9 |
| Xavier / Orin | 1700 | 33.765616 | 26.883403 | 281.9 |
| Thor | 200 | 28.317418 | 36.323953 | 692.1 |
| Thor | 1000 | 29.254760 | 36.511421 | 1492.1 |
| Thor | 1700 | 30.496161 | 36.759702 | 192.1 |

   All rows use V = 25 m/s.

2. **F4** matches a hand calculation for clients 1, 16 and 30. The fleet range (seed 4126) is 0.1664–0.6273, with a mean of 0.3902.
3. **Non-dominated filter:** correct on a synthetic 10-point case. The normalisation edge cases (range 0, a single member) are handled, and so are tie-breaking and the feasibility fallback.
4. **Determinism:** two 100-round S1 runs at seed 4126 give identical dispatch logs and identical `wen_decisions.csv`.
5. **F3:** a never-trained vehicle gets the fleet-mean prior. Before any report, every F3 is equal and normalises to 0. Lost S2 uploads do not update F3.
6. **Ground truth** for every `wen` dispatch equals the GPU formulas.
7. **No ground-truth reads:** `wen_faithful.py` imports no ground-truth module, and it scores only `ReportedVehicle` records.

The full suite, `wen_env/tests`, passes: 166 tests.

## Step 4: smoke (200 rounds, seed 4126)

See `wen_smoke.md`.

| Measure | S1 | S2 |
|---|---|---|
| s / round | 3.492 | 3.403 |
| Share Xavier / Orin / Thor | 0.22 / 0.78 / 0.00 | 0.33 / 0.66 / 0.00 |
| Fallback rate | 0 | 0 |
| Distinct vehicles | 15 | 23 |
| Mean \|x̂ − x\| | 508 m | 485 m |
| Mean upload distance | 449 m | 307 m |

## 1. Study 1 — S1 (mean ± SD over 3 seeds)

| arm | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | s/round | s/round before / after 80% | energy to 80% (kJ) | share X/O/T rounds 1–500 | share X/O/T last 500 | waste (disp / GPU-s / kJ) | mean \|x_upload − 1000\| (m) | mean \|x̂ − x\| (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_random | 1733 ± 345 | 3102 ± 500 | 7108 ± 573 | 3.773 ± 0.023 | 3.739 ± 0.014 / 3.784 ± 0.031 | 30.2 ± 6.7 | 0.46 ± 0.03 / 0.36 ± 0.02 / 0.17 ± 0.01 | 0.50 ± 0.03 / 0.34 ± 0.01 / 0.16 ± 0.02 | — | 495 ± 3 | — |
| S1_gpu | 1315 ± 207 | 2429 ± 295 | 7508 ± 191 (1/3 not reached; last 0.897) | 3.244 ± 0.009 | 2.989 ± 0.025 / 3.308 ± 0.005 | 22.2 ± 3.3 | 0.30 ± 0.03 / 0.43 ± 0.05 / 0.26 ± 0.03 | 0.53 ± 0.01 / 0.30 ± 0.01 / 0.17 ± 0.01 | — | 283 ± 1 | 5 ± 0 |
| S1_wen | 2772 ± 150 | 9734 ± 1147 | — (3/3 not reached; last 0.824, 0.753, 0.858) | 3.632 ± 0.225 | 3.611 ± 0.249 / 3.638 ± 0.221 | 38.2 ± 12.1 | 0.33 ± 0.26 / 0.67 ± 0.26 / 0.00 ± 0.00 | 0.27 ± 0.19 / 0.73 ± 0.19 / 0.00 ± 0.00 | — | 513 ± 43 | 574 ± 51 |
| S1_location_only | 1441 ± 110 | 2550 ± 12 | 7403 ± 1457 | 3.349 ± 0.014 | 3.227 ± 0.007 / 3.381 ± 0.018 | 32.5 ± 2.7 | 0.56 ± 0.01 / 0.30 ± 0.00 / 0.14 ± 0.01 | 0.54 ± 0.02 / 0.30 ± 0.03 / 0.16 ± 0.02 | — | 291 ± 10 | 31 ± 0 |
| S1_cpu | 1652 ± 192 | 2862 ± 557 | 8495 ± 1393 (1/3 not reached; last 0.890) | 3.734 ± 0.039 | 3.726 ± 0.034 / 3.735 ± 0.043 | 28.7 ± 4.9 | 0.48 ± 0.04 / 0.27 ± 0.06 / 0.25 ± 0.02 | 0.53 ± 0.03 / 0.29 ± 0.04 / 0.18 ± 0.01 | — | 470 ± 9 | 557 ± 5 |

## 1. Study 1 — S2 (Stage 0 off for every arm)

| arm | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | s/round | s/round before / after 80% | energy to 80% (kJ) | share X/O/T rounds 1–500 | share X/O/T last 500 | waste (disp / GPU-s / kJ) | mean \|x_upload − 1000\| (m) | mean \|x̂ − x\| (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S2_random | 1817 ± 140 | 3116 ± 113 | 7685 ± 627 | 4.406 ± 0.053 | 4.359 ± 0.060 / 4.421 ± 0.058 | 35.3 ± 2.9 | 0.48 ± 0.02 / 0.34 ± 0.00 / 0.18 ± 0.01 | 0.50 ± 0.01 / 0.33 ± 0.01 / 0.17 ± 0.02 | 0.227 ± 0.005 / 810 ± 106 / 36.8 ± 4.5 | 301 ± 5 | — |
| S2_gpu | 1416 ± 187 | 2484 ± 471 | 6869 ± 394 | 3.560 ± 0.030 | 3.289 ± 0.093 / 3.638 ± 0.044 | 24.9 ± 2.1 | 0.32 ± 0.03 / 0.41 ± 0.01 / 0.28 ± 0.04 | 0.52 ± 0.03 / 0.32 ± 0.02 / 0.16 ± 0.04 | 0.123 ± 0.004 / 435 ± 15 / 19.7 ± 0.6 | 213 ± 2 | 5 ± 0 |
| S2_wen | 1608 ± 200 | 3211 ± 235 | 12924 ± 1495 | 3.578 ± 0.136 | 3.618 ± 0.293 / 3.572 ± 0.113 | 29.1 ± 6.4 | 0.43 ± 0.11 / 0.56 ± 0.12 / 0.00 ± 0.01 | 0.39 ± 0.10 / 0.61 ± 0.10 / 0.00 ± 0.00 | 0.055 ± 0.015 / 218 ± 129 / 11.1 ± 5.2 | 324 ± 13 | 537 ± 29 |

## 2. Constraint table (all wen decisions per scenario, 3 seeds pooled)

| scenario | tier | cut by T only | cut by E only | cut by both | fallback rate | mean n_feasible | mean n_pareto |
|---|---|---|---|---|---|---|---|
| S1 | xavier | 0.125 | 0.000 | 0.000 | 0.0000 | 22.07 | 7.32 |
| S1 | orin | 0.133 | 0.000 | 0.000 | 0.0000 | 22.07 | 7.32 |
| S1 | thor | 0.000 | 0.598 | 0.000 | 0.0000 | 22.07 | 7.32 |
| S2 | xavier | 0.203 | 0.000 | 0.000 | 0.0000 | 11.58 | 4.97 |
| S2 | orin | 0.201 | 0.000 | 0.000 | 0.0000 | 11.58 | 4.97 |
| S2 | thor | 0.000 | 0.626 | 0.000 | 0.0000 | 11.58 | 4.97 |

## 3. Participation (per-vehicle dispatch counts; mean ± SD over seeds)

| arm | min dispatches per vehicle | max | vehicles with 0 | Gini |
|---|---|---|---|---|
| S1_random | 47 ± 10 | 80 ± 10 | 0.0 ± 0.0 | 0.069 ± 0.017 |
| S1_gpu | 41 ± 3 | 116 ± 6 | 0.0 ± 0.0 | 0.125 ± 0.009 |
| S1_wen | 0 ± 1 | 999 ± 91 | 1.7 ± 2.1 | 0.816 ± 0.019 |
| S2_random | 55 ± 4 | 91 ± 7 | 0.0 ± 0.0 | 0.056 ± 0.001 |
| S2_gpu | 41 ± 8 | 119 ± 15 | 0.0 ± 0.0 | 0.134 ± 0.014 |
| S2_wen | 0 ± 1 | 588 ± 68 | 2.3 ± 2.1 | 0.657 ± 0.027 |

## 4. P15–P21 (v3 rule: same direction in all 3 seeds)

| P | comparison | seed 4126 | seed 4127 | seed 4128 | verdict |
|---|---|---|---|---|---|
| P15 | wen mean |x_upload − 1000| > location_only's (S1) | ✓ 465 vs 279 m | ✓ 526 vs 298 m | ✓ 549 vs 295 m | **held** |
| P16 | gpu s/round < wen (S1) | ✓ 3.244 vs 3.460 | ✓ 3.253 vs 3.549 | ✓ 3.234 vs 3.887 | **held** |
| P17 | gpu TTA@80% < wen (S1) | ✓ 1158 vs 2625 s | ✓ 1237 vs 2925 s | ✓ 1550 vs 2767 s | **held** |
| P18 | energy to 80%: gpu < wen (S1) | ✓ 19.4 vs 33.0 kJ | ✓ 21.2 vs 29.7 kJ | ✓ 25.9 vs 52.0 kJ | **held** |
| P19 | wen Thor share (whole run) < gpu's (S1) | ✓ 0.001 vs 0.209 | ✓ 0.000 vs 0.195 | ✓ 0.002 vs 0.191 | **held** |
| P20 | wen waste fraction < random's (S2) | ✓ 0.045 vs 0.231 | ✓ 0.048 vs 0.228 | ✓ 0.072 vs 0.221 | **held** |
| P21 | gpu (Stage 0 off) TTA@80% < wen (S2) | ✓ 1348 vs 1412 s | ✓ 1273 vs 1600 s | ✓ 1628 vs 1811 s | **held** |

## 5. Curves

`wen_faithful_v1_curves.png` plots accuracy against the online clock, one panel per scenario, for random, gpu and wen. The line is the mean over 3 seeds, with a min–max band.

## 6. Step 6: TTA@80 / 85 / 90% (online clock) for every run (45 runs)

- The main runs, location_only, cpu_cal and wen are all included.
- "Not reached" gives the run's last evaluated accuracy.
- S1 gpu and S1 cpu at seed 4126 are the runs that hung when the disk filled, at rounds 1,957 and 1,929. This table reports them as not reaching 90%. By the user's earlier decision, they count as 90% in `main_runs_v1.md` and `cpu_cal_v1.md`.

| run | TTA@80% | TTA@85% | TTA@90% | last round | last accuracy |
|---|---|---|---|---|---|
| S1_cpu_s4126 | 1535 s (r410) | 2837 s (r750) | not reached (last 0.8901) | 1920 | 0.8901 |
| S1_gpu_eps0_s4126 | 1117 s (r400) | 2100 s (r720) | 6636 s (r2110) | 2110 | 0.9039 |
| S1_gpu_oracle_s4126 | 1168 s (r390) | 2296 s (r740) | 5945 s (r1830) | 1830 | 0.9008 |
| S1_gpu_prop_s4126 | 1386 s (r370) | 2530 s (r670) | 7180 s (r1920) | 1920 | 0.9006 |
| S1_gpu_s4126 | 1158 s (r390) | 2089 s (r680) | not reached (last 0.8975) | 1950 | 0.8975 |
| S1_observed_s4126 | 1426 s (r390) | 2962 s (r810) | 6920 s (r1880) | 1880 | 0.9022 |
| S1_random_s4126 | 1419 s (r380) | 2568 s (r680) | 6645 s (r1770) | 1770 | 0.9029 |
| S2_cpu_stage0_s4126 | 1548 s (r430) | 2744 s (r750) | 6927 s (r1890) | 1890 | 0.9010 |
| S2_gpu_s4126 | 1348 s (r420) | 2254 s (r670) | 6931 s (r1940) | 1940 | 0.9021 |
| S2_gpu_stage0_s4126 | 1109 s (r380) | 2003 s (r680) | 5279 s (r1720) | 1720 | 0.9011 |
| S2_random_s4126 | 1813 s (r410) | 3002 s (r670) | 7833 s (r1770) | 1770 | 0.9040 |
| S1_location_only_s4126 | 1326 s (r410) | 2547 s (r780) | 5723 s (r1710) | 1710 | 0.9006 |
| S1_cpu_cal_s4126 | 1249 s (r390) | 2265 s (r690) | 7329 s (r2160) | 2160 | 0.9026 |
| S1_wen_s4126 | 2625 s (r760) | 11049 s (r3200) | not reached (last 0.8240) | 4000 | 0.8240 |
| S2_wen_s4126 | 1412 s (r420) | 3110 s (r910) | 11262 s (r3250) | 3250 | 0.9010 |
| S1_random_s4127 | 1678 s (r450) | 3177 s (r840) | 7749 s (r2040) | 2040 | 0.9003 |
| S1_cpu_s4127 | 1874 s (r500) | 3431 s (r910) | 9479 s (r2510) | 2510 | 0.9008 |
| S1_cpu_cal_s4127 | 1497 s (r460) | 3099 s (r940) | 8876 s (r2620) | 2620 | 0.9002 |
| S1_gpu_s4127 | 1237 s (r410) | 2616 s (r840) | 7644 s (r2350) | 2350 | 0.9007 |
| S1_observed_s4127 | 1748 s (r480) | 3364 s (r910) | 9638 s (r2600) | 2600 | 0.9010 |
| S1_gpu_oracle_s4127 | 1460 s (r480) | 2479 s (r800) | 8233 s (r2510) | 2510 | 0.9032 |
| S1_gpu_eps0_s4127 | 1432 s (r500) | 2972 s (r980) | 10256 s (r3110) | 3110 | 0.9036 |
| S1_gpu_prop_s4127 | 1585 s (r430) | 3005 s (r810) | 8149 s (r2210) | 2210 | 0.9026 |
| S1_location_only_s4127 | 1546 s (r480) | 2563 s (r790) | 8310 s (r2470) | 2470 | 0.9006 |
| S1_wen_s4127 | 2925 s (r840) | 8938 s (r2540) | not reached (last 0.7532) | 4000 | 0.7532 |
| S2_random_s4127 | 1959 s (r450) | 3118 s (r720) | 8226 s (r1850) | 1850 | 0.9017 |
| S2_gpu_s4127 | 1273 s (r390) | 2171 s (r670) | 7229 s (r2050) | 2050 | 0.9002 |
| S2_gpu_stage0_s4127 | 1448 s (r500) | 2694 s (r900) | 7631 s (r2470) | 2470 | 0.9000 |
| S2_cpu_stage0_s4127 | 1803 s (r500) | 3115 s (r860) | 7499 s (r2050) | 2050 | 0.9009 |
| S2_wen_s4127 | 1600 s (r450) | 3044 s (r870) | 14159 s (r4000) | 4000 | 0.9008 |
| S1_random_s4128 | 2102 s (r560) | 3559 s (r940) | 6931 s (r1840) | 1840 | 0.9025 |
| S1_cpu_s4128 | 1549 s (r420) | 2318 s (r630) | 7510 s (r2030) | 2030 | 0.9009 |
| S1_cpu_cal_s4128 | 1747 s (r550) | 2634 s (r820) | 7579 s (r2290) | 2290 | 0.9004 |
| S1_gpu_s4128 | 1550 s (r520) | 2581 s (r850) | 7373 s (r2280) | 2280 | 0.9006 |
| S1_observed_s4128 | 1693 s (r470) | 3143 s (r860) | 8650 s (r2350) | 2350 | 0.9017 |
| S1_gpu_oracle_s4128 | 1452 s (r480) | 2510 s (r810) | 8678 s (r2640) | 2640 | 0.9035 |
| S1_gpu_eps0_s4128 | 1645 s (r580) | 2684 s (r910) | 7911 s (r2460) | 2460 | 0.9001 |
| S1_gpu_prop_s4128 | 1633 s (r430) | 2661 s (r710) | 7135 s (r1910) | 1910 | 0.9026 |
| S1_location_only_s4128 | 1452 s (r450) | 2539 s (r790) | 8176 s (r2450) | 2450 | 0.9012 |
| S1_wen_s4128 | 2767 s (r710) | 9214 s (r2370) | not reached (last 0.8578) | 4000 | 0.8578 |
| S2_random_s4128 | 1678 s (r390) | 3228 s (r750) | 6997 s (r1610) | 1610 | 0.9034 |
| S2_gpu_s4128 | 1628 s (r480) | 3025 s (r870) | 6448 s (r1800) | 1800 | 0.9001 |
| S2_gpu_stage0_s4128 | 1402 s (r480) | 2494 s (r850) | 7005 s (r2290) | 2290 | 0.9018 |
| S2_cpu_stage0_s4128 | 1732 s (r480) | 2848 s (r780) | 8443 s (r2300) | 2300 | 0.9029 |
| S2_wen_s4128 | 1811 s (r460) | 3480 s (r900) | 13352 s (r3580) | 3580 | 0.9005 |

## Deviations from Prompt E

1. **The first smoke attempt crashed at start-up.** A log line in `install_wen` read `epsilon`, which the wen selector has no attribute for. It was fixed with `getattr`, and both smoke runs were restarted from scratch. No dispatch had happened.
2. **F4 histograms are computed on the server.** They are rebuilt from each client's disjoint partition (`wen_partition`, same labels and seed), which is exactly the histogram a client would report at registration. No message is added to PLATO's protocol.
3. **`pick_rank_in_pareto` is always 1** by construction, because the pick is the minimum score; ties go to the lower id, which has rank 1.
4. **Upload position for P15** is the position where training finishes, i.e. where the upload starts: `x_finish` in the dispatch log. P15 uses |x_finish − 1000|.
5. **x̂ for the non-wen arms** in the study table is recomputed from the logged `x_dispatch`, `V_k` and `T_hat_train` with the estimator's own wrap. For wen it is the logged `x_hat_upload`.
6. **S1 wen did not reach 90% in any seed** within the 4,000-round cap. Last accuracies: 0.824, 0.753, 0.858; maxima: 0.851, 0.862, 0.872. TTA@90% for S1 wen is "not reached" in all 3 seeds. This is a result, not a run failure.
7. **S2 wen at seed 4127 reached 90.08% exactly at round 4,000,** the cap.
