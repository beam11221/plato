# Study 2 plug-in study, seed 4126 — final

**Status (21 Oct 2026).** All 5 cells finished. Cells 1–2 stopped at 90%; cells 3–5 ran the full 4,000 rounds without reaching 90%.

**Deviations.**
- Cells 3 (`S1_wen_fresh_gpu`) and 4 (`S2_wen_fresh_gpu`) hung on 19 Oct at ~23:30–23:42 UTC on `OSError: [Errno 5] Input/output error` while writing a payload / model file to /workspace (MooseFS; disk 60.5 / 200 GB, writes worked again afterwards). They were stopped on 20 Oct 06:03 UTC, archived as `_failed/*_hung_eio_2340` (r3250 and r3030), and rerun from scratch (user decision, option A1). Cell 5 started at 06:04 UTC, alongside the reruns instead of after cells 3–4.
- The watchdog logged STALL for both hung runs (~00:00 UTC) but does not stop runs; no action was taken until 06:00 UTC.
- The reruns of cells 3–4 are row-identical to the hung runs up to the hang (dispatch log and accuracy, checked at ~r2000), so the reported runs equal uninterrupted runs.

## 1. Study 2a — Stage 0 plug-in (S2, seed 4126)

| cell | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | s/round | energy to 80% (kJ) | waste (disp / GPU-s / kJ) | Stage 0 rejected | by reason time / energy / both | by tier X / O / T | Stage 0 fallback | share X / O / T | mean \|x_upload − 1000\| (m) | Gini |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2_random | 1813 | 3002 | 7833 | 4.425 | 34.6 | 0.231 / 819 / 37.5 | — | — | — | — | 0.48 / 0.34 / 0.18 | 304 | 0.057 |
| S2_random_s0 | 1324 | 2260 | 6242 | 3.320 | 25.7 | 0.014 / 50 / 2.2 | 0.240 | 0.240 / 0.000 / 0.000 | 0.273 / 0.202 / 0.220 | 0.0000 | 0.46 / 0.35 / 0.19 | 254 | 0.079 |
| S2_wen_fresh | 1145 | 3108 | 10004 | 3.403 | 19.8 | 0.036 / 105 / 5.7 | — | — | — | — | 0.31 / 0.65 / 0.04 | 310 | 0.662 |
| S2_wen_fresh_s0 | 1149 | 2642 | 8387 | 3.302 | 19.1 | 0.002 / 10 / 0.5 | 0.253 | 0.253 / 0.000 / 0.000 | 0.283 / 0.227 / 0.214 | 0.0000 | 0.34 / 0.61 / 0.05 | 304 | 0.637 |
| S2_gpu | 1348 | 2254 | 6931 | 3.573 | 24.7 | 0.123 / 444 / 20.1 | — | — | — | — | 0.46 / 0.34 / 0.20 | 213 | 0.122 |
| S2_gpu_stage0 | 1109 | 2003 | 5279 | 3.069 | 20.3 | 0.008 / 17 / 0.9 | 0.240 | — / — / — | counts 3742 / 1919 / 1005 | 0.0000 | 0.46 / 0.33 / 0.21 | 178 | 0.116 |

S2_gpu_stage0 is the existing run: its Stage 0 is reconstructed from the server log (rejection rate and per-tier rejection counts; reasons were not logged).

## 2. Study 2b — GPU plug-in (seed 4126)

| cell | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | s/round | energy to 80% (kJ) | waste (disp / GPU-s / kJ) | share X / O / T | Thor share | mean \|x_upload − 1000\| (m) | mean \|x̂ − x\| (m) | Gini | Wen cuts per tier X / O / T (T only, E only, both) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_wen_fresh | 1967 | 7263 | — | 3.460 | 28.6 | — | 0.20 / 0.80 / 0.00 | 0.002 | 456 | 514 | 0.819 | X 0.124/0.000/0.000; O 0.133/0.000/0.000; T 0.000/0.598/0.000 |
| S1_wen_fresh_gpu | 1675 | 5046 | — | 2.917 | 16.0 | — | 0.00 / 0.56 / 0.44 | 0.436 | 272 | 1 | 0.787 | X 0.000/0.000/0.198; O 0.000/0.000/0.000; T 0.000/0.000/0.000 |
| S1_gpu | 1158 | 2089 | 6326 | 3.244 | 19.4 | — | 0.46 / 0.33 / 0.21 | 0.209 | 283 | 5 | 0.114 | — |
| S1_random | 1419 | 2568 | 6645 | 3.754 | 23.6 | — | 0.47 / 0.36 / 0.17 | 0.174 | 492 | — | 0.084 | — |
| S2_wen_fresh | 1145 | 3108 | 10004 | 3.403 | 19.8 | 0.036 / 105 / 5.7 | 0.31 / 0.65 / 0.04 | 0.038 | 310 | 497 | 0.662 | X 0.204/0.000/0.000; O 0.200/0.000/0.000; T 0.000/0.635/0.000 |
| S2_wen_fresh_gpu | 1327 | 3205 | — | 3.081 | 13.1 | 0.101 / 134 / 13.0 | 0.00 / 0.56 / 0.44 | 0.435 | 207 | 1 | 0.725 | X 0.000/0.000/0.000; O 0.000/0.000/0.000; T 0.000/0.000/0.000 |
| S2_gpu | 1348 | 2254 | 6931 | 3.573 | 24.7 | 0.123 / 444 / 20.1 | 0.46 / 0.34 / 0.20 | 0.202 | 213 | 5 | 0.122 | — |
| S2_random | 1813 | 3002 | 7833 | 4.425 | 34.6 | 0.231 / 819 / 37.5 | 0.48 / 0.34 / 0.18 | 0.182 | 304 | — | 0.057 | — |

## 3. Cell 5 — Wen 2 × 2 (S2): {CPU, GPU} F1/F2 × {Stage 0 off, on}

| cell | F1/F2 | Stage 0 | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | waste (disp / GPU-s / kJ) | s/round | energy to 80% (kJ) | total energy (kJ) | rounds |
|---|---|---|---|---|---|---|---|---|---|---|
| S2_wen_fresh | CPU | off | 1145 | 3108 | 10004 | 0.036 / 105 / 5.7 | 3.403 | 19.8 | 161.0 | 2940 |
| S2_wen_fresh_s0 | CPU | on | 1149 | 2642 | 8387 | 0.002 / 10 / 0.5 | 3.302 | 19.1 | 139.0 | 2540 |
| S2_wen_fresh_gpu | GPU | off | 1327 | 3205 | — | 0.101 / 134 / 13.0 | 3.081 | 13.1 | 123.0 | 4000 |
| S2_wen_fresh_gpu_s0 | GPU | on | 1256 | 2644 | — | 0.012 / 15 / 1.4 | 2.731 | 13.0 | 112.5 | 4000 |

## 4. GPU limits and realised cuts

T_max = **13.564085609369194 s**, E_max = **113.50862433236982 J** (P90 of the gpu-family F1 / F2 over 900,000 (vehicle, sample) pairs; qualitative check PASS). Per-tier table: `wen_limits_gpu.md` (copied below).


## All positions

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 7.814–14.235 | 112.359–113.643 | 0.0% | 0.0% | 20.0% | 20.0% |
| orin | 5.387–11.808 | 27.757–29.058 | 0.0% | 0.0% | 0.0% | 0.0% |
| thor | 5.215–11.636 | 25.118–26.402 | 0.0% | 0.0% | 0.0% | 0.0% |
| fleet | 5.215–14.235 | 25.118–113.643 | 0.0% | 0.0% | 10.0% | 10.0% |

## Positions inside [400, 1600] at decision time

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 7.814–13.170 | 112.359–113.430 | 0.0% | 0.0% | 0.0% | 0.0% |
| orin | 5.387–10.433 | 27.757–28.782 | 0.0% | 0.0% | 0.0% | 0.0% |
| thor | 5.215–10.240 | 25.118–26.123 | 0.0% | 0.0% | 0.0% | 0.0% |
| fleet | 5.215–13.170 | 25.118–113.430 | 0.0% | 0.0% | 0.0% | 0.0% |


Realised cuts in the runs (share of that tier's candidates; T only / E only / both) and Wen fallback rate:

| cell | Xavier | Orin | Thor | fallback |
|---|---|---|---|---|
| S1_wen_fresh | 0.124/0.000/0.000 | 0.133/0.000/0.000 | 0.000/0.598/0.000 | 0.0000 |
| S1_wen_fresh_gpu | 0.000/0.000/0.198 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.0000 |
| S2_wen_fresh | 0.204/0.000/0.000 | 0.200/0.000/0.000 | 0.000/0.635/0.000 | 0.0000 |
| S2_wen_fresh_s0 | 0.057/0.000/0.000 | 0.089/0.000/0.000 | 0.000/0.539/0.000 | 0.0000 |
| S2_wen_fresh_gpu | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.0000 |
| S2_wen_fresh_gpu_s0 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.0000 |

## 5. P26–P33 (seed 4126 direction; status: single seed — not evaluated)

| P | comparison | seed 4126 | direction | status |
|---|---|---|---|---|
| P26 | random + Stage 0 waste fraction < random (S2) | 0.014 vs 0.231 | as predicted | single seed — not evaluated |
| P27 | random + Stage 0 TTA@80% < random (S2) | 1324 vs 1813 | as predicted | single seed — not evaluated |
| P28 | wen_fresh + Stage 0 waste fraction < wen_fresh (S2) | 0.002 vs 0.036 | as predicted | single seed — not evaluated |
| P29 | wen_fresh + GPU mean \|x̂ − x\| < wen_fresh (S1) | 1 m vs 514 m | as predicted | single seed — not evaluated |
| P30 | wen_fresh + GPU mean \|x_upload − 1000\| < wen_fresh (S1) | 272 m vs 456 m | as predicted | single seed — not evaluated |
| P31 | wen_fresh + GPU s/round < wen_fresh (S1) | 2.917 vs 3.460 | as predicted | single seed — not evaluated |
| P32 | wen_fresh + GPU Thor share > wen_fresh (S1) | 0.436 vs 0.002 | as predicted | single seed — not evaluated |
| P33 | wen_fresh + GPU + Stage 0 waste fraction ≤ wen_fresh + GPU (S2) | 0.012 vs 0.101 | as predicted | single seed — not evaluated |

## 6. Curves

`study2_plugin_s4126_curves.png`: accuracy vs online clock, one panel per comparison group (2a random, 2a wen_fresh, 2b S1, 2b S2, cell 5).


## 7. Step 0 results

### 0a — eligibility in every existing S2 run

Per run: dispatches, dispatches with x_dispatch outside [400, 1600], dispatches to a vehicle still in flight (dispatched before its previous task's t_finish_clean).

| run | dispatches | outside coverage | not idle |
|---|---|---|---|
| S2_cpu_stage0_s4126 | 1901 | 0 | 0 |
| S2_cpu_stage0_s4126_smoke2 | 203 | 0 | 0 |
| S2_cpu_stage0_s4127 | 2068 | 0 | 0 |
| S2_cpu_stage0_s4128 | 2320 | 0 | 0 |
| S2_gpu_s4126 | 2214 | 0 | 0 |
| S2_gpu_s4126_smoke | 250 | 0 | 0 |
| S2_gpu_s4127 | 2327 | 0 | 0 |
| S2_gpu_s4128 | 2064 | 0 | 0 |
| S2_gpu_stage0_s4126 | 1736 | 0 | 0 |
| S2_gpu_stage0_s4126_dbg | 12 | 0 | 0 |
| S2_gpu_stage0_s4126_replay200 | 204 | 0 | 0 |
| S2_gpu_stage0_s4126_smoke | 204 | 0 | 0 |
| S2_gpu_stage0_s4126_smoke2 | 204 | 0 | 0 |
| S2_gpu_stage0_s4127 | 2499 | 0 | 0 |
| S2_gpu_stage0_s4128 | 2312 | 0 | 0 |
| S2_random_s4126 | 2305 | 0 | 0 |
| S2_random_s4127 | 2399 | 0 | 0 |
| S2_random_s4128 | 2070 | 0 | 0 |
| S2_wen_fresh_s4126 | 3051 | 0 | 0 |
| S2_wen_fresh_s4126_smoke | 205 | 0 | 0 |
| S2_wen_s4126 | 3403 | 0 | 0 |
| S2_wen_s4126_smoke | 206 | 0 | 0 |
| S2_wen_s4127 | 4203 | 0 | 0 |
| S2_wen_s4128 | 3860 | 0 | 0 |

### 0b.2 — 200-round replay with the current code (batch B)

| arm family | replay run | source commit (dirty) | dispatch rows (r ≤ 200) | identical to the existing run |
|---|---|---|---|---|
| S1_gpu | S1_gpu_s4126_replay200 | 4fd265dd (false) | 202 | yes |
| S1_cpu | S1_cpu_s4126_replay200 | 4fd265dd (false) | 202 | yes |
| S2_gpu_stage0 | S2_gpu_stage0_s4126_replay200 | 4fd265dd (false) | 204 | yes |
| S1_cpu_cal | S1_cpu_cal_s4126_replay200 | 4fd265dd (false) | 202 | yes |
| S1_wen | S1_wen_s4126_replay200 | 4fd265dd (false) | 202 | yes |
| S1_wen_fresh | S1_wen_fresh_s4126_replay200 | 4fd265dd (false) | 202 | yes |

### 0c — Stage 0 in S1 (S1 gpu + Stage 0, 200 rounds)

- Stage 0 calls: 200; vehicles checked: 5602; rejected: 0; fallbacks: 0.
- Dispatch log (rounds ≤ 200, 202 rows) identical to S1 gpu (Stage 0 off): **yes**.

## Step 4 — smoke (200 rounds, seed 4126; accuracy not read)

| cell | s/round | share X / O / T | Stage 0 rejected (of eligible) | by reason time / energy / both | by tier X / O / T (rejected / eligible) | Stage 0 fallback | Wen fallback | waste (S2) | distinct vehicles | mean \|x̂ − x\| (m) | mean \|x_upload − 1000\| (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S2_random_s0 | 3.292 | 0.43 / 0.35 / 0.22 | 0.243 | 0.243 / 0.000 / 0.000 | 0.274 / 0.217 / 0.200 | 0.0000 | — | 0.010 | 30 | — | 249 |
| S2_wen_fresh_s0 | 3.458 | 0.40 / 0.54 / 0.06 | 0.256 | 0.256 / 0.000 / 0.000 | 0.284 / 0.229 / 0.226 | 0.0000 | 0.0000 | 0.000 | 25 | 479 | 325 |
| S1_wen_fresh_gpu | 2.963 | 0.00 / 0.65 / 0.35 | — | — | — | — | 0.0000 | — | 12 | 1 | 273 |
| S2_wen_fresh_gpu | 3.249 | 0.00 / 0.63 / 0.37 | — | — | — | — | 0.0000 | 0.129 | 13 | 1 | 211 |
| S2_wen_fresh_gpu_s0 | 2.781 | 0.00 / 0.67 / 0.33 | 0.238 | 0.238 / 0.000 / 0.000 | 0.259 / 0.205 / 0.223 | 0.0000 | 0.0000 | 0.005 | 16 | 1 | 186 |

Constraint cuts per tier (share of that tier's candidates, Wen hosts):

| cell | tier | T only | E only | both |
|---|---|---|---|---|
| S2_wen_fresh_s0 | xavier | 0.065 | 0.000 | 0.000 |
| S2_wen_fresh_s0 | orin | 0.083 | 0.000 | 0.000 |
| S2_wen_fresh_s0 | thor | 0.000 | 0.536 | 0.000 |
| S1_wen_fresh_gpu | xavier | 0.000 | 0.000 | 0.198 |
| S1_wen_fresh_gpu | orin | 0.000 | 0.000 | 0.000 |
| S1_wen_fresh_gpu | thor | 0.000 | 0.000 | 0.000 |
| S2_wen_fresh_gpu | xavier | 0.000 | 0.000 | 0.000 |
| S2_wen_fresh_gpu | orin | 0.000 | 0.000 | 0.000 |
| S2_wen_fresh_gpu | thor | 0.000 | 0.000 | 0.000 |
| S2_wen_fresh_gpu_s0 | xavier | 0.000 | 0.000 | 0.000 |
| S2_wen_fresh_gpu_s0 | orin | 0.000 | 0.000 | 0.000 |
| S2_wen_fresh_gpu_s0 | thor | 0.000 | 0.000 | 0.000 |
