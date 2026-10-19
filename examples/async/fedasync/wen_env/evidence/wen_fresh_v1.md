# wen_fresh v1: Wen-faithful with a fresh F3 (Prompt H) — interim

**Status (19 Oct 2026, 04:30 UTC).**
- **Scope.** Seed 4126 only (S1 and S2). On 18 Oct, 11:00 UTC, the user asked to run one seed first; seeds 4127 / 4128 have not been started.
- **Predictions.** v3.3 is committed as `77e2d7b7bc2439b9f36b160678b9b70861224dfc` and recorded in both run manifests.
- **8 h rule.** The projected extra time was over 8 h per run. I reported it and waited, and the user accepted it on 18 Oct.
- **Stop conditions.**
  - Clock replay: online = replayed in both runs (max |Δ| 0.0 s, 0 merge-order residuals).
  - No other stop condition was hit.
- **Run ends.**
  - S1 ran the full 4,000 rounds without reaching 90%.
  - S2 stopped at 90% (0.9009) at round 2,940.
- **P22–P25 below are one-seed directions, not v3 verdicts.** The v3 rule needs the same direction in all 3 seeds.

Contents: § 0 Step 0; § 1–5 Step 7 tables (seed 4126); § 6 Step 6; § 7 extra wall time. Appendices: A implementation and ADR, B tests, C smoke, D per-vehicle tables.

## 0. The `wen` F3 prior (Step 0)

| run | accepted reports recovered | unrecovered | never-trained candidate rows checked | rows where logged F3 ≠ replayed prior |
|---|---|---|---|---|
| S1_wen_s4126 | 3999 | 0 | 9245 | 0 |
| S1_wen_s4127 | 3999 | 0 | 19249 | 0 |
| S1_wen_s4128 | 3999 | 0 | 15217 | 0 |
| S2_wen_s4126 | 3249 | 0 | 4215 | 0 |
| S2_wen_s4127 | 3999 | 0 | 13152 | 0 |
| S2_wen_s4128 | 3579 | 0 | 9180 | 0 |

Per-vehicle example, `S1_wen_s4127`:

| client | scenario | candidate rows | mean F3 as a candidate | dispatches | first dispatch rounds | its own reported losses |
|---|---|---|---|---|---|---|
| 26 | S1 | 4002 | 0.3395 | 0 | [] | [] |
| 30 | S1 | 4001 | 1.0676 | 1 | [477] | [1.0746] |
| 20 | S1 | 3998 | 1.0587 | 2 | [519, 2661] | [1.1534, 0.9362] |
| 8 | S1 | 3993 | 0.7593 | 3 | [123, 736, 1057] | [0.8874, 0.6823, 0.712] |

Prior over the run, `S1_wen_s4127` (decision → round, losses averaged, prior):

| decision | round | losses averaged | prior |
|---|---|---|---|
| 10 | 8 | 7 | 1.8587 |
| 50 | 48 | 47 | 1.5171 |
| 100 | 98 | 97 | 1.2872 |
| 500 | 498 | 497 | 0.6163 |
| 1000 | 998 | 997 | 0.3776 |
| 2000 | 1998 | 1997 | 0.2200 |
| 3000 | 2998 | 2997 | 0.1633 |
| 4002 | 4000 | 3999 | 0.1302 |

### Step 0 answers

1. **Definition** (code used by the 6 `wen` runs: `wen_env/wen_faithful.py`, identical to the audit-bundle copy).
   - The prior is the mean over **all** reports in history, including repeated reports by the same vehicle, not just each vehicle's latest report.
   - `on_reports_received` (`wen_faithful.py:135–141`) adds every delivered report's loss to `_loss_sum` and increments `_loss_n`.
   - `f3` (`wen_faithful.py:143–146`) returns the vehicle's own `last_loss` if it has reported. Otherwise it returns `_loss_sum / _loss_n`, or 0.0 before any report.
   - Lost S2 uploads are excluded (`delivered`, line 136).
2. **Recomputation.**
   - `f3` evaluates `_loss_sum / _loss_n` at every call, i.e. for every candidate of every decision (`objectives`, line 155).
   - The value changes only when a report is received, so in effect it is updated once per merge.
   - The replay above reproduces the logged F3 of every never-trained candidate exactly: 0 mismatches in 70,258 rows across the 6 runs.
3. **Why never-trained Thor shows ≈ 0.34 while vehicles picked once early stay at 0.79–0.95** (S1 seed 4127).
   - The prior is a running mean of all losses. It falls as the model trains: 1.86 at round 8, 0.62 at round 498, 0.38 at round 998, 0.13 at round 4,000.
   - A never-trained vehicle is a candidate at every decision with F3 = the current prior. Client 26 therefore averages **0.3395** over its 4,002 candidate rows, the mean of that falling curve.
   - A vehicle picked once keeps its own last reported loss until it is picked again. Client 30 reported **1.0746** at round 477 and was never picked again, so its F3 stays at 1.0746 for ≈ 3,500 rounds (mean 1.0676). Client 8 reported 0.8874 / 0.6823 / 0.712 at rounds 123 / 736 / 1,057 (mean 0.7593).
   - The 0.340 in the Prompt F per-vehicle table is reproduced as 0.3395, the mean of client 26's logged F3 rows, which equal the replayed prior bit for bit.

## 1. Study table — S1 (seeds 4126; mean ± SD when > 1)

| arm | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | s/round | s/round before / after 80% | energy to 80% (kJ) | share X/O/T rounds 1–500 | share X/O/T last 500 | waste (disp / GPU-s / kJ) | mean \|x_upload − 1000\| (m) | mean \|x̂ − x\| (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_random | 1419 | 2568 | 6645 | 3.754 | 3.734 / 3.760 | 23.6 | 0.43 / 0.39 / 0.18 | 0.48 / 0.35 / 0.18 | — | 492 | — |
| S1_gpu | 1158 | 2089 | 6326 | 3.244 | 2.969 / 3.313 | 19.4 | 0.32 / 0.38 / 0.30 | 0.52 / 0.31 / 0.18 | — | 283 | 5 |
| S1_wen | 2625 | 11049 | — (1/1 not reached; last 0.824) | 3.460 | 3.454 / 3.462 | 33.0 | 0.20 / 0.80 / 0.00 | 0.18 / 0.82 / 0.00 | — | 465 | 515 |
| S1_wen_fresh | 1967 | 7263 | — (1/1 not reached; last 0.850) | 3.460 | 3.513 / 3.451 | 28.6 | 0.31 / 0.67 / 0.02 | 0.18 / 0.82 / 0.00 | — | 456 | 514 |
| S1_location_only | 1326 | 2547 | 5723 | 3.347 | 3.235 / 3.382 | 29.6 | 0.55 / 0.29 / 0.16 | 0.55 / 0.27 / 0.18 | — | 279 | 31 |
| S1_cpu | 1535 | 2837 | 6866 | 3.724 | 3.743 / 3.719 | 26.6 | 0.49 / 0.25 / 0.26 | 0.52 / 0.32 / 0.16 | — | 462 | 557 |

S1 gpu / cpu at seed 4126 count as reaching 90% at their maximum accuracy (Reading B, `wen_faithful_v1_step6_footnote.md`).

## 1. Study table — S2 (Stage 0 off for every arm)

| arm | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | s/round | s/round before / after 80% | energy to 80% (kJ) | share X/O/T rounds 1–500 | share X/O/T last 500 | waste (disp / GPU-s / kJ) | mean \|x_upload − 1000\| (m) | mean \|x̂ − x\| (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S2_random | 1813 | 3002 | 7833 | 4.425 | 4.422 / 4.427 | 34.6 | 0.46 / 0.35 / 0.19 | 0.49 / 0.33 / 0.18 | 0.231 / 819 / 37.5 | 304 | — |
| S2_gpu | 1348 | 2254 | 6931 | 3.573 | 3.211 / 3.673 | 24.7 | 0.34 / 0.42 / 0.24 | 0.48 / 0.31 / 0.20 | 0.123 / 444 / 20.1 | 213 | 5 |
| S2_wen | 1412 | 3110 | 11262 | 3.465 | 3.362 / 3.481 | 23.5 | 0.34 / 0.66 / 0.00 | 0.31 / 0.69 / 0.00 | 0.045 / 174 / 8.8 | 311 | 505 |
| S2_wen_fresh | 1145 | 3108 | 10004 | 3.403 | 3.368 / 3.407 | 19.8 | 0.37 / 0.57 / 0.05 | 0.29 / 0.68 / 0.03 | 0.036 / 105 / 5.7 | 310 | 497 |

## 2. Concentration (wen_fresh next to wen, random, gpu)

Replay check: all 7,053 logged wen_fresh decisions are reproduced by `wen_faithful.decide` on the logged candidates and F values.

### Whole-run dispatch Gini (per seed) and per-vehicle extremes

| scenario | arm | seed 4126 | mean | min / max dispatches (mean) | vehicles with 0 (mean) |
|---|---|---|---|---|---|
| S1 | wen_fresh | 0.819 | 0.819 | 0.0 / 984.0 | 6.0 |
| S1 | wen | 0.834 | 0.834 | 1.0 / 1036.0 | 0.0 |
| S1 | random | 0.084 | 0.084 | 36.0 / 75.0 | 0.0 |
| S1 | gpu | 0.114 | 0.114 | 39.0 / 109.0 | 0.0 |
| S2 | wen_fresh | 0.662 | 0.662 | 0.0 / 477.0 | 3.0 |
| S2 | wen | 0.679 | 0.679 | 1.0 / 530.0 | 0.0 |
| S2 | random | 0.057 | 0.057 | 58.0 / 93.0 | 0.0 |
| S2 | gpu | 0.122 | 0.122 | 41.0 / 136.0 | 0.0 |

### Onset (final top vehicle's cumulative share; first / sustained round)

| scenario | arm | seed | top share | >10% | >20% |
|---|---|---|---|---|---|
| S1 | wen_fresh | 4126 | 0.246 | 12 / 12 | 21 / 295 |
| S1 | wen | 4126 | 0.259 | 7 / 12 | 21 / 21 |
| S1 | random | 4126 | 0.042 | None / — | — / — |
| S1 | gpu | 4126 | 0.056 | 36 / — | — / — |
| S2 | wen_fresh | 4126 | 0.156 | 1 / 1 | 1 / — |
| S2 | wen | 4126 | 0.156 | 1 / 1 | 1 / — |
| S2 | random | 4126 | 0.040 | 13 / — | — / — |
| S2 | gpu | 4126 | 0.061 | None / — | — / — |

### Rolling Gini (250-round windows; mean over 1 seed(s))

| scenario | arm | r≤250 | r≤500 | r≤750 | r≤1000 | r≤1250 | r≤1500 | r≤1750 | r≤2000 | r≤2250 | r≤2500 | r≤2750 | r≤3000 | r≤3250 | r≤3500 | r≤3750 | r≤4000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | wen_fresh | 0.67 | 0.83 | 0.84 | 0.84 | 0.84 | 0.84 | 0.86 | 0.85 | 0.84 | 0.84 | 0.82 | 0.83 | 0.84 | 0.84 | 0.83 | 0.84 |
| S1 | wen | 0.82 | 0.85 | 0.85 | 0.86 | 0.86 | 0.84 | 0.85 | 0.84 | 0.85 | 0.84 | 0.84 | 0.84 | 0.84 | 0.85 | 0.84 | 0.85 |
| S1 | random | 0.19 | 0.19 | 0.20 | 0.18 | 0.19 | 0.18 | 0.16 | 0.56 | — | — | — | — | — | — | — | — |
| S1 | gpu | 0.42 | 0.30 | 0.22 | 0.26 | 0.30 | 0.26 | 0.25 | 0.27 | — | — | — | — | — | — | — | — |
| S2 | wen_fresh | 0.58 | 0.63 | 0.69 | 0.70 | 0.71 | 0.69 | 0.69 | 0.69 | 0.68 | 0.70 | 0.70 | 0.68 | — | — | — | — |
| S2 | wen | 0.65 | 0.67 | 0.70 | 0.70 | 0.70 | 0.68 | 0.70 | 0.72 | 0.70 | 0.69 | 0.69 | 0.70 | 0.71 | — | — | — |
| S2 | random | 0.12 | 0.16 | 0.15 | 0.16 | 0.16 | 0.15 | 0.10 | 0.49 | — | — | — | — | — | — | — | — |
| S2 | gpu | 0.37 | 0.32 | 0.27 | 0.25 | 0.27 | 0.24 | 0.24 | 0.31 | — | — | — | — | — | — | — | — |

### Spearman correlation of dispatch count with … (wen_fresh, per seed)

| scenario | seed | F4 rank | mean F3 | mean F1 | mean F2 | V_i | tier |
|---|---|---|---|---|---|---|---|
| S1 | 4126 | -0.72 | -0.81 | +0.85 | +0.17 | -0.14 | -0.23 |
| S2 | 4126 | -0.59 | -0.87 | +0.04 | -0.53 | -0.57 | -0.17 |

### F3 over time, wen_fresh vs wen: top-5 vehicles vs the rest (mean candidate F3 per 100-round window, mean over seeds; every 500 rounds)

| scenario | round ≤ | wen_fresh top-5 | wen_fresh rest | wen top-5 | wen rest |
|---|---|---|---|---|---|
| S1 | 500 | 0.387 | 0.882 | 0.431 | 1.056 |
| S1 | 1000 | 0.076 | 0.703 | 0.176 | 0.996 |
| S1 | 1500 | 0.045 | 0.607 | 0.021 | 1.020 |
| S1 | 2000 | 0.051 | 0.576 | 0.019 | 1.053 |
| S1 | 2500 | 0.041 | 0.493 | 0.036 | 0.983 |
| S1 | 3000 | 0.012 | 0.443 | 0.005 | 0.880 |
| S1 | 3500 | 0.041 | 0.465 | 0.006 | 0.867 |
| S1 | 4000 | 0.042 | 0.461 | 0.008 | 0.866 |
| S2 | 500 | 0.372 | 0.648 | 0.283 | 0.764 |
| S2 | 1000 | 0.130 | 0.477 | 0.064 | 0.656 |
| S2 | 1500 | 0.057 | 0.357 | 0.030 | 0.586 |
| S2 | 2000 | 0.042 | 0.302 | 0.013 | 0.540 |
| S2 | 2500 | 0.039 | 0.280 | 0.013 | 0.508 |
| S2 | 3000 | 0.014 | 0.211 | 0.015 | 0.484 |
| S2 | 3500 | — | — | — | — |
| S2 | 4000 | — | — | — | — |

### Label coverage of the aggregated (accepted, merged) updates

Same definitions as Prompt F Step 4.

| scenario | arm | seed | r250: KL / coverage / vehicles | r500: KL / coverage / vehicles | r1000: KL / coverage / vehicles | r2000: KL / coverage / vehicles | r4000: KL / coverage / vehicles |
|---|---|---|---|---|---|---|---|
| S1 | wen_fresh | 4126 | 0.0252 / 0.733 / 22 | 0.0248 / 0.733 / 22 | 0.0184 / 0.767 / 23 | 0.0156 / 0.800 / 24 | 0.0152 / 0.800 / 24 |
| S1 | wen | 4126 | 0.0251 / 0.667 / 20 | 0.0245 / 0.833 / 25 | 0.0225 / 0.900 / 27 | 0.0179 / 1.000 / 30 | 0.0155 / 1.000 / 30 |
| S1 | random | 4126 | 0.0026 / 1.000 / 30 | 0.0012 / 1.000 / 30 | 0.0004 / 1.000 / 30 | 0.0004 / 1.000 / 30 (r1770) | 0.0004 / 1.000 / 30 (r1770) |
| S1 | gpu | 4126 | 0.0041 / 1.000 / 30 | 0.0024 / 1.000 / 30 | 0.0020 / 1.000 / 30 | 0.0010 / 1.000 / 30 (r1957) | 0.0010 / 1.000 / 30 (r1957) |
| S2 | wen_fresh | 4126 | 0.0185 / 0.833 / 25 | 0.0193 / 0.833 / 25 | 0.0218 / 0.833 / 25 | 0.0201 / 0.900 / 27 | 0.0203 / 0.900 / 27 (r2940) |
| S2 | wen | 4126 | 0.0304 / 0.767 / 23 | 0.0235 / 0.833 / 25 | 0.0243 / 0.900 / 27 | 0.0224 / 1.000 / 30 | 0.0221 / 1.000 / 30 (r3250) |
| S2 | random | 4126 | 0.0006 / 1.000 / 30 | 0.0005 / 1.000 / 30 | 0.0002 / 1.000 / 30 | 0.0002 / 1.000 / 30 (r1770) | 0.0002 / 1.000 / 30 (r1770) |
| S2 | gpu | 4126 | 0.0027 / 1.000 / 30 | 0.0025 / 1.000 / 30 | 0.0017 / 1.000 / 30 | 0.0011 / 1.000 / 30 (r1940) | 0.0011 / 1.000 / 30 (r1940) |

### Per-vehicle table — wen_fresh, S1, seed 4126

| client | tier | \|S\| | V (m/s) | F4 | F4 rank | dispatches | share | rank | mean F1 | mean F2 | mean F3 | pick nF1 | nF2 | nF3 | nF4 | random | gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | orin | 1667 | 21.0 | 0.1774 | 2 | 984 | 0.246 | 1 | 33.06 | 26.74 | 0.153 | 0.90 | 0.07 | 0.02 | 0.01 | 60 | 79 |
| 19 | orin | 1667 | 30.6 | 0.1664 | 1 | 919 | 0.230 | 2 | 33.10 | 26.75 | 0.215 | 0.88 | 0.07 | 0.04 | 0.00 | 61 | 72 |
| 22 | orin | 1666 | 17.6 | 0.2351 | 4 | 725 | 0.181 | 3 | 33.04 | 26.73 | 0.144 | 0.74 | 0.07 | 0.01 | 0.10 | 66 | 57 |
| 5 | xavier | 1667 | 17.1 | 0.2338 | 3 | 637 | 0.159 | 4 | 33.15 | 26.76 | 0.125 | 0.81 | 0.07 | 0.01 | 0.10 | 51 | 55 |
| 21 | orin | 1666 | 25.7 | 0.2548 | 5 | 448 | 0.112 | 5 | 32.99 | 26.72 | 0.201 | 0.64 | 0.06 | 0.05 | 0.18 | 59 | 69 |
| 20 | orin | 1667 | 17.7 | 0.3295 | 8 | 72 | 0.018 | 6 | 32.68 | 26.67 | 0.398 | 0.43 | 0.05 | 0.19 | 0.44 | 68 | 65 |
| 23 | orin | 1666 | 31.7 | 0.4204 | 19 | 32 | 0.008 | 7 | 32.60 | 26.64 | 0.377 | 0.23 | 0.03 | 0.30 | 0.79 | 68 | 58 |
| 13 | xavier | 1667 | 32.4 | 0.3039 | 7 | 28 | 0.007 | 8 | 32.62 | 26.65 | 0.642 | 0.27 | 0.05 | 0.66 | 0.35 | 49 | 58 |
| 15 | xavier | 1667 | 28.3 | 0.3611 | 11 | 21 | 0.005 | 9 | 32.60 | 26.65 | 0.473 | 0.23 | 0.03 | 0.46 | 0.56 | 63 | 63 |
| 6 | xavier | 1667 | 30.0 | 0.3500 | 10 | 17 | 0.004 | 10 | 32.60 | 26.65 | 0.533 | 0.19 | 0.04 | 0.60 | 0.45 | 46 | 65 |
| 1 | xavier | 1667 | 21.8 | 0.5206 | 26 | 14 | 0.003 | 11 | 32.61 | 26.65 | 0.438 | 0.03 | 0.01 | 0.47 | 0.97 | 70 | 39 |
| 14 | xavier | 1667 | 24.7 | 0.3825 | 12 | 13 | 0.003 | 12 | 32.60 | 26.65 | 0.649 | 0.21 | 0.07 | 0.60 | 0.55 | 50 | 58 |
| 18 | orin | 1667 | 25.6 | 0.3968 | 15 | 13 | 0.003 | 13 | 32.58 | 26.65 | 0.646 | 0.22 | 0.04 | 0.60 | 0.58 | 65 | 62 |
| 4 | xavier | 1667 | 24.0 | 0.4279 | 22 | 12 | 0.003 | 14 | 32.60 | 26.65 | 0.510 | 0.02 | 0.02 | 0.58 | 0.79 | 50 | 49 |
| 10 | xavier | 1667 | 31.9 | 0.3453 | 9 | 12 | 0.003 | 15 | 32.60 | 26.65 | 0.764 | 0.18 | 0.04 | 0.80 | 0.42 | 55 | 81 |
| 26 | thor | 1666 | 21.5 | 0.2553 | 6 | 10 | 0.002 | 16 | 28.99 | 36.46 | 0.532 | 0.00 | 1.00 | 0.18 | 0.17 | 68 | 60 |
| 8 | xavier | 1667 | 32.0 | 0.4229 | 20 | 9 | 0.002 | 17 | 32.60 | 26.65 | 0.564 | 0.16 | 0.04 | 0.51 | 0.66 | 62 | 52 |
| 3 | xavier | 1667 | 25.1 | 0.4802 | 24 | 8 | 0.002 | 18 | 32.59 | 26.65 | 0.511 | 0.10 | 0.02 | 0.49 | 0.82 | 36 | 52 |
| 9 | xavier | 1667 | 30.6 | 0.4267 | 21 | 8 | 0.002 | 19 | 32.52 | 26.64 | 0.581 | 0.11 | 0.06 | 0.65 | 0.69 | 75 | 63 |
| 11 | xavier | 1667 | 32.6 | 0.3846 | 13 | 8 | 0.002 | 20 | 32.60 | 26.65 | 0.693 | 0.00 | 0.00 | 0.92 | 0.69 | 63 | 70 |
| 12 | xavier | 1667 | 26.9 | 0.4418 | 23 | 5 | 0.001 | 21 | 32.59 | 26.65 | 0.699 | 0.08 | 0.00 | 0.82 | 0.77 | 48 | 46 |
| 2 | xavier | 1667 | 27.0 | 0.5375 | 28 | 3 | 0.001 | 22 | 32.59 | 26.65 | 0.558 | 0.07 | 0.07 | 0.37 | 1.00 | 63 | 75 |
| 7 | xavier | 1667 | 17.4 | 0.5297 | 27 | 3 | 0.001 | 23 | 32.61 | 26.65 | 0.917 | 0.00 | 0.00 | 1.00 | 0.81 | 47 | 72 |
| 17 | orin | 1667 | 23.0 | 0.6273 | 30 | 1 | 0.000 | 24 | 32.60 | 26.65 | 0.483 | 0.21 | 0.00 | 0.00 | 1.00 | 62 | 65 |
| 24 | orin | 1666 | — | 0.5790 | 29 | 0 | 0.000 | 25 | 32.58 | 26.64 | 0.734 | — | — | — | — | 61 | 55 |
| 25 | orin | 1666 | — | 0.5120 | 25 | 0 | 0.000 | 26 | 32.58 | 26.63 | 0.867 | — | — | — | — | 65 | 69 |
| 27 | thor | 1666 | — | 0.4022 | 17 | 0 | 0.000 | 27 | 28.98 | 36.46 | 0.691 | — | — | — | — | 70 | 65 |
| 28 | thor | 1666 | — | 0.3995 | 16 | 0 | 0.000 | 28 | 28.98 | 36.46 | 0.827 | — | — | — | — | 65 | 89 |
| 29 | thor | 1666 | — | 0.4123 | 18 | 0 | 0.000 | 29 | 28.98 | 36.46 | 0.705 | — | — | — | — | 50 | 87 |
| 30 | thor | 1666 | — | 0.3887 | 14 | 0 | 0.000 | 30 | 28.98 | 36.46 | 0.811 | — | — | — | — | 56 | 109 |

### Per-vehicle table — wen_fresh, S2, seed 4126

| client | tier | \|S\| | V (m/s) | F4 | F4 rank | dispatches | share | rank | mean F1 | mean F2 | mean F3 | pick nF1 | nF2 | nF3 | nF4 | random | gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | orin | 1667 | 21.0 | 0.1774 | 2 | 477 | 0.156 | 1 | 32.89 | 26.71 | 0.204 | 0.76 | 0.10 | 0.09 | 0.02 | 76 | 73 |
| 22 | orin | 1666 | 17.6 | 0.2351 | 4 | 477 | 0.156 | 2 | 32.56 | 26.63 | 0.139 | 0.74 | 0.09 | 0.05 | 0.10 | 80 | 70 |
| 5 | xavier | 1667 | 17.1 | 0.2338 | 3 | 406 | 0.133 | 3 | 32.59 | 26.65 | 0.162 | 0.74 | 0.11 | 0.06 | 0.10 | 65 | 58 |
| 19 | orin | 1667 | 30.6 | 0.1664 | 1 | 343 | 0.112 | 4 | 33.44 | 26.82 | 0.326 | 0.83 | 0.11 | 0.15 | 0.00 | 73 | 92 |
| 21 | orin | 1666 | 25.7 | 0.2548 | 5 | 329 | 0.108 | 5 | 33.15 | 26.75 | 0.211 | 0.72 | 0.08 | 0.10 | 0.13 | 76 | 79 |
| 20 | orin | 1667 | 17.7 | 0.3295 | 8 | 221 | 0.072 | 6 | 32.60 | 26.65 | 0.243 | 0.49 | 0.09 | 0.10 | 0.39 | 66 | 76 |
| 26 | thor | 1666 | 21.5 | 0.2553 | 6 | 115 | 0.038 | 7 | 28.80 | 36.42 | 0.244 | 0.04 | 1.00 | 0.10 | 0.02 | 89 | 54 |
| 15 | xavier | 1667 | 28.3 | 0.3611 | 11 | 87 | 0.029 | 8 | 33.24 | 26.78 | 0.263 | 0.52 | 0.08 | 0.19 | 0.36 | 86 | 57 |
| 14 | xavier | 1667 | 24.7 | 0.3825 | 12 | 78 | 0.026 | 9 | 32.87 | 26.71 | 0.390 | 0.33 | 0.04 | 0.48 | 0.47 | 58 | 78 |
| 4 | xavier | 1667 | 24.0 | 0.4279 | 22 | 64 | 0.021 | 10 | 32.86 | 26.70 | 0.311 | 0.18 | 0.03 | 0.43 | 0.72 | 78 | 62 |
| 13 | xavier | 1667 | 32.4 | 0.3039 | 7 | 59 | 0.019 | 11 | 33.42 | 26.81 | 0.429 | 0.64 | 0.11 | 0.36 | 0.09 | 69 | 73 |
| 1 | xavier | 1667 | 21.8 | 0.5206 | 26 | 57 | 0.019 | 12 | 32.65 | 26.66 | 0.293 | 0.09 | 0.02 | 0.48 | 0.95 | 72 | 41 |
| 18 | orin | 1667 | 25.6 | 0.3968 | 15 | 52 | 0.017 | 13 | 32.93 | 26.72 | 0.439 | 0.21 | 0.06 | 0.55 | 0.57 | 83 | 71 |
| 6 | xavier | 1667 | 30.0 | 0.3500 | 10 | 39 | 0.013 | 14 | 33.28 | 26.79 | 0.392 | 0.46 | 0.08 | 0.51 | 0.26 | 69 | 95 |
| 17 | orin | 1667 | 23.0 | 0.6273 | 30 | 39 | 0.013 | 15 | 32.69 | 26.67 | 0.317 | 0.07 | 0.01 | 0.34 | 1.00 | 78 | 68 |
| 3 | xavier | 1667 | 25.1 | 0.4802 | 24 | 33 | 0.011 | 16 | 32.89 | 26.71 | 0.362 | 0.04 | 0.00 | 0.55 | 0.82 | 72 | 75 |
| 7 | xavier | 1667 | 17.4 | 0.5297 | 27 | 31 | 0.010 | 17 | 32.29 | 26.59 | 0.631 | 0.01 | 0.01 | 0.85 | 0.93 | 76 | 88 |
| 2 | xavier | 1667 | 27.0 | 0.5375 | 28 | 30 | 0.010 | 18 | 33.06 | 26.74 | 0.362 | 0.08 | 0.03 | 0.42 | 0.94 | 82 | 70 |
| 9 | xavier | 1667 | 30.6 | 0.4267 | 21 | 26 | 0.009 | 19 | 33.31 | 26.79 | 0.383 | 0.34 | 0.07 | 0.43 | 0.63 | 76 | 64 |
| 23 | orin | 1666 | 31.7 | 0.4204 | 19 | 24 | 0.008 | 20 | 33.37 | 26.79 | 0.348 | 0.28 | 0.08 | 0.43 | 0.63 | 76 | 68 |
| 10 | xavier | 1667 | 31.9 | 0.3453 | 9 | 17 | 0.006 | 21 | 33.40 | 26.81 | 0.621 | 0.22 | 0.06 | 0.86 | 0.32 | 81 | 77 |
| 25 | orin | 1666 | 24.2 | 0.5120 | 25 | 15 | 0.005 | 22 | 32.76 | 26.67 | 0.643 | 0.01 | 0.01 | 0.77 | 0.89 | 79 | 77 |
| 12 | xavier | 1667 | 26.9 | 0.4418 | 23 | 13 | 0.004 | 23 | 33.03 | 26.74 | 0.582 | 0.08 | 0.01 | 0.71 | 0.72 | 78 | 46 |
| 8 | xavier | 1667 | 32.0 | 0.4229 | 20 | 9 | 0.003 | 24 | 33.37 | 26.80 | 0.455 | 0.36 | 0.11 | 0.23 | 0.58 | 73 | 56 |
| 11 | xavier | 1667 | 32.6 | 0.3846 | 13 | 8 | 0.003 | 25 | 33.40 | 26.81 | 0.620 | 0.12 | 0.04 | 0.81 | 0.63 | 72 | 81 |
| 24 | orin | 1666 | 31.5 | 0.5790 | 29 | 1 | 0.000 | 26 | 33.34 | 26.79 | 0.600 | 0.00 | 0.00 | 0.83 | 1.00 | 91 | 72 |
| 27 | thor | 1666 | 26.6 | 0.4022 | 17 | 1 | 0.000 | 27 | 28.97 | 36.45 | 0.559 | 0.00 | 1.00 | 0.43 | 0.00 | 93 | 74 |
| 28 | thor | 1666 | — | 0.3995 | 16 | 0 | 0.000 | 28 | 29.45 | 36.55 | 0.707 | — | — | — | — | 75 | 82 |
| 29 | thor | 1666 | — | 0.4123 | 18 | 0 | 0.000 | 29 | 29.10 | 36.48 | 0.622 | — | — | — | — | 72 | 101 |
| 30 | thor | 1666 | — | 0.3887 | 14 | 0 | 0.000 | 30 | 29.38 | 36.54 | 0.683 | — | — | — | — | 91 | 136 |

## 3. Constraint table (wen_fresh; all decisions per scenario, 1 seed(s) pooled)

| scenario | tier | cut by T only | cut by E only | cut by both | fallback rate | mean n_feasible | mean n_pareto |
|---|---|---|---|---|---|---|---|
| S1 | xavier | 0.124 | 0.000 | 0.000 | 0.0000 | 22.09 | 7.23 |
| S1 | orin | 0.133 | 0.000 | 0.000 | 0.0000 | 22.09 | 7.23 |
| S1 | thor | 0.000 | 0.598 | 0.000 | 0.0000 | 22.09 | 7.23 |
| S2 | xavier | 0.204 | 0.000 | 0.000 | 0.0000 | 11.56 | 5.00 |
| S2 | orin | 0.200 | 0.000 | 0.000 | 0.0000 | 11.56 | 5.00 |
| S2 | thor | 0.000 | 0.635 | 0.000 | 0.0000 | 11.56 | 5.00 |

## 4. P22–P25 (v3 rule: same direction in all 3 seeds; no re-wording)

| P | scenario | comparison | seed 4126 | all 1 seed(s) |
|---|---|---|---|---|
| P22 | S1 | wen_fresh Gini < wen | ✓ 0.819 vs 0.834 | yes |
| P22 | S2 | wen_fresh Gini < wen | ✓ 0.662 vs 0.679 | yes |
| P23 | S1 | wen_fresh Gini > random | ✓ 0.819 vs 0.084 | yes |
| P23 | S2 | wen_fresh Gini > random | ✓ 0.662 vs 0.057 | yes |
| P24 | S1 | gpu TTA@80% < wen_fresh | ✓ 1158 vs 1967 s | yes |
| P25 | S2 | gpu TTA@80% < wen_fresh (Stage 0 off) | ✗ 1348 vs 1145 s | no |

| P | verdict |
|---|---|
| P22 | direction as predicted in seed 4126; **no v3 verdict** (the rule needs 3 seeds) |
| P23 | direction as predicted in seed 4126; **no v3 verdict** (the rule needs 3 seeds) |
| P24 | direction as predicted in seed 4126; **no v3 verdict** (the rule needs 3 seeds) |
| P25 | direction not as predicted in seed 4126; **no v3 verdict** (the rule needs 3 seeds) |

## 5. Curves

`wen_fresh_v1_curves.png` (accuracy vs online clock; random / gpu / wen / wen_fresh; mean with min–max band), `wen_fresh_v1_gini.png`, `wen_fresh_v1_f3.png`.

## 6. Static counterfactual: F1 + F2 only, F3 + F4 only (Step 6)

Static replay on the logged candidate sets and logged F values (constraints unchanged; ignores feedback, as in Prompt F Step 3). The all-four replay reproduced every logged pick.

| scenario | seed | decisions | pick changes: F1 + F2 only | F3 + F4 only | Gini: logged (all four) | F1 + F2 only | F3 + F4 only |
|---|---|---|---|---|---|---|---|
| S1 | 4126 | 4002 | 0.907 | 0.418 | 0.834 | 0.290 | 0.889 |
| S1 | 4127 | 4002 | 0.892 | 0.449 | 0.817 | 0.296 | 0.881 |
| S1 | 4128 | 4002 | 0.895 | 0.591 | 0.796 | 0.309 | 0.879 |
| S2 | 4126 | 3403 | 0.704 | 0.411 | 0.679 | 0.447 | 0.777 |
| S2 | 4127 | 4203 | 0.643 | 0.542 | 0.627 | 0.438 | 0.737 |
| S2 | 4128 | 3860 | 0.733 | 0.453 | 0.665 | 0.468 | 0.778 |

## 7. Extra wall time spent on fresh evaluation

Step 5 runs, seed 4126 (5 runs in parallel at the start, then 2):

| run | evaluated versions | mean fresh_eval_s | total fresh evaluation (h) | run wall time (h) |
|---|---|---|---|---|
| S1_wen_fresh_s4126 | 4000 | 7.005 | 7.78 | 12.42 |
| S2_wen_fresh_s4126 | 2940 | 7.910 | 6.46 | 10.00 |

### Step 3 test 6 measurements

**Under real conditions** (the three 100-round test runs in parallel, each with 3 training clients):

| run | evaluated versions | mean fresh_eval_s | max | merges | projected extra wall time per 4,000-round run (h) |
|---|---|---|---|---|---|
| S1_wen_fresh_stale_s4126 | 100 | 11.113 | 16.245 | 100 | 12.35 |
| S1_wen_fresh_s4126_det1 | 100 | 11.234 | 16.033 | 100 | 12.48 |
| S1_wen_fresh_s4126_det2 | 100 | 11.138 | 15.042 | 100 | 12.38 |

**Standalone, on an otherwise idle GPU:** 4.40 s per evaluation (≈ 4.9 h per 4,000 versions). Batch 500 / 2,000 / 5,000 and channels_last gave 4.31–4.40 s. Peak GPU memory is 1.5 GB.

**Projection > 8 h, so I reported and waited.** The user accepted running 5 in parallel. Actual totals are in the table above.

## Deviations from Prompt H

1. **Step 1 was delayed.** The addendum was not in the repo when Prompt H arrived. The user then supplied it in `gpu_tiers_dataset/prompts/`, and it was committed before any run.
2. **F3 is evaluated lazily** at the first decision after each model change, not inside the merge itself. Every model version that a decision sees is evaluated exactly once. A version that no decision sees (the last merge of a run) is not evaluated. `fresh_eval_s` is logged per evaluated version.
3. **Test 5 needed a second attempt** after a determinism fix (§ B).
4. **Extra wall time is over 8 h per run** (12.4 h projected). Step 5 was started on the user's decision.
5. **Smoke ran in parallel with the test runs.** Step 5 was launched automatically, and only after every Step 3 test passed (`wen_env/logs/fresh_chain.sh`).
6. **Step 5 and Step 7 cover seed 4126 only**, at the user's request. Study tables show single values, P22–P25 are one-seed directions, the concentration section's "all seeds in an appendix" is seed 4126 only, and rolling Gini is not averaged.
7. **Last rolling-Gini window.** For runs that stopped early (random, gpu), the window ending at r≤2000 contains only a few rounds, so its Gini is not comparable with the others.

## A. Implementation (Step 2) and ADR

- **ADR-0018:** `docs/decisions/ADR-0018-wen-fresh-f3.md`.
- **`wen_env/wen_fresh.py`.**
  - `FreshLoss` loads the 50,000 training images and the `wen_partition` split once, onto the GPU.
  - It computes per-sample cross-entropy in eval mode with the CIFAR-10 test transform: ToTensor plus the training Normalize (ImageNet mean/std), no augmentation.
  - Per-vehicle means are summed in float64, with deterministic cuDNN.
  - `WenFreshSelection` subclasses `WenFaithfulSelection` and overrides only `f3`, plus the refresh and log columns.
- **`wen_server.py`.** `global_version` (+1 after each aggregation that changed the model). F3 is re-evaluated at the first decision after a version change and cached until the next one.
- **Logs.** `wen_decisions.csv` / `wen_candidates.csv` gain `global_version`. `wen_fresh_eval.csv` holds one row per version: `global_version, round, t, fresh_eval_s, F3_1..F3_30`.
- **Small, behaviour-neutral edits for `wen`.**
  - `wen_faithful.py`: field lists as class attributes, plus a `log_extra()` hook that returns nothing for `wen`.
  - `wen_config.py`: `wen_fresh` validated like `wen`; `wen_fresh_stale_f3_test` only for `wen_fresh`.
  - `wen_manifest.py`: v3.3 guard; new keys `predictions_v3_3_addendum_commit` and `wen_fresh_stale_f3_test`.
  - `wen_fedasync.py`: wiring.
  - `wen_make_configs.py`: `@stale` suffix.
- **Configs** (`wen_env/configs/runs/`).
  - They differ from the `wen` configs only in `arm`, `base_path` and `port`.
  - Test configs: `S1_wen_fresh_stale_s4126` (100 rounds), `S1_wen_fresh_s4126_det1/_det2` (100 rounds).
  - Smoke configs: `S{1,2}_wen_fresh_s4126_smoke` (200 rounds).
  - Run configs: `S{1,2}_wen_fresh_s412{6,7,8}` (4,000 rounds).

## B. Tests (Step 3)

Suite: **178 passed, 0 failed** (`wen_env/tests`). Then `test_wen_fresh.py` was run again on the runs: **11 passed, 0 skipped** (`wen_env/logs/fresh_pytest.txt`).

| # | Test | Result |
|---|---|---|
| 1 | `wen_fresh` with `wen_fresh_stale_f3_test = true` (`runs/wen_env/S1_wen_fresh_stale_s4126`) reproduces the `S1_wen_s4126` dispatch log for 100 rounds (cycle, round, client, t_dispatch, duration, t_finish_clean, accepted) | **pass** |
| 2 | F3 of clients 1, 21, 30 matches a direct torchvision evaluation (CIFAR10 + Subset + DataLoader) to 1e-6 | **pass** |
| 3 | Cache: F3 changes only when `global_version` changes; no recompute between merges; `global_version` logged | **pass** |
| 4 | Online clock = replayed clock for the 100-round `wen_fresh` run (`S1_wen_fresh_s4126_det1`): max \|Δ\| = 0.0 s, 0 merge-order residuals | **pass** |
| 5 | Two 100-round runs, seed 4126 (`_det1`, `_det2`): identical dispatch logs, `wen_decisions.csv` and `wen_candidates.csv` | **pass** (second attempt; see the incident below) |
| 6 | `fresh_eval_s` and projected extra wall time | § 7 |

**Incident: test 5 failed on the first attempt.**
- On the first attempt, the dispatch and decision logs were identical. But F3 in `wen_candidates.csv` differed in the last bit (2.2e-16) for 2 of 100 versions.
- **Cause:** the per-vehicle sum used CUDA `index_add_`, whose atomic adds have no fixed order.
- **Fix:** per-image losses are summed per vehicle on the CPU with `np.bincount`, in image order. 5 repeated evaluations were then bitwise identical, and the speed did not change.
- All three test runs were repeated with the fixed code. The first attempt is archived as `runs/wen_env/_failed/*_indexadd_nondet`.

## C. Smoke (Step 4): wen_fresh (200 rounds, seed 4126; accuracy not read)

| run | s/round | share X / O / T | fallback | candidates cut (all) | distinct vehicles | Gini | mean \|x̂ − x\| at upload (m) | mean \|x_upload − 1000\| (m) | mean d (m) | waste |
|---|---|---|---|---|---|---|---|---|---|---|
| S1_wen_fresh_s4126_smoke | 3.642 | 0.38 / 0.60 / 0.01 | 0.000 | 0.210 | 22 | 0.683 | 516 | 431 | 431 | 0.000 |
| S2_wen_fresh_s4126_smoke | 3.382 | 0.38 / 0.56 / 0.06 | 0.000 | 0.287 | 25 | 0.586 | 488 | 311 | 312 | 0.015 |

Constraint cuts per tier (share of that tier's candidates):

| run | tier | T only | E only | both |
|---|---|---|---|---|
| S1_wen_fresh_s4126_smoke | xavier | 0.133 | 0.000 | 0.000 |
| S1_wen_fresh_s4126_smoke | orin | 0.126 | 0.000 | 0.000 |
| S1_wen_fresh_s4126_smoke | thor | 0.000 | 0.579 | 0.000 |
| S2_wen_fresh_s4126_smoke | xavier | 0.228 | 0.000 | 0.000 |
| S2_wen_fresh_s4126_smoke | orin | 0.191 | 0.000 | 0.000 |
| S2_wen_fresh_s4126_smoke | thor | 0.000 | 0.618 | 0.000 |

n_pareto distribution (decisions):

- S1_wen_fresh_s4126_smoke: 2: 1, 4: 5, 5: 9, 6: 9, 7: 22, 8: 27, 9: 26, 10: 34, 11: 28, 12: 15, 13: 18, 14: 5, 15: 1, 16: 2
- S2_wen_fresh_s4126_smoke: 1: 1, 2: 2, 3: 11, 4: 24, 5: 36, 6: 37, 7: 45, 8: 25, 9: 14, 10: 6, 11: 2, 12: 2


Rolling Gini of dispatch counts (50-round windows, rounds 1–200):

| run | 1–50 | 51–100 | 101–150 | 151–200 |
|---|---|---|---|---|
| S1_wen_fresh_s4126_smoke | 0.713 | 0.759 | 0.679 | 0.792 |
| S2_wen_fresh_s4126_smoke | 0.626 | 0.687 | 0.679 | 0.635 |

Fleet F3 range (fresh loss of the global model, all 30 vehicles):

| run | round 50 (version) | round 100 | round 200 |
|---|---|---|---|
| S1_wen_fresh_s4126_smoke | 1.222–2.359 (v49) | 0.745–1.902 (v99) | 0.505–1.674 (v199) |
| S2_wen_fresh_s4126_smoke | 0.991–2.042 (v49) | 0.710–1.473 (v99) | 0.515–1.340 (v199) |

Fresh-evaluation time:

| run | evaluated versions | mean fresh_eval_s | max | merges | projected extra wall time per 4,000-round run (h) |
|---|---|---|---|---|---|
| S1_wen_fresh_s4126_smoke | 200 | 12.444 | 19.163 | 200 | 13.83 |
| S2_wen_fresh_s4126_smoke | 200 | 13.586 | 18.044 | 203 | 15.10 |

## Appendix — per-vehicle tables, wen_fresh, seed 4126

### S1, seed 4126

| client | tier | \|S\| | V (m/s) | F4 | F4 rank | dispatches | share | rank | mean F1 | mean F2 | mean F3 | pick nF1 | nF2 | nF3 | nF4 | random | gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | orin | 1667 | 21.0 | 0.1774 | 2 | 984 | 0.246 | 1 | 33.06 | 26.74 | 0.153 | 0.90 | 0.07 | 0.02 | 0.01 | 60 | 79 |
| 19 | orin | 1667 | 30.6 | 0.1664 | 1 | 919 | 0.230 | 2 | 33.10 | 26.75 | 0.215 | 0.88 | 0.07 | 0.04 | 0.00 | 61 | 72 |
| 22 | orin | 1666 | 17.6 | 0.2351 | 4 | 725 | 0.181 | 3 | 33.04 | 26.73 | 0.144 | 0.74 | 0.07 | 0.01 | 0.10 | 66 | 57 |
| 5 | xavier | 1667 | 17.1 | 0.2338 | 3 | 637 | 0.159 | 4 | 33.15 | 26.76 | 0.125 | 0.81 | 0.07 | 0.01 | 0.10 | 51 | 55 |
| 21 | orin | 1666 | 25.7 | 0.2548 | 5 | 448 | 0.112 | 5 | 32.99 | 26.72 | 0.201 | 0.64 | 0.06 | 0.05 | 0.18 | 59 | 69 |
| 20 | orin | 1667 | 17.7 | 0.3295 | 8 | 72 | 0.018 | 6 | 32.68 | 26.67 | 0.398 | 0.43 | 0.05 | 0.19 | 0.44 | 68 | 65 |
| 23 | orin | 1666 | 31.7 | 0.4204 | 19 | 32 | 0.008 | 7 | 32.60 | 26.64 | 0.377 | 0.23 | 0.03 | 0.30 | 0.79 | 68 | 58 |
| 13 | xavier | 1667 | 32.4 | 0.3039 | 7 | 28 | 0.007 | 8 | 32.62 | 26.65 | 0.642 | 0.27 | 0.05 | 0.66 | 0.35 | 49 | 58 |
| 15 | xavier | 1667 | 28.3 | 0.3611 | 11 | 21 | 0.005 | 9 | 32.60 | 26.65 | 0.473 | 0.23 | 0.03 | 0.46 | 0.56 | 63 | 63 |
| 6 | xavier | 1667 | 30.0 | 0.3500 | 10 | 17 | 0.004 | 10 | 32.60 | 26.65 | 0.533 | 0.19 | 0.04 | 0.60 | 0.45 | 46 | 65 |
| 1 | xavier | 1667 | 21.8 | 0.5206 | 26 | 14 | 0.003 | 11 | 32.61 | 26.65 | 0.438 | 0.03 | 0.01 | 0.47 | 0.97 | 70 | 39 |
| 14 | xavier | 1667 | 24.7 | 0.3825 | 12 | 13 | 0.003 | 12 | 32.60 | 26.65 | 0.649 | 0.21 | 0.07 | 0.60 | 0.55 | 50 | 58 |
| 18 | orin | 1667 | 25.6 | 0.3968 | 15 | 13 | 0.003 | 13 | 32.58 | 26.65 | 0.646 | 0.22 | 0.04 | 0.60 | 0.58 | 65 | 62 |
| 4 | xavier | 1667 | 24.0 | 0.4279 | 22 | 12 | 0.003 | 14 | 32.60 | 26.65 | 0.510 | 0.02 | 0.02 | 0.58 | 0.79 | 50 | 49 |
| 10 | xavier | 1667 | 31.9 | 0.3453 | 9 | 12 | 0.003 | 15 | 32.60 | 26.65 | 0.764 | 0.18 | 0.04 | 0.80 | 0.42 | 55 | 81 |
| 26 | thor | 1666 | 21.5 | 0.2553 | 6 | 10 | 0.002 | 16 | 28.99 | 36.46 | 0.532 | 0.00 | 1.00 | 0.18 | 0.17 | 68 | 60 |
| 8 | xavier | 1667 | 32.0 | 0.4229 | 20 | 9 | 0.002 | 17 | 32.60 | 26.65 | 0.564 | 0.16 | 0.04 | 0.51 | 0.66 | 62 | 52 |
| 3 | xavier | 1667 | 25.1 | 0.4802 | 24 | 8 | 0.002 | 18 | 32.59 | 26.65 | 0.511 | 0.10 | 0.02 | 0.49 | 0.82 | 36 | 52 |
| 9 | xavier | 1667 | 30.6 | 0.4267 | 21 | 8 | 0.002 | 19 | 32.52 | 26.64 | 0.581 | 0.11 | 0.06 | 0.65 | 0.69 | 75 | 63 |
| 11 | xavier | 1667 | 32.6 | 0.3846 | 13 | 8 | 0.002 | 20 | 32.60 | 26.65 | 0.693 | 0.00 | 0.00 | 0.92 | 0.69 | 63 | 70 |
| 12 | xavier | 1667 | 26.9 | 0.4418 | 23 | 5 | 0.001 | 21 | 32.59 | 26.65 | 0.699 | 0.08 | 0.00 | 0.82 | 0.77 | 48 | 46 |
| 2 | xavier | 1667 | 27.0 | 0.5375 | 28 | 3 | 0.001 | 22 | 32.59 | 26.65 | 0.558 | 0.07 | 0.07 | 0.37 | 1.00 | 63 | 75 |
| 7 | xavier | 1667 | 17.4 | 0.5297 | 27 | 3 | 0.001 | 23 | 32.61 | 26.65 | 0.917 | 0.00 | 0.00 | 1.00 | 0.81 | 47 | 72 |
| 17 | orin | 1667 | 23.0 | 0.6273 | 30 | 1 | 0.000 | 24 | 32.60 | 26.65 | 0.483 | 0.21 | 0.00 | 0.00 | 1.00 | 62 | 65 |
| 24 | orin | 1666 | — | 0.5790 | 29 | 0 | 0.000 | 25 | 32.58 | 26.64 | 0.734 | — | — | — | — | 61 | 55 |
| 25 | orin | 1666 | — | 0.5120 | 25 | 0 | 0.000 | 26 | 32.58 | 26.63 | 0.867 | — | — | — | — | 65 | 69 |
| 27 | thor | 1666 | — | 0.4022 | 17 | 0 | 0.000 | 27 | 28.98 | 36.46 | 0.691 | — | — | — | — | 70 | 65 |
| 28 | thor | 1666 | — | 0.3995 | 16 | 0 | 0.000 | 28 | 28.98 | 36.46 | 0.827 | — | — | — | — | 65 | 89 |
| 29 | thor | 1666 | — | 0.4123 | 18 | 0 | 0.000 | 29 | 28.98 | 36.46 | 0.705 | — | — | — | — | 50 | 87 |
| 30 | thor | 1666 | — | 0.3887 | 14 | 0 | 0.000 | 30 | 28.98 | 36.46 | 0.811 | — | — | — | — | 56 | 109 |

### S2, seed 4126

| client | tier | \|S\| | V (m/s) | F4 | F4 rank | dispatches | share | rank | mean F1 | mean F2 | mean F3 | pick nF1 | nF2 | nF3 | nF4 | random | gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | orin | 1667 | 21.0 | 0.1774 | 2 | 477 | 0.156 | 1 | 32.89 | 26.71 | 0.204 | 0.76 | 0.10 | 0.09 | 0.02 | 76 | 73 |
| 22 | orin | 1666 | 17.6 | 0.2351 | 4 | 477 | 0.156 | 2 | 32.56 | 26.63 | 0.139 | 0.74 | 0.09 | 0.05 | 0.10 | 80 | 70 |
| 5 | xavier | 1667 | 17.1 | 0.2338 | 3 | 406 | 0.133 | 3 | 32.59 | 26.65 | 0.162 | 0.74 | 0.11 | 0.06 | 0.10 | 65 | 58 |
| 19 | orin | 1667 | 30.6 | 0.1664 | 1 | 343 | 0.112 | 4 | 33.44 | 26.82 | 0.326 | 0.83 | 0.11 | 0.15 | 0.00 | 73 | 92 |
| 21 | orin | 1666 | 25.7 | 0.2548 | 5 | 329 | 0.108 | 5 | 33.15 | 26.75 | 0.211 | 0.72 | 0.08 | 0.10 | 0.13 | 76 | 79 |
| 20 | orin | 1667 | 17.7 | 0.3295 | 8 | 221 | 0.072 | 6 | 32.60 | 26.65 | 0.243 | 0.49 | 0.09 | 0.10 | 0.39 | 66 | 76 |
| 26 | thor | 1666 | 21.5 | 0.2553 | 6 | 115 | 0.038 | 7 | 28.80 | 36.42 | 0.244 | 0.04 | 1.00 | 0.10 | 0.02 | 89 | 54 |
| 15 | xavier | 1667 | 28.3 | 0.3611 | 11 | 87 | 0.029 | 8 | 33.24 | 26.78 | 0.263 | 0.52 | 0.08 | 0.19 | 0.36 | 86 | 57 |
| 14 | xavier | 1667 | 24.7 | 0.3825 | 12 | 78 | 0.026 | 9 | 32.87 | 26.71 | 0.390 | 0.33 | 0.04 | 0.48 | 0.47 | 58 | 78 |
| 4 | xavier | 1667 | 24.0 | 0.4279 | 22 | 64 | 0.021 | 10 | 32.86 | 26.70 | 0.311 | 0.18 | 0.03 | 0.43 | 0.72 | 78 | 62 |
| 13 | xavier | 1667 | 32.4 | 0.3039 | 7 | 59 | 0.019 | 11 | 33.42 | 26.81 | 0.429 | 0.64 | 0.11 | 0.36 | 0.09 | 69 | 73 |
| 1 | xavier | 1667 | 21.8 | 0.5206 | 26 | 57 | 0.019 | 12 | 32.65 | 26.66 | 0.293 | 0.09 | 0.02 | 0.48 | 0.95 | 72 | 41 |
| 18 | orin | 1667 | 25.6 | 0.3968 | 15 | 52 | 0.017 | 13 | 32.93 | 26.72 | 0.439 | 0.21 | 0.06 | 0.55 | 0.57 | 83 | 71 |
| 6 | xavier | 1667 | 30.0 | 0.3500 | 10 | 39 | 0.013 | 14 | 33.28 | 26.79 | 0.392 | 0.46 | 0.08 | 0.51 | 0.26 | 69 | 95 |
| 17 | orin | 1667 | 23.0 | 0.6273 | 30 | 39 | 0.013 | 15 | 32.69 | 26.67 | 0.317 | 0.07 | 0.01 | 0.34 | 1.00 | 78 | 68 |
| 3 | xavier | 1667 | 25.1 | 0.4802 | 24 | 33 | 0.011 | 16 | 32.89 | 26.71 | 0.362 | 0.04 | 0.00 | 0.55 | 0.82 | 72 | 75 |
| 7 | xavier | 1667 | 17.4 | 0.5297 | 27 | 31 | 0.010 | 17 | 32.29 | 26.59 | 0.631 | 0.01 | 0.01 | 0.85 | 0.93 | 76 | 88 |
| 2 | xavier | 1667 | 27.0 | 0.5375 | 28 | 30 | 0.010 | 18 | 33.06 | 26.74 | 0.362 | 0.08 | 0.03 | 0.42 | 0.94 | 82 | 70 |
| 9 | xavier | 1667 | 30.6 | 0.4267 | 21 | 26 | 0.009 | 19 | 33.31 | 26.79 | 0.383 | 0.34 | 0.07 | 0.43 | 0.63 | 76 | 64 |
| 23 | orin | 1666 | 31.7 | 0.4204 | 19 | 24 | 0.008 | 20 | 33.37 | 26.79 | 0.348 | 0.28 | 0.08 | 0.43 | 0.63 | 76 | 68 |
| 10 | xavier | 1667 | 31.9 | 0.3453 | 9 | 17 | 0.006 | 21 | 33.40 | 26.81 | 0.621 | 0.22 | 0.06 | 0.86 | 0.32 | 81 | 77 |
| 25 | orin | 1666 | 24.2 | 0.5120 | 25 | 15 | 0.005 | 22 | 32.76 | 26.67 | 0.643 | 0.01 | 0.01 | 0.77 | 0.89 | 79 | 77 |
| 12 | xavier | 1667 | 26.9 | 0.4418 | 23 | 13 | 0.004 | 23 | 33.03 | 26.74 | 0.582 | 0.08 | 0.01 | 0.71 | 0.72 | 78 | 46 |
| 8 | xavier | 1667 | 32.0 | 0.4229 | 20 | 9 | 0.003 | 24 | 33.37 | 26.80 | 0.455 | 0.36 | 0.11 | 0.23 | 0.58 | 73 | 56 |
| 11 | xavier | 1667 | 32.6 | 0.3846 | 13 | 8 | 0.003 | 25 | 33.40 | 26.81 | 0.620 | 0.12 | 0.04 | 0.81 | 0.63 | 72 | 81 |
| 24 | orin | 1666 | 31.5 | 0.5790 | 29 | 1 | 0.000 | 26 | 33.34 | 26.79 | 0.600 | 0.00 | 0.00 | 0.83 | 1.00 | 91 | 72 |
| 27 | thor | 1666 | 26.6 | 0.4022 | 17 | 1 | 0.000 | 27 | 28.97 | 36.45 | 0.559 | 0.00 | 1.00 | 0.43 | 0.00 | 93 | 74 |
| 28 | thor | 1666 | — | 0.3995 | 16 | 0 | 0.000 | 28 | 29.45 | 36.55 | 0.707 | — | — | — | — | 75 | 82 |
| 29 | thor | 1666 | — | 0.4123 | 18 | 0 | 0.000 | 29 | 29.10 | 36.48 | 0.622 | — | — | — | — | 72 | 101 |
| 30 | thor | 1666 | — | 0.3887 | 14 | 0 | 0.000 | 30 | 29.38 | 36.54 | 0.683 | — | — | — | — | 91 | 136 |
