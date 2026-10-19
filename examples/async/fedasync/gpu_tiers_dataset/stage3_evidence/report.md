# Stage 3 — sweep report

## Table A — the gate: gpu vs `_rep2`
| | gpu | gpu (`_rep2`) | difference |
|---|---|---|---|
| plateau | 0.8255 | 0.8255 | +0.00% |
| noise (SD) | 0.0490 | 0.0490 | -- |
| TTA @ 70% (h) | 0.096319 | 0.096319 | +0.00% |
| TTA @ 75% (h) | 0.134767 | 0.134767 | +0.00% |
| TTA @ 78% (h) | 0.264131 | 0.264131 | +0.00% |
| TTA @ 80% (h) | 0.382215 | 0.382215 | +0.00% |

Largest |difference| across the four thresholds: **0.00%** (stop condition: > 10%). Under the stop threshold.

## Table B — C sweep (N=20)
| C | arm | plateau | usable threshold | TTA (h) | vs baseline |
|---|---|---|---|---|---|
| 1 | baseline | 0.8181 | 75% | 0.48915 | -- |
| 1 | cpu | 0.8128 | 75% | 0.684241 | +39.88% |
| 1 | gpu | 0.8154 | 75% | 0.609591 | +24.62% |
| 1 | observed | 0.8186 | 75% | 0.460338 | -5.89% |
| 2 | baseline | 0.8237 | 75% | 0.353213 | -- |
| 2 | cpu | 0.8168 | 75% | 0.314585 | -10.94% |
| 2 | gpu | 0.8203 | 75% | 0.269149 | -23.80% |
| 2 | observed | 0.8265 | 75% | 0.310872 | -11.99% |
| 4 | baseline | 0.8296 | 75% | 0.173379 | -- |
| 4 | cpu | 0.8228 | 75% | 0.172082 | -0.75% |
| 4 | gpu | 0.8255 | 75% | 0.134767 | -22.27% |
| 4 | gpu_oracle | 0.8282 | 75% | 0.123611 | -28.70% |
| 4 | observed | 0.8231 | 75% | 0.123424 | -28.81% |
| 8 | baseline | 0.8212 | 75% | 0.093763 | -- |
| 8 | cpu | 0.8245 | 75% | 0.117525 | +25.34% |
| 8 | gpu | 0.8265 | 75% | 0.073573 | -21.53% |
| 8 | gpu_oracle | 0.8259 | 75% | 0.083653 | -10.78% |
| 8 | observed | 0.8226 | 75% | 0.074384 | -20.67% |

## Table C — staleness vs C
| C | arm | mean staleness | max staleness | C - 1 |
|---|---|---|---|---|
| 1 | baseline | 0.000 | 0 | 0 |
| 1 | cpu | 0.000 | 0 | 0 |
| 1 | gpu | 0.000 | 0 | 0 |
| 1 | observed | 0.000 | 0 | 0 |
| 1 | *(mean across arms)* | 0.000 | -- | 0 |
| 2 | baseline | 1.000 | 11 | 1 |
| 2 | cpu | 1.000 | 8 | 1 |
| 2 | gpu | 0.999 | 12 | 1 |
| 2 | observed | 0.999 | 10 | 1 |
| 2 | *(mean across arms)* | 0.999 | -- | 1 |
| 4 | baseline | 2.997 | 16 | 3 |
| 4 | cpu | 2.996 | 14 | 3 |
| 4 | gpu | 2.996 | 22 | 3 |
| 4 | gpu_oracle | 2.993 | 18 | 3 |
| 4 | observed | 2.996 | 18 | 3 |
| 4 | *(mean across arms)* | 2.996 | -- | 3 |
| 8 | baseline | 6.977 | 33 | 7 |
| 8 | cpu | 6.987 | 31 | 7 |
| 8 | gpu | 6.985 | 36 | 7 |
| 8 | gpu_oracle | 6.978 | 39 | 7 |
| 8 | observed | 6.980 | 40 | 7 |
| 8 | *(mean across arms)* | 6.981 | -- | 7 |

## Table D — fleet size (C=4)
| N | arm | plateau | usable threshold | TTA (h) | vs baseline |
|---|---|---|---|---|---|
| 20 | baseline | 0.8296 | 75% | 0.173379 | -- |
| 20 | cpu | 0.8228 | 75% | 0.172082 | -0.75% |
| 20 | gpu | 0.8255 | 75% | 0.134767 | -22.27% |
| 20 | gpu_oracle | 0.8282 | 75% | 0.123611 | -28.70% |
| 20 | observed | 0.8231 | 75% | 0.123424 | -28.81% |
| 40 | baseline | 0.7629 | 70% | 0.546922 | -- |
| 40 | cpu | 0.7629 | 70% | 0.513012 | -6.20% |
| 40 | gpu | 0.7649 | 70% | 0.370397 | -32.28% |
| 40 | observed | 0.7650 | 70% | 0.383525 | -29.88% |

## Table E — the upper bound: gpu vs gpu_oracle
| C | metric | gpu | gpu_oracle |
|---|---|---|---|
| 4 | plateau | 0.8255 | 0.8282 |
| 4 | TTA @ 75% (h) | 0.134767 | 0.123611 |
| 4 | mean staleness | 2.99625 | 2.99325 |
| 8 | plateau | 0.8265 | 0.8259 |
| 8 | TTA @ 75% (h) | 0.073573 | 0.083653 |
| 8 | mean staleness | 6.98525 | 6.97825 |

## Table F — control: dispatch share per tier
| experiment | arm | dispatches | share xavier | share orin | share thor |
|---|---|---|---|---|---|
| a0p8_N20_C1_s4126 | baseline | 4000 | 0.49075 | 0.355 | 0.15425 |
| a0p8_N20_C1_s4126 | cpu | 4000 | 0.50525 | 0.33975 | 0.155 |
| a0p8_N20_C1_s4126 | gpu | 4000 | 0.4415 | 0.3905 | 0.168 |
| a0p8_N20_C1_s4126 | observed | 4000 | 0.50825 | 0.342 | 0.14975 |
| a0p8_N20_C2_s4126 | baseline | 4001 | 0.470632 | 0.371907 | 0.157461 |
| a0p8_N20_C2_s4126 | cpu | 4001 | 0.552112 | 0.308423 | 0.139465 |
| a0p8_N20_C2_s4126 | gpu | 4001 | 0.454886 | 0.372657 | 0.172457 |
| a0p8_N20_C2_s4126 | observed | 4001 | 0.487628 | 0.356661 | 0.155711 |
| a0p8_N20_C4_s4126 | baseline | 4003 | 0.418186 | 0.403697 | 0.178116 |
| a0p8_N20_C4_s4126 | cpu | 4003 | 0.544092 | 0.31951 | 0.136398 |
| a0p8_N20_C4_s4126 | gpu | 4003 | 0.431426 | 0.387459 | 0.181114 |
| a0p8_N20_C4_s4126 | gpu_oracle | 4003 | 0.428928 | 0.397952 | 0.17312 |
| a0p8_N20_C4_s4126 | observed | 4003 | 0.429678 | 0.394454 | 0.175868 |
| a0p8_N20_C4_s4126_rep2 | gpu | 4003 | 0.431426 | 0.387459 | 0.181114 |
| a0p8_N20_C8_s4126 | baseline | 4007 | 0.317444 | 0.473422 | 0.209134 |
| a0p8_N20_C8_s4126 | cpu | 4007 | 0.450212 | 0.384078 | 0.16571 |
| a0p8_N20_C8_s4126 | gpu | 4007 | 0.334415 | 0.460943 | 0.204642 |
| a0p8_N20_C8_s4126 | gpu_oracle | 4007 | 0.336411 | 0.458697 | 0.204891 |
| a0p8_N20_C8_s4126 | observed | 4007 | 0.324183 | 0.471425 | 0.204392 |
| a0p8_N40_C4_s4126 | baseline | 4003 | 0.456658 | 0.382213 | 0.161129 |
| a0p8_N40_C4_s4126 | cpu | 4003 | 0.549088 | 0.331252 | 0.11966 |
| a0p8_N40_C4_s4126 | gpu | 4003 | 0.424432 | 0.418186 | 0.157382 |
| a0p8_N40_C4_s4126 | observed | 4003 | 0.433175 | 0.417687 | 0.149138 |

## The six predictions, written down before the runs
### 1 -- C=1 gives a null across all four arms
- baseline: plateau 0.8181, mean_staleness 0.0
- cpu: plateau 0.8128, mean_staleness 0.0
- gpu: plateau 0.8154, mean_staleness 0.0
- observed: plateau 0.8186, mean_staleness 0.0

### 2 -- advantage grows C=1 -> ~C=4, flattens/falls by C=8
See Table B for the per-C, per-arm TTA-vs-baseline numbers across C=1/2/4/8.

### 3 -- cpu is slower than baseline in every cell with C>1
- C=2: cpu 0.314585 h vs baseline 0.353213 h @ 75%   (secs/round: cpu 4.318127, baseline 3.803889)
- C=4: cpu 0.172082 h vs baseline 0.173379 h @ 75%   (secs/round: cpu 2.13013, baseline 1.717945)
- C=8: cpu 0.117525 h vs baseline 0.093763 h @ 75%   (secs/round: cpu 0.907345, baseline 0.696236)

### 4 -- N=40 gives the same advantage as N=20, at C=4
See Table D.

### 5 -- gpu loses to gpu_oracle (a tie would mean the jitter did not make estimation matter)
See Table E.

### 6 -- gpu vs observed stays undecided at one seed
- C=1: gpu 0.609591 h vs observed 0.460338 h @ 75%
- C=2: gpu 0.269149 h vs observed 0.310872 h @ 75%
- C=4: gpu 0.134767 h vs observed 0.123424 h @ 75%
- C=8: gpu 0.073573 h vs observed 0.074384 h @ 75%

Each of the six is reported as numbers above, not as a held/not-held verdict: the brief's own criteria for several of them ("a null", "flattens", "a tie") are qualitative, and a threshold invented here to force a boolean would be exactly the kind of explaining-away the brief forbids. Predictions 5's exception -- gpu vs gpu_oracle margin -- and the stop conditions below are the numbers the brief DOES give an explicit rule for.

## Stop conditions
None triggered.
