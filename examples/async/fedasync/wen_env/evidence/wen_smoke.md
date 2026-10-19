# Prompt E, Step 4: Wen smoke (200 rounds, seed 4126; accuracy not read)

| run | s/round | share X / O / T | fallback | candidates cut (all) | distinct vehicles | Gini | mean \|x̂ − x\| at upload (m) | mean \|x_upload − 1000\| (m) | mean d (m) | waste |
|---|---|---|---|---|---|---|---|---|---|---|
| S1_wen_s4126_smoke | 3.492 | 0.22 / 0.78 / 0.00 | 0.000 | 0.210 | 15 | 0.831 | 508 | 449 | 450 | 0.000 |
| S2_wen_s4126_smoke | 3.403 | 0.33 / 0.66 / 0.00 | 0.000 | 0.285 | 23 | 0.660 | 485 | 307 | 307 | 0.019 |

Constraint cuts per tier (share of that tier's candidates):

| run | tier | T only | E only | both |
|---|---|---|---|---|
| S1_wen_s4126_smoke | xavier | 0.128 | 0.000 | 0.000 |
| S1_wen_s4126_smoke | orin | 0.132 | 0.000 | 0.000 |
| S1_wen_s4126_smoke | thor | 0.000 | 0.582 | 0.000 |
| S2_wen_s4126_smoke | xavier | 0.206 | 0.000 | 0.000 |
| S2_wen_s4126_smoke | orin | 0.214 | 0.000 | 0.000 |
| S2_wen_s4126_smoke | thor | 0.000 | 0.618 | 0.000 |

n_pareto distribution (decisions):

- S1_wen_s4126_smoke: 2: 1, 3: 3, 4: 20, 5: 24, 6: 32, 7: 22, 8: 38, 9: 30, 10: 17, 11: 10, 12: 4, 13: 1
- S2_wen_s4126_smoke: 1: 2, 2: 6, 3: 21, 4: 28, 5: 43, 6: 43, 7: 33, 8: 17, 9: 8, 10: 4, 11: 1
