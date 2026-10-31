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
