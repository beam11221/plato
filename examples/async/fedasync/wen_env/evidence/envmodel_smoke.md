# Prompt B, Step 9: smoke-run numbers (plumbing only; accuracy not read)

| cell | dispatches | upload share | waste | median feasible-idle (gpu T̂) | median in-coverage idle | s / round | clock max diff (s) | order residuals |
|---|---|---|---|---|---|---|---|---|
| S1_cpu_s4126_smoke | 202 | 0.865 | 0.000 | 19.0 | 28.0 | 3.734 | 0.00e+00 | 0 |
| S1_gpu_s4126_smoke | 202 | 0.865 | 0.000 | 19.0 | 28.0 | 3.721 | 0.00e+00 | 0 |
| S1_random_s4126_smoke | 202 | 0.868 | 0.000 | 18.5 | 28.0 | 3.767 | 0.00e+00 | 0 |
| S2_gpu_s4126_smoke | 250 | 0.850 | 0.192 | 13.0 | 16.0 | 4.191 | 0.00e+00 | 0 |
| S2_gpu_stage0_s4126_smoke | 204 | 0.861 | 0.010 | 12.0 | 16.0 | 3.308 | 0.00e+00 | 0 |

## Per tier: true vs estimated (means over dispatches)

| cell | tier | n | true T_train | true T_up | true total | T̂_train | T̂ total | s(τ̂) | energy (J) | waste |
|---|---|---|---|---|---|---|---|---|---|---|
| S1_cpu_s4126_smoke | xavier | 104 | 2.618 | 9.626 | 12.243 | 23.277 | 32.871 | 0.494 | 106.6 | 0.000 |
| S1_cpu_s4126_smoke | orin | 62 | 0.365 | 9.367 | 9.732 | 23.271 | 32.501 | 0.497 | 29.2 | 0.000 |
| S1_cpu_s4126_smoke | thor | 36 | 0.184 | 9.569 | 9.753 | 19.684 | 29.139 | 0.517 | 25.8 | 0.000 |
| S1_gpu_s4126_smoke | xavier | 103 | 2.627 | 9.669 | 12.296 | 2.784 | 12.112 | 0.480 | 107.0 | 0.000 |
| S1_gpu_s4126_smoke | orin | 63 | 0.359 | 9.465 | 9.824 | 0.357 | 9.254 | 0.532 | 28.8 | 0.000 |
| S1_gpu_s4126_smoke | thor | 36 | 0.184 | 9.322 | 9.506 | 0.185 | 9.182 | 0.534 | 25.8 | 0.000 |
| S1_random_s4126_smoke | xavier | 100 | 2.658 | 9.711 | 12.370 | — | — | — | 108.3 | 0.000 |
| S1_random_s4126_smoke | orin | 71 | 0.358 | 9.497 | 9.855 | — | — | — | 28.7 | 0.000 |
| S1_random_s4126_smoke | thor | 31 | 0.187 | 9.750 | 9.937 | — | — | — | 26.2 | 0.000 |
| S2_gpu_s4126_smoke | xavier | 131 | 2.593 | 8.583 | 11.177 | 2.784 | 11.033 | 0.497 | 105.4 | 0.191 |
| S2_gpu_s4126_smoke | orin | 77 | 0.365 | 8.525 | 8.890 | 0.357 | 8.548 | 0.546 | 29.1 | 0.195 |
| S2_gpu_s4126_smoke | thor | 42 | 0.188 | 8.366 | 8.555 | 0.185 | 8.242 | 0.554 | 26.2 | 0.190 |
| S2_gpu_stage0_s4126_smoke | xavier | 94 | 2.584 | 8.322 | 10.906 | 2.784 | 10.589 | 0.504 | 105.0 | 0.021 |
| S2_gpu_stage0_s4126_smoke | orin | 72 | 0.354 | 8.306 | 8.660 | 0.357 | 8.113 | 0.556 | 28.2 | 0.000 |
| S2_gpu_stage0_s4126_smoke | thor | 38 | 0.186 | 8.442 | 8.628 | 0.185 | 8.107 | 0.556 | 25.9 | 0.000 |

## Waste (S2)

- S2_gpu_s4126_smoke: 19.2% of dispatches; 72.2 GPU-s, 485.0 upload-s, 3304.5 J wasted
- S2_gpu_stage0_s4126_smoke: 1.0% of dispatches; 5.5 GPU-s, 24.8 upload-s, 225.8 J wasted

## T_round and s(τ̂) for the gpu arm (Step 5 report)

| | S1 gpu | S2 gpu | before (anchored, no upload) |
|---|---|---|---|
| startup T_round (mean T̂ over the fleet at t = 0, divided by C) | 3.585 s | 3.585 s | 2.081 s |
| steady state: mean T̂_total of the dispatched clients / C | 3.57 s | 3.27 s | — |
| measured seconds per round | 3.72 s | 4.19 s (waste holds the round) | — |
| s(τ̂), Thor / Orin / Xavier | 0.534 / 0.532 / 0.480 | 0.554 / 0.546 / 0.497 | 0.82 / 0.72 / 0.35 |

Upload (about 9.5 s for every tier) dominates T̂. The tier spread in s(τ̂) therefore shrinks from 0.82–0.35 to 0.53–0.48. This is the communication-bound regime the pre-registered predictions describe.
