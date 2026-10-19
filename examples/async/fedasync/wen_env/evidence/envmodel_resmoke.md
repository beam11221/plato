# Prompt C, Step 3: re-smoke (200 rounds, seed 4126; accuracy not read)

| cell | rule | s / round | share X / O / T | explored | mean rank (exploit) | V_max − V_min: q1 / median / q3 / max | Stage 0 rejected | clock diff |
|---|---|---|---|---|---|---|---|---|
| S1_cpu_s4126_smoke2 | egreedy ε=0.2 | 3.716 | 0.38 / 0.28 / 0.34 | 0.228 | 1.00 | 0.1067 / 0.1253 / 0.1646 / 0.5498 | — | 0.0e+00 |
| S1_gpu_eps0_s4126_smoke2 | egreedy ε=0.0 | 2.776 | 0.22 / 0.50 / 0.28 | 0.000 | 1.00 | 0.1940 / 0.2261 / 0.2702 / 0.6266 | — | 0.0e+00 |
| S1_gpu_prop_s4126_smoke2 | proportional | 3.721 | 0.51 / 0.31 / 0.18 | — | — | — | — | 0.0e+00 |
| S1_gpu_s4126_smoke2 | egreedy ε=0.2 | 3.014 | 0.22 / 0.44 / 0.34 | 0.228 | 1.00 | 0.2015 / 0.2307 / 0.2805 / 0.7320 | — | 0.0e+00 |
| S1_random_s4126_smoke2 | egreedy ε=1.0 | 3.823 | 0.44 / 0.39 / 0.18 | 1.000 | — | — | — | 0.0e+00 |
| S2_cpu_stage0_s4126_smoke2 | egreedy ε=0.2 | 3.602 | 0.43 / 0.29 / 0.28 | 0.227 | 1.00 | 0.0394 / 0.0915 / 0.1392 / 1.0373 | 74.4% | 0.0e+00 |
| S2_gpu_stage0_s4126_smoke2 | egreedy ε=0.2 | 2.997 | 0.29 / 0.44 / 0.27 | 0.225 | 1.00 | 0.1377 / 0.1756 / 0.2121 / 0.8283 | 24.7% | 0.0e+00 |
