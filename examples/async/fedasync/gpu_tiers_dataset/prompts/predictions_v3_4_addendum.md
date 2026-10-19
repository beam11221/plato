# Pre-registered predictions v3.4 — Study 2 plug-in study (locked 19 Oct 2026, before any Study 2 run)

Adds Study 2 to v3 (`f2bb90a1`), v3.1 (`098e1dd1`), v3.2 (`66a2ed82`) and v3.3 (`77e2d7b7`), which are unchanged. Commit this file before the first Study 2 run and record its hash in every Study 2 manifest.

## Question

Are our two components plug-ins that help other selectors? (a) **Stage 0** (feasibility filter with our estimator). (b) **GPU-aware estimator** replacing Wen's CPU F1 / F2. Hosts are tested as proposed; they are not re-engineered.

## Cells (host for Wen = `wen_fresh`)

| # | Cell | Scenario | Compared with (existing run) |
|---|---|---|---|
| 1 | random + Stage 0 | S2 | random (Stage 0 off) |
| 2 | wen_fresh + Stage 0 | S2 | wen_fresh |
| 3 | wen_fresh + GPU | S1 | wen_fresh |
| 4 | wen_fresh + GPU | S2 | wen_fresh |
| 5 | wen_fresh + GPU + Stage 0 | S2 | wen_fresh + GPU, wen_fresh + Stage 0 |

Pipeline for every cell: **eligibility (environment: idle, and in S2 inside coverage at dispatch) → Stage 0 (if on) → host's own scoring and constraints → host's selection.**

- Stage 0 uses our estimator: T̂ = T̂_train(GPU) + T̂_up, residence (1,600 − x)/V_k ≥ T̂, Ê (GPU family) ≤ 600 J; same fallback as the existing gpu + Stage 0 arm.
- GPU plug-in: F1 = T̂_train(GPU) + T̂_up with x̂ = x + V_k·T̂_train(GPU); F2 = P_tier·T̂_train + p·T̂_up; T_max, E_max = 90th percentile of the new F1, F2 (W5 rule; planning estimate ≈ 13.57 s, ≈ 113.5 J). F3 (fresh), F4, Pareto, min–max, pick rule unchanged.

## Predictions (v3 rule: same direction in all 3 seeds; seed 4126 alone gives a direction, not a verdict)

**Study 2a — Stage 0 plug-in (S2)**
- **P26:** random + Stage 0 has a lower waste fraction (dispatches) than random.
- **P27:** random + Stage 0 has a lower TTA@80% than random.
- **P28:** wen_fresh + Stage 0 has a lower waste fraction (dispatches) than wen_fresh.

**Study 2b — GPU plug-in (S1)**
- **P29 (mechanism):** wen_fresh + GPU has a lower mean |x̂ − x| at upload than wen_fresh.
- **P30 (mechanism):** wen_fresh + GPU has a lower mean true upload distance |x_upload − 1000| than wen_fresh.
- **P31:** wen_fresh + GPU has a lower s/round than wen_fresh.
- **P32:** wen_fresh + GPU has a higher Thor dispatch share (whole run) than wen_fresh.

**Cell 5**
- **P33:** wen_fresh + GPU + Stage 0 has a waste fraction (dispatches) no higher than wen_fresh + GPU.

## Reported, not predicted

TTA@80 / 85 / 90% for cells 2–5 against their comparators (the F3 / F4 collapse may mask any effect); whether S1 wen_fresh + GPU reaches 90%; dispatch Gini; Stage 0 rejection rate per tier; constraint cuts per tier under the GPU limits; the S2 versions of P29–P32 (cell 4); energy to 80%.

Context (not a prediction): `wen_fresh` already has low S2 waste (3.6% at seed 4126) because its CPU-scale x̂ steers picks to the coverage entry; Stage 0 has little waste left to remove there. Under the GPU energy model Xavier, not Thor, is the expensive tier, so the GPU plug-in should shift Wen's picks away from Xavier and toward Orin / Thor.
