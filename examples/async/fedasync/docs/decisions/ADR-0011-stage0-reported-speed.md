# ADR-0011: Stage 0 uses each vehicle's reported V_k, and an energy cap

Status: accepted 14 Oct 2026, implemented 16 Oct 2026 (Prompt B, Step 6).

## Decision

In S2, a vehicle at x ∈ [400, 1600] is feasible if and only if both of these hold:

```
R_i = (1600 − x_i) / V_k  ≥  T̂_train + T̂_up
Ê_i ≤ E_max
```

- Traffic is one-way eastbound, so 1,600 m is always the exit.
- V_k is the vehicle's own constant speed, which it reports. It replaces the old worst-case `v_max = 19.44` m/s of the E1 trace (ADR-0004 era, `sumo_close_loop/stage0_filter.py`).
- Each arm uses its own estimator for T̂ and Ê.

**Pluggability.** `wen_env/wen_selection.WenStage0Gate` wraps any `ClientSelectionStrategy`, so the Wen selector can reuse it later.

**Fallback.** If no candidate is feasible, the unfiltered pool is used and a fallback is logged. This is the existing Stage 0 rule: never leave a slot empty that could have been filled.

## Check

`tests/test_wen_truth_and_selection.py` takes about 4,000 admissions on the real trace using the gpu_oracle T̂ with |χ|² = 1.
- With |χ|² = 1, it asserts that Stage 0 never admits a vehicle that then fails coverage.
- With real fading, a failure can still occur when a draw lengthens the upload. These cases are counted separately and must stay below 5%.
