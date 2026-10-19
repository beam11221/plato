# ADR-0010: Energy model and E_max

Status: accepted 14 Oct 2026, implemented 16 Oct 2026.

## Decision

**Ground truth, per dispatch:**

```
E = P_tier × T_train_true + p × T_up_true
```

- P_tier = 40 / 75 / 130 W for Xavier / Orin / Thor. These are the top power modes on the NVIDIA datasheets, consistent with the TFLOPS figures.
- p = 0.2 W, i.e. 23 dBm (Wen Eq. 8).

**Estimate.** `Ê = P_tier × T̂_train + p × T̂_up`. Each arm uses its own T̂.

**E_max = 600 J.**
- It is non-binding for the gpu-family arms: their largest per-task estimate is about 114 J (Xavier: 40 W × 2.78 s plus about 2 J of upload).
- Energy acts as a reported metric and through Stage 0's `Ê ≤ E_max`.
- An E_max sweep is a separate sensitivity study, never tuned on outcome.

**Logging.** Every dispatch logs `E_train` and `E_up`. Rejected (S2) uploads also add their energy to the waste tally.

## Note

**E_max binds for the cpu arm, in every tier.** The cpu arm's Ê is P × 23.3 s or P × 19.7 s, which comes to 931 J for Xavier, 1,746 J for Orin and 2,561 J for Thor. All three exceed 600 J.

With Stage 0 on, the cpu arm would therefore reject every vehicle on energy alone and fall back to the unfiltered pool on every cycle. That would make "cpu + Stage 0" (run plan #13) identical to cpu without Stage 0.

This is a direct consequence of the locked values (ρ = 1,250, E_max = 600 J, P_tier). It must be resolved before that cell is run. Options:
- apply the energy test with the GPU-side Ê only;
- raise E_max;
- accept the result as the finding.

It does not affect any cell of the Prompt B smoke run.
