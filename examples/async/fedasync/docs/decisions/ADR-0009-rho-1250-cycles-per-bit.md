# ADR-0009: ρ = 1,250 cycles/bit for the CPU-cycle estimator

Status: accepted 15 Oct 2026.

## Decision

The CPU arm uses Wen Eq. 4:

```
T̂_cpu = b · |S| · ρ / f
```

| Symbol | Value |
|---|---|
| b | 24,576 bits per image (32 × 32 × 3 × 8) |
| ρ | 1,250 cycles/bit, the same for every vehicle |
| f | the tier's CPU clock: 2.2 / 2.2 / 2.6 GHz (Xavier / Orin / Thor) |

The clocks come from `gpu_tiers_dataset/tiers/tiers_wen.csv`. That is a new file: the older tier files list 2.0 / 2.0 / 2.6 GHz and were left untouched, because finished runs depend on their bytes.

**Result:** T̂_cpu = 23.28 / 23.28 / 19.70 s at |S| = 1,667.

## Citation chain

- Wen et al. give no value for ρ.
- Wen's Eq. 4 cites [35] (Li et al., TVT 2022), which models upload energy only.
- Wen's Table 2 cites [12, 14, 42, 44–46]. Of these, [44] (Xiao et al., T-ITS 2022, Table II) draws the processing density per vehicle from [900, 1600] cycles/bit.
- 1,250 is the midpoint of that range.

## Alternatives

A calibrated ρ ≈ 427 would make the fleet-mean T̂_cpu equal the true fleet mean. It is proposed as a sensitivity cell, not yet confirmed.

## Consequence

At ρ = 1,250 the CPU estimate is about 20–23 s, which is roughly 8–100× the true training time. It swamps the upload term, so the cpu arm ranks almost purely by clock. That clock ties Xavier with Orin.
