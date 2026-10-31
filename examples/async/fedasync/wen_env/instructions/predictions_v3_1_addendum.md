# Pre-registered predictions v3.1 — addendum (locked 17 Oct 2026, before any cpu_cal run)

Adds one arm to v3 (`docs/predictions_v3.md`, commit f2bb90a1). v3 is unchanged. Commit this file before the first `cpu_cal` run and record its commit hash in every `cpu_cal` manifest.

## New arm: `cpu_cal` (S1 only)

Identical to `cpu` in every respect (CPU-cycle model b·|S_i|·ρ / f_tier, T̂_up per Wen Eq. 5–6, ε-greedy ε = 0.2, same seeds 4126 / 4127 / 4128, same stop rule as the main runs) except **ρ is calibrated** so that the fleet mean of T̂_cpu_train equals the fleet mean of the datasheet GPU training time (no jitter): ρ_cal = mean_i(T_train_tier(i)) / mean_i(b·|S_i| / f_tier(i)) ≈ 85 cycles/bit. Expected T̂_cpu_train ≈ 1.58 / 1.58 / 1.34 s (Xavier / Orin / Thor).

Purpose: separate the two faults of `cpu` — wrong scale (fixed here) and wrong tier ordering (kept: Xavier = Orin, Thor slightly favoured). It is a **CPU-favourable ablation**: ρ_cal has no literature source and uses fleet-mean information a real system would not have. The main `cpu` arm stays at ρ = 1,250.

## Predictions (v3 rule: same direction in all 3 seeds; mean ± SD reported; magnitudes not predicted)

- **P14a Scale fix recovers the position benefit:** cpu_cal's s/round gain over random is at least half of location-only's gain over random (location-only: −11.2%).
- **P14b Tier ordering still matters:** gpu has lower s/round than cpu_cal.
- **P14c Energy:** energy to reach 80% is lower for gpu than for cpu_cal.
- **P14d Scale was the bigger fault:** cpu_cal has lower s/round than cpu (ρ = 1,250).

Context only (not predictions): with T̂_cpu_train within 1.34–1.58 s for every tier, ranking is dominated by T̂_up as in location-only, so cpu_cal is expected to sit near location-only on time, and to dispatch Xavier at least as often as random.
