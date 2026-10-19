# ADR-0016: `cpu_cal`, a CPU-favourable calibration ablation

Status: accepted (Prompt D). Implemented on 17 Oct 2026.

## Purpose

The `cpu` arm uses Wen's Eq. 4 with ρ = 1,250 cycles/bit (ADR-0009). Its training estimate, T̂_train ≈ 23.3 / 23.3 / 19.7 s, is about 15 times the fleet's true mean training time. In the main runs that scale swamped the upload term, so `cpu` selected almost independently of position (main runs v1; mechanism tables: mean upload distance 470 m against 284 m for gpu).

`cpu_cal` removes that scale error and keeps everything else about the CPU model. It asks whether the CPU signal fails *only* because its scale is wrong, or also because it ranks the tiers wrongly. On the CPU clock, Xavier and Orin tie at 2.2 GHz, and Thor (2.6 GHz) is only 1.18 times faster. The GPU datasheet gives a gap of about 15 times.

## Decision

**Same code path as `cpu`; only ρ differs.** `CpuEstimator(rho=ρ_cal)` is used for:
- the training estimate, Wen Eq. 4: T̂ = b·|S_i|·ρ / f;
- the energy estimate, Wen Eq. 7 + 8: Ê = e·ρ·b·|S_i|·f² + p·T̂_up (CPU family, as in D2).

Ground-truth energy is unchanged.

**The formula for ρ_cal.** It is computed from the fleet, never typed by hand:

```
ρ_cal = mean_i T_train_tier(i) / mean_i (b · |S_i| / f_tier(i))
```

- T_train_tier is the datasheet (GPU) training time without jitter, using each client's own |S_i|.
- b = 24,576 bits per image.
- f = 2.2 / 2.2 / 2.6 GHz.

`wen_make_configs.py` computes ρ_cal and writes it into each config as `[wen_env] cpu_rho`. At start-up, `wen_fedasync.py` recomputes ρ_cal from the fleet and refuses any mismatch larger than 1e-9 relative. The manifest records the value.

**Values for fleet `n30_x15-o10-t5_phys_p1667_j0p149_s4126`:**
- ρ_cal = **84.98715511665667** cycles/bit.
- T̂_train = **1.58262 s** for Xavier and Orin at |S| = 1,667 (Orin 1.58167 s at |S| = 1,666), and **1.33834 s** for Thor at |S| = 1,666.
- The fleet mean of T̂_train equals the fleet mean of the datasheet training time to 1e-9 relative (1.541751 s).

**Scope.** S1 only, ε-greedy with ε = 0.2, and the same stop rule as the main runs (`target_accuracy = 0.9`), so that it is comparable with the existing 36 runs.

## Caveat

ρ_cal uses **fleet-mean information that a deployed system would not have**: the true GPU training time, averaged over the fleet. `cpu_cal` is therefore not a deployable estimator. It is an upper-bound-style ablation that gives the CPU model every advantage except per-tier ranking.

## Safeguards

- **No run before the predictions are locked.** No `cpu_cal` run, of any length, may start until `docs/predictions_v3_1_addendum.md` is committed. `wen_manifest.py` refuses otherwise, and records the addendum's commit hash in each manifest.
- **The ρ = 1,250 check.** A config may set `cpu_rho_test_override = true` to bypass the equality check against the recomputed ρ_cal. This is used only for the reproduction check: `cpu_cal` at ρ = 1,250 must reproduce the `cpu` dispatch log (Prompt D, Step 2, test 1).
