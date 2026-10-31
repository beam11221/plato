# Pre-registered predictions v3.2 — addendum (locked 18 Oct 2026, before any Wen-faithful run)

Adds the `wen` arm (Wen-faithful selector, async adaptation W1–W10) to v3 (`docs/predictions_v3.md`, commit f2bb90a1) and v3.1 (`docs/predictions_v3_1_addendum.md`). v3 and v3.1 are unchanged. Commit this file before the first `wen` run and record its commit hash in every `wen` manifest.

## Arm: `wen` (S1 and S2, Stage 0 off)

Wen et al. (Information Sciences 664, 2024) RA-MaOVSM adapted to async FedAsync, C = 3, one pick per free slot:
- Objectives per idle eligible candidate (all minimised): F1 = T̂_com + T̂_up, T̂_com = b·|S_k|·ρ / f_tier, ρ = 1,250, d from Wen's predicted upload position Loc_x + T̂_com·V_k (closed-loop wrap); F2 = e·ρ·b·|S_k|·f² + p·T̂_up; F3 = last reported training loss (never-trained prior = fleet mean of reported losses); F4 = KL(label histogram ‖ uniform 0.1).
- Constraints Eq. 10 with limits rescaled to our workload: T_max, E_max = 90th percentile of F1, F2 over the fleet × trace positions (≈ 34.3 s, 36.4 J; exact values written by the code before any run). If no candidate is feasible, constraints are dropped for that decision.
- Exact enumeration of the non-dominated set, min–max normalisation within it (range 0 → 0), pick the smallest mean normalised value; ties → lower client_id. No exploration.
- Ground truth identical to every arm (GPU training time with jitter, Wen upload with Rayleigh fading, GPU energy). Same local training and FedAsync aggregation as every arm.

## Comparators (existing runs, not rerun)

S1: `random` (ε = 1), `gpu` (ε = 0.2), `location_only`, `cpu`. S2: `random` and `gpu` (ε = 0.2), both Stage 0 off.

## Predictions (v3 rule: same direction in all 3 seeds 4126 / 4127 / 4128; mean ± SD reported; magnitudes not predicted)

- **P15 Position misplaced (mechanism, S1):** wen's mean true upload distance |x_upload − 1000| is larger than location_only's.
- **P16 Time per round (S1):** gpu has lower s/round than wen.
- **P17 Time-to-accuracy (S1):** gpu has lower TTA@80% than wen.
- **P18 Energy (S1):** gpu uses less ground-truth energy to reach 80% than wen.
- **P19 Tier mix (S1):** wen's Thor dispatch share (whole run) is lower than gpu's.
- **P20 Waste (S2):** wen's waste fraction (dispatches) is lower than random's.
- **P21 Time-to-accuracy (S2):** gpu (Stage 0 off) has lower TTA@80% than wen.

## Reported, not predicted

wen vs random on s/round and TTA (S1 and S2): the position gain may be absent (see P15), and the deterministic selector with a static F4 may concentrate participation. wen vs cpu on s/round. TTA@85% and TTA@90% for every comparison above. Participation spread (distinct vehicles, per-vehicle dispatch counts).

Context (not a prediction): Wen predicts the upload position from CPU-scale T̂_com ≈ 20–23 s while true training takes 0.2–2.8 s, so it favours vehicles roughly 400–760 m upstream of the RSU. Its true upload distance is therefore expected to resemble the cpu arm's, not location_only's. Under the CPU energy model Thor (highest f) has the largest F2 and is the only tier cut by E_max.
