# Pre-registered predictions v3.5 — Polaris comparator in S1 (locked 21 Oct 2026, before any `polaris` run)

Adds the `polaris` arm to v3 (`f2bb90a1`), v3.1 (`098e1dd1`), v3.2 (`66a2ed82`), v3.3 (`77e2d7b7`) and v3.4, which are unchanged. Commit this file before the first `polaris` run and record its hash in every `polaris` manifest.

## Arm: `polaris` (S1, Stage 0 off — a no-op in S1)

Kang & Li, "Polaris: Accelerating Asynchronous Federated Learning with Client Selection", IEEE TCC 12(2), 2024. Fidelity rule: **paper first; where the paper is silent, the authors' own Plato code; where the code contradicts the paper, the paper (documented).**

- Sampling probabilities over the eligible pool from P4 (Eq. 23): min_q Σ p_i² G_i² / q_i + A·Σ p_i² q_i τ_i G_i, s.t. Σ q_i = 1, q_i > 0. Solution q_i = √(f1_i / (f2_i + μ)), f1_i = p_i² G_i², f2_i = A·p_i² τ_i G_i, μ by bisection so that Σ q_i = 1.
- p_i = n_i / Σ_fleet n (paper). A = 1. G_i = L2 norm of the client's model update on conv layers, unseen clients α·(mean of the latest aggregation's norms), α = 10, initial G = 0.5; τ_i = observed staleness + 0.1, initial 0.01 (Plato defaults).
- One draw per free slot from q renormalised over the remaining pool; dedicated RNG keyed by (seed, round, k).
- FedAsync aggregation, local training and ground truth identical to every arm.

## Comparators (existing runs, seed 4126 first)

S1: random, gpu (Ours), observed, location_only, wen, wen_fresh.

## Predictions (v3 rule: same direction in all 3 seeds; seed 4126 alone gives a direction only)

- **P34:** gpu has lower s/round than polaris.
- **P35:** gpu has lower TTA@80% than polaris.
- **P36:** gpu uses less ground-truth energy to reach 80% than polaris.
- **P37:** polaris has a lower whole-run dispatch Gini than gpu.

## Reported, not predicted

polaris vs random (s/round, TTA@80 / 85 / 90%); polaris vs observed (same τ source, different selection rule); polaris vs wen_fresh; whether polaris reaches 90%; tier share; mean upload distance; q statistics (max q, effective number of candidates 1 / Σ q²).

Context (not a prediction): the paper formulation spreads q over most of the pool, and probability-proportional sampling was ≈ random in this comm-bound regime (main runs: gpu_prop s/round 3.721 vs random 3.773). Polaris has no notion of position, tier or upload time — only observed staleness — so its s/round is expected near random; any gain should come from G_i (fewer rounds).
