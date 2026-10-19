# ADR-0017: Wen-faithful selector, async adaptation

Status: accepted on 18 Oct 2026 (Prompt E, decisions W1–W10). Implemented in `wen_env/wen_faithful.py`.

## Context

The comparator is Wen et al. (Information Sciences 664, 2024), RA-MaOVSM: a synchronous FL scheme that selects vehicles with a many-objective search (RVEA-PDCG). It scores four objectives (time, energy, training loss, label divergence) under the time and energy constraints of Eq. 10. Study 1 compares our selector, Wen's selector and random selection inside the same asynchronous FedAsync environment. None of our components appear inside Wen or random.

## Decision

| # | Decision |
|---|---|
| W1 | Async: Wen's objectives and rule pick **one vehicle per free slot**. There are no sync rounds. |
| W2 | N = 30 (15 Xavier / 10 Orin / 5 Thor, the existing fleet), C = 3. |
| W3 | **Exact enumeration replaces RVEA-PDCG** (see below). Normalisation is min–max within the non-dominated set, per objective; an objective with range 0 maps to 0, so a single member maps to all zeros. Pick = smallest mean of the four normalised values; ties go to the lower `client_id`. Deterministic: no ε, no RNG. |
| W4 | F1 = T̂_com + T̂_up with T̂_com = b·\|S_k\|·ρ / f_tier, ρ = 1,250. F2 = e·ρ·b·\|S_k\|·f² + p·T̂_up. F3 = the last training loss the vehicle reported (a never-trained vehicle gets the mean of all losses reported so far, or a constant before any report). F4 = KL(P_k ‖ uniform 0.1). |
| W5 | Eq. 10 with **rescaled** limits (see below). If no candidate is feasible, both constraints are dropped for that decision only. |
| W6 | Ground truth is the same for every arm: GPU training time with jitter, Wen Eq. 5–6 upload with Rayleigh fading, GPU energy. Wen's *estimates* are CPU-based. |
| W7 | S2: eligibility = idle and inside [400, 1600] at dispatch; `coverage_gated` upload; **no Stage 0**. |
| W8 | Metrics: TTA at 80 / 85 / 90%, s/round, energy to 80%, waste. Stop rule: `target_accuracy = 0.9`, at most 4,000 rounds, identical to the main runs. |
| W9 | The random baseline is the existing async `random` runs; they are not rerun. |
| W10 | Same local training and the same FedAsync aggregation as every arm. Only the selector differs. |

### Why exact enumeration replaces RVEA-PDCG

RVEA-PDCG is a population-based evolutionary search over selections, needed when the decision space is combinatorial: Wen picks a subset per synchronous round.

In the async adaptation each decision chooses **one** vehicle from at most 30 candidates. The exact Pareto set can therefore be computed directly in O(n²), and Wen's own rule for choosing one solution from that set (§5.3.2: smallest normalised mean of the objectives) applies unchanged. This removes the optimiser's stochasticity and its settings without changing what Wen's objectives and selection rule prefer.

### Why the constraint limits are rescaled

Wen's published limits (T_max = 30 s, E_max = 20 J) assume Wen's own workload.

Under ours, the CPU model gives T̂_com ≈ 19.7–23.3 s and E_com ≈ 24.8–34.6 J **before** any upload. E_max = 20 J would therefore reject every vehicle, and T_max = 30 s would reject almost every Xavier and Orin vehicle once upload is added.

The limits are instead the **90th percentiles** of F1 and F2 over every vehicle × every 1 s trace position, t = 1–30,000 s (`wen_env/wen_limits.py`):

- **T_max = 34.334345905647005 s**
- **E_max = 36.423846640394224 J**

| Tier | F1 range (s) | F2 range (J) | Cut by T only | Cut by E only | Total cut |
|---|---|---|---|---|---|
| Xavier | 28.307–34.728 | 25.792–27.076 | 12.1% | 0.0% | 12.1% |
| Orin | 28.293–34.728 | 25.777–27.076 | 11.9% | 0.0% | 11.9% |
| Thor | 24.714–31.135 | 35.603–36.888 | 0.0% | 60.0% | 60.0% |
| Fleet | 24.714–34.728 | 25.777–36.888 | 10.0% | 10.0% | 20.0% |

As planned: T_max cuts only Xavier and Orin, E_max cuts only Thor, and the total cut is about 20%. Thor is not fully excluded, and both limits cut some candidates. The full table, including positions inside [400, 1600], is in `wen_env/evidence/wen_limits.md`.

## Known limitation

Wen predicts the upload position from its CPU-scale T̂_com (about 20–23 s, against 0.2–2.8 s of true training): x̂ = Loc_x + T̂_com·V_k. It therefore prefers vehicles about 400–760 m upstream of the RSU, which will in fact upload from about where they are now. This is a property of Wen's model under our workload, kept on purpose. It is the subject of prediction P15.

## Implementation notes

**Code structure.**
- Pure functions: `kl_to_uniform`, `dominates`, `non_dominated`, `normalise`, `decide`.
- `WenFaithfulSelection` wires them to the server.
- F1 and F2 come from `CpuEstimator`, the same object the cpu arm and the limit computation use.

**F3 (training loss).** It is the client's own report, `report.train_loss`: PLATO's average training loss over the last local epoch. Only updates that reached the RSU are counted (`sumo_gates.delivered`).

**F4 (label divergence).** It is computed from each client's label histogram, which a client would report once at registration. That histogram is its own disjoint partition (`wen_partition`), rebuilt on the server from the same seed and labels.

**Side logs per run.**
- `wen_decisions.csv`: one row per decision. It holds the candidate, feasibility and constraint-cut counts (each also split by tier), the fallback flag, `n_pareto`, the pick's F1–F4 and normalised values, and the predicted upload position.
- `wen_candidates.csv`: every scored candidate's F1–F4, with its feasibility, Pareto and pick flags.

**Guard.** `wen_manifest.py` refuses any `wen` run until `docs/predictions_v3_2_addendum.md` is committed. It records that commit (66a2ed82) and T_max / E_max in every `wen` manifest.
