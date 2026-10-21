# ADR-0020: Polaris as a host in S1

Status: implemented on 21 Oct 2026 (Prompt J, decisions PL1 to PL8). Predictions v3.5 addendum committed as `0f459b68`. **The 4,000-round run waits for the user's decision on difference D6 below.**

## Context

The lab asked for Polaris next to random and Wen in Scenario 1. Our score V_i = G_i / √(1 + τ̂) is derived from Polaris, so the comparison has to be with Polaris **as its authors proposed it**, not with a tuned or patched version.

- Paper: Y. Kang and B. Li, "Polaris: Accelerating Asynchronous Federated Learning with Client Selection", IEEE TCC 12(2), 446–458, 2024. It was read from the authors' copy, `iqua.ece.toronto.edu/papers/ykang-tcc24.pdf`, because no `Polaris.pdf` was found on the pod.
- Code: `examples/client_selection/polaris/` (Plato, same research group).
- Read-only study before this ADR: `wen_env/evidence/polaris_integration_findings.md`.

## Decision

1. **Fidelity rule.**
   - The paper comes first.
   - Where the paper is silent, the authors' Plato code applies.
   - Where the code contradicts the paper, the paper applies, and the difference is recorded below.

2. **Host = problem P4 (Eq. 23), solved exactly (PL1).** At each selection call, over the eligible pool P:
   - f1_i = p_i² G_i², f2_i = A·p_i² τ_i G_i;
   - the KKT condition gives q_i(μ) = √(f1_i / (f2_i + μ)) with μ > −min f2;
   - μ is found by bisection to |Σq − 1| < 1e-12;
   - G_i = 0 gives q_i = 0, and identical states give uniform q.

   No GP solver, cvxopt or MOSEK is involved (`wen_env/wen_polaris.py`).

3. **Aggregation is the environment's FedAsync sequential mixing, the same for every arm (PL2).** The paper presents Polaris as "plug-and-play … combined with various aggregation methods". Polaris contributes the selection and the G_i tracking only.

4. **Inputs.**
   - **p_i (PL3):** n_i / Σ over the 30 clients of n_j, fleet-static (paper §2: Σ_{i=1}^{N} p_i = 1).
   - **A (PL4):** A = 1, the code's λ.
   - **G_i:** the L2 norm, over layers whose name contains `conv`, of the client's update against the global model **it was sent**. This is the paper's Δ_i(ω_{t−τ}) (Eq. 5): "we estimate G_i via the local model updates", §4.
     - Never-merged clients get 0.5 before the first aggregation, and α·(mean norm of the latest aggregation) after it (Eq. 26, with α = 10 from the code).
     - The tracker is the existing `FedAsyncPolarisAggregationStrategy`, given an optional `start_weights` table. `WenPolarisHostServer.customize_server_payload` fills that table with the conv weights of each payload sent. The default `None` leaves every earlier arm unchanged.
   - **τ_i (PL4):** the PLATO staleness of the client's last merged update + 0.1, and 0.01 before its first merge (code). See D6.

   Polaris reads only these observed or reported quantities. It sees no time estimate, position or tier (`tests/test_wen_separation.py`).

5. **Pool, not N.** The paper solves P4 over all N clients. In async FL only idle vehicles can be dispatched, so P4 is solved over the eligible pool, as the Plato code also does.

6. **Sampling (PL5, PL8).**
   - P4 is solved once per selection call.
   - Each free slot gets one draw from q renormalised over the candidates not yet picked. This is sequential sampling without replacement, as in the paper's "sample K clients without replacement".
   - The RNG is `default_rng([seed, round, k, 0x90A715])`.
   - The first round is uniform, as the paper requires ("for the initial round, clients were chosen uniformly"), because all states start identical.

7. **Logs (PL7).**
   - `polaris_decisions.csv` has one row per call: pool, G, τ, p, f1, f2, μ, q, max q, effective pool size 1/Σq², picks and their q.
   - In the dispatch log, `score` holds the q of the pick.

8. **Arm and scope (PL6).**
   - Arm `polaris` runs in S1 only, without Stage 0, with `selection = "polaris"` and `polaris_alpha = 10`. This is the one exception to the α = 1 lock.
   - Seed 4126 runs first.
   - No `polaris` run of any length may start before the v3.5 addendum is committed.

## Code-vs-paper differences

| # | Plato code | Paper | Host uses |
|---|---|---|---|
| D1 | `solvers.gp` constraints applied to the log variables (x ≥ 0, Σx = 1), and x/Σx used as q (`polaris_selection_strategy.py:189–206`). This concentrates q: on S1-like pools the median max q is 0.66. | P4 with Σq = 1 | paper |
| D2 | staleness term τ² (`polaris_selection_strategy.py:169, 178`) | linear τ_i (Eq. 23) | paper |
| D3 | p_i = n_i / Σn over the reporting batch only (`:133`) | Σ_{i=1}^{N} p_i = 1 | paper (fleet-static) |
| D4 | saves and restores `random`'s state but samples with `np.random.choice` (`:97–99`) | — | dedicated keyed RNG |
| D5 | `solver="mosek"` is silently ignored by cvxopt's `gp`; MOSEK is only an import gate | "convex optimization tools, e.g. CVX" | no solver needed |
| D6 | stored τ = staleness + 0.1, initial 0.01 (`:129`) | τ_i(t) ≥ 1, and τ = 1 means no staleness (§3, "Aggregation with staleness") | **code (Prompt J PL4); open, see below** |
| D7 | update norm against the current global at merge time (`polaris_aggregation_strategy.py:85`, via `compute_weight_deltas`); this repo's port does the same | Δ_i(ω_{t−τ}), the update from the client's starting point (Eq. 5) | paper (start-point baseline) |
| D8 | FedAvg over buffered deltas, `minimum_clients_aggregated = 10` | Algorithm 1 is FedBuff, but Polaris is "plug-and-play" with any aggregation | environment FedAsync (PL2) |

**D6 is a contradiction between Prompt J and the paper,** so it was reported rather than resolved here.
- Prompt J labels the +0.1 and the 0.01 "code defaults, paper silent". The paper does define the convention: τ ≥ 1, and a fresh update has τ = 1.
- Measured effect on the 84 S1 states replayed from `S1_observed_s4126` (rounds > 200): q under τ = s + 0.1 and under τ = s + 1 differ by at most TV 3.0e-5. The reason is in the observation below.

**D7** follows the prompt's own definition of G_i ("client weights − the global weights it started from"), which matches the paper. It is implemented inside the existing tracker rather than as a second tracker, which honours the prompt's "reuse" instruction. Plato's code measures against the current global instead.

## Observation (no tuning)

With p_i ≈ 1/30 and A = 1, f1 and f2 are about 1e-3, while μ is about 1.9 (Σ√f1 squared). So μ ≫ f2 and q_i ≈ √f1_i / Σ√f1 = G_i / ΣG.
- On the replayed S1 states, TV(q, G/ΣG) = 1.2e-4.
- The staleness term is therefore almost inactive in this environment. Polaris as proposed behaves here as sampling in proportion to G_i.
- This follows from the paper's P4 with N = 30 and the code's A = 1. It is recorded, not changed.

## How `polaris` differs from the arms we already have

| arm | τ source | quality term | selection rule |
|---|---|---|---|
| `observed` (ours, ablation) | observed staleness | G_i, α = 1 | ε-greedy on V = G / √(1 + τ) |
| `gpu` (Ours) | GPU estimator T̂ → τ̂ | G_i, α = 1 | ε-greedy on V |
| `polaris` | observed staleness + 0.1 | G_i (update from the starting point), α = 10 | sample from P4's optimal q |

## Not done

`polaris_plato`, the Plato-code probability rule reproduced exactly (Prompt J Step 2.3, optional), was not implemented. It needs cvxopt, which is in neither project venv. D1 already documents its behaviour, measured with cvxopt 1.3.3 in a scratch venv (`polaris_integration_findings.md`).
