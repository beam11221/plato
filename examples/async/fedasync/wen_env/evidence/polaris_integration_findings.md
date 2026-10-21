# Polaris as a host in the Wen environment: findings and open questions

Date: 21 Oct 2026. Status: **read-only study, no code changed.** Input for the next instruction (Polaris host in S1, N = 30, ResNet-18, CIFAR-10).

Sources read:
- `examples/client_selection/polaris/`, the upstream Plato example: `polaris.py`, `polaris_server.py`, `polaris_aggregation_strategy.py`, `polaris_selection_strategy.py`, `polaris_FEMNIST_LeNet5.toml`, `pyproject.toml`.
- Plato docs, "6. Client Selection Algorithms.md", § Polaris.
- `fedasync_polaris_vi_server.py`, `fedasync_polaris_vi_selection.py`, `docs/decisions/ADR-0002`, ADR-0019.
- `wen_env/` (server, gates, estimators, Wen hosts, config) and `gpu_tiers_dataset/prompts/prompt-I-study2-plugin-runpod.md`.

Verification scripts, run in a throwaway venv with cvxopt 1.3.3. Nothing was installed into the project venvs.
- `/tmp/claude-0/-workspace/09fdb1be-bf07-4e46-b50e-8716dce86fe3/scratchpad/scripts/gp_check.py`: a single instance, run with the upstream call exactly as written.
- `/tmp/claude-0/-workspace/09fdb1be-bf07-4e46-b50e-8716dce86fe3/scratchpad/scripts/gp_sweep.py`: 200 S1-like pools.

---

## 1. What upstream Polaris is

**Paper.** Y. Kang and B. Li, "Polaris: Accelerating Asynchronous Federated Learning with Client Selection", IEEE Trans. Cloud Computing 12(2), 446–458, Feb 2024 (citation as given in Plato's docs).

**Example setup** (`polaris_FEMNIST_LeNet5.toml`):

| item | value |
|---|---|
| data | FEMNIST (LEAF, 62 classes), `sampler = "all_inclusive"`: each client keeps its natural per-writer data |
| clients | 200 total, 20 per round, `max_concurrency = 6` |
| model / training | LeNet-5, 5 epochs, batch 32, SGD lr 0.01, momentum 0.9, 150 rounds |
| async | `synchronous = false`, `simulate_wall_time = true`, `minimum_clients_aggregated = 10`, `staleness_bound = 10` |
| client speed | Zipf (s = 1.2), `avg_training_time = 10` |
| Polaris keys | none set, so code defaults apply: α = 10, initial G = 0.5, initial τ = 0.01, β = 1, staleness weight λ = 1 |

**Mechanism.**
1. **Aggregation** (`PolarisAggregationStrategy`) is FedAvg over weight deltas, weighted by sample count; it is not FedAsync mixing. After each aggregation:
   - G_i = L2 norm of client i's delta over the layers whose name contains `conv`.
   - Clients never seen get the proxy G = α × (mean norm of this batch).
2. **Reports** (`on_reports_received`): τ_i ← staleness + 0.1, and w_i ← n_i / Σn over **this batch**.
3. **Selection** (`_calculate_selection_probability`): a geometric program (GP) solved with `cvxopt.solvers.gp`. It is meant to choose p to minimise

   Σ_i β·w_i²·G_i² / p_i  +  λ·Σ_i w_i²·τ_i²·G_i·p_i,  with Σ p_i = 1,

   where the first term is the variance and the second the staleness penalty. Then k clients are sampled without replacement with `np.random.choice(pool, k, p)`.

---

## 2. Findings

### F1. The code does not solve the paper's GP (verified)
- `cvxopt.solvers.gp` works in log variables x = log p and returns x, not p.
- The upstream constraints are applied to x, not to p:
  - `G = −I, h = 0` gives x ≥ 0 (`polaris_selection_strategy.py:190`);
  - `A = 1ᵀ, b = 1` gives Σx = 1 (`:193`).
- The code then uses x / Σx as the probabilities (`:205–206`).
- So what it actually minimises is Σ f1_i·e^{−x_i} + f2_i·e^{x_i} over the simplex in x. That is a different problem.

Effect, measured on 200 random S1-like pools (pool of 20–27; G ∈ [0.5, 5]; τ ∈ [0.1, 10]; w = 1/30):

| | max p (median) | share of pools with max p > 0.9 | clients with p > 1 % (median) |
|---|---|---|---|
| upstream code | 0.657 | 0.17 | 2 |
| paper formulation (Σp = 1, solved by KKT bisection) | 0.075 | 0.00 | 23 |

- The upstream rule puts most of the mass on 2–3 clients. In one 6-client instance it gave x = [2e-8, 2e-8, 1e-7, 3e-8, 4e-9, 0.99999], where the paper's optimum is [0.21, 0.12, 0.07, 0.06, 0.23, 0.31].
- When the states are identical (the start of a run), both give uniform p.
- The paper formulation has a closed-form-plus-bisection solution, p_i ∝ √(f1_i / (f2_i + μ)), so it needs no GP solver at all.

### F2. MOSEK is never used (verified)
- `solvers.gp(..., solver="mosek")` (`:201`): cvxopt's `gp` accepts `**kwargs` but reads only `options`, so `solver` is silently ignored and cvxopt's own interior-point `cp` solver always runs.
- MOSEK is only an import gate (`_ensure_solver_available`); no licence is ever exercised.

### F3. The sampling RNG is not the one whose state is saved
- The code saves and restores the state of `random` (`:97`) but samples with `np.random.choice` (`:99`).
- The determinism of the selection therefore depends on the global NumPy state, which is shared with everything else in the server process.

### F4. Aggregation weights are batch-relative and never fleet-normalised
- w_i = n_i / Σn is computed over the reporting batch only, and only those entries are overwritten (`:133`). Clients that never report keep 1/N.
- Upstream with `minimum_clients_aggregated = 10`, a typical batch of 10 gives w ≈ 0.1.
- In the Wen environment a batch is usually one update, so the latest reporter gets w = 1 while the others hold 1/30 or an old value. Its f1 and f2 are then scaled by up to 900× relative to its peers.
- Effect on concentration in the same sweep: the upstream rule is unchanged (max p median 0.650); the paper rule goes from 0.075 to 0.286.

### F5. G_i is computed on conv layers only
- The filter is `"conv" in layer_name` (`polaris_aggregation_strategy.py:85`).
- For Plato's ResNet-18 this keeps `conv1/conv2` and the stem conv, and drops `shortcut.*` (the 1×1 projection convs), BN and `linear`.
- Our port (`fedasync_polaris_vi_server.py`) uses the same filter, but on (client weights − baseline weights) rather than on Plato's deltas. The quantity is the same.

### F6. Smaller upstream issues
- **Wrong citation.** `polaris_server.py:6` cites "L. Yang, W. Wang, V. Smith, 2020"; the docs cite Kang & Li, TCC 2024.
- **Unread TOML keys.** `step_window`, `penalty`, `exploration_factor` and `desired_duration` are not read by any Polaris code (they look like Oort keys); `result_path` misspells "polairs2".
- **+0.1 offset.** The stored staleness is τ + 0.1. Our port drops the offset (ADR-0002).
- **Python version.** `pyproject.toml` requires Python ≥ 3.13. The project `.venv` is 3.13.8; `.venv_m2` is 3.12.
- **Not installed.** cvxopt and mosek are in neither `.venv` nor `.venv_m2`. Under F1 option (a), neither would be needed.

### F7. What this repository already takes from Polaris
- **G_i tracking:** `FedAsyncPolarisAggregationStrategy`, the same formula and α proxy, wrapped around FedAsync sequential mixing.
- **Observed-τ estimator:** the `observed` arm (ADR-0002), without +0.1.
- **The score:** **V_i = G_i / √(1 + τ̂)** (`fedasync_polaris_vi_selection.py:324`), used as the scorer under `WenSelector` ε-greedy (ε = 0.2). This is not Polaris's GP.
- **α is locked at 1** for every host other than `random` (`wen_config.py:77`); upstream uses 10.
- So no arm today is "Polaris as proposed". ADR-0002 already records that the GP was out of scope at that time.

### F8. Where a Polaris host would sit in the Wen environment
- **Pipeline** (ADR-0019, `wen_server.install_wen`): `WenCoverageGate` (eligibility) → `WenStage0Gate` (optional) → host. Polaris would be a host in the position of `WenSelector` / `WenFreshSelection`.
- **In S1:**
  - eligibility is "idle", so the pool is 30 minus the busy and awaiting-merge clients (≤ 3), i.e. ≥ 27;
  - Stage 0 has no exit boundary, so it is a no-op (verified in Prompt I Step 0c);
  - an aggregation usually merges one update, and a selection call usually fills one free slot (k = 1), except at the first cycle (k = 3).
- **τ in this environment** is PLATO's staleness: the number of merges during the vehicle's whole task, training plus upload (`wen_server.py` docstring).
- **Logs a new host must keep:** `dispatch_log.csv` columns, the clock checks, waste, and `global_version`. Prompt I's conventions also apply:
  - default-off keys;
  - unknown values raise;
  - generated configs differ only in arm keys;
  - plug-in-off tests reproduce the host's dispatch log.

---

## 3. Open questions (decisions for the user)

Recommendations are marked ★. None has been acted on.

**Q1. Which Polaris is the host?**
- (a) ★ The paper's formulation: Σp = 1, p_i ∝ √(f1_i / (f2_i + μ)), μ by bisection. No cvxopt or MOSEK needed. This is the closest to "host run as proposed" (Prompt I principle), and it can be checked against cvxopt on the correct GP in a unit test.
- (b) Upstream code line for line: concentrated p (F1), cvxopt required.
- (c) Both: (a) as the host and (b) as a documented "Plato-code" variant.

**Q2. Aggregation for the Polaris arm.**
- (a) ★ Keep the environment's FedAsync sequential mixing (mixing 0.5, polynomial a = 0.5) for every arm. Polaris then contributes selection and G_i tracking only; the docs state Polaris is compatible with any aggregation.
- (b) The upstream FedAvg over buffered deltas (`minimum_clients_aggregated`). This would change the training dynamics and make the arm incomparable with the other cells.

**Q3. The aggregation weight w_i.** (F4)
- (a) ★ Fleet-static w_i = |S_i| / Σ|S| (≈ 1/30, from reported partition sizes).
- (b) Upstream batch-relative w_i (in this environment the latest reporter gets w = 1).
- (c) Uniform 1/30.

**Q4. Parameters.**
- α: 1 (the current lock) or 10 (upstream)? If 10, the `wen_config.py:77` lock needs a Polaris exception.
- Keep the +0.1 on τ (upstream) or drop it (ADR-0002)?
- β = λ = 1 (upstream default)?
- Initial G = 0.5 and initial τ = 0.01 (upstream defaults)?

**Q5. Sampling and determinism.**
- (a) ★ A dedicated generator `default_rng([seed, round, k, SALT])`, the `WenSelector` convention: one draw per free slot from p renormalised over the remaining candidates. This is equivalent to sequential sampling without replacement, and the determinism test (two identical runs) passes.
- (b) Upstream `np.random.choice(pool, k, replace=False, p)` on the global state (F3).

**Q6. Arms and scope.**
- Is it only `polaris`, in S1? Or also `polaris_gpu`, where τ comes from our estimator (T̂_total → rounds through the live T_round, as the `gpu` arm does) instead of observed staleness: the Study 2b analogue for Polaris.
- `polaris_s0` is a no-op in S1; is S2 in scope later?
- Seeds: 4126 only first, as in Prompt I?
- Comparators: S1 random, gpu, observed, wen_fresh?

**Q7. Logging.**
- A per-decision side log `polaris_decisions.csv`: pool, p vector, μ, G_i, τ_i, w_i, pick, and p of the pick?
- Should the dispatch log's `score` column hold p_i for this arm?

**Q8. The GP across selection calls.**
- Solve once per selection call and reuse p for the extra picks of the same call (the first cycle, k = 3), or re-solve after each pick on the remaining pool? Upstream solves once and samples k at once. Under Q5 (a), "solve once, renormalise" matches that.
