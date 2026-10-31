# Prompt J — Polaris as a main comparator in S1 (RunPod)

Continues from `polaris_integration_findings.md` (your read-only study, 21 Oct) and Prompt I. Same rules: do not edit `plato/` core; keep the ground-truth / reported / estimated separation; locked values only; stop and report on any conflict. **Do not disturb Study 2 runs that may still be in progress** (shared disk guard and checkpoint cleaner).

## Why this experiment

The lab asked for Polaris next to random and Wen in **Scenario 1**. Our score V_i = G_i / √(1 + τ̂) is derived from Polaris, so the committee will ask how we differ from Polaris and whether we are better. The question is:

> **How does Polaris, run as its authors proposed, perform in S1 compared with random, Wen and our method?**

It is **not** "can Polaris be made to work well here". Do not tune Polaris, do not add our components to it, and do not fix what the paper specifies even if it looks suboptimal for this environment.

## Fidelity rule (locked)

1. **The paper first** — Kang & Li, IEEE TCC 12(2), 2024 (`Polaris.pdf` in the project files).
2. **Where the paper is silent**, the authors' own Plato code (`examples/client_selection/polaris/`) — same research group.
3. **Where the code contradicts the paper**, follow the paper and record the difference in the ADR.

Every value below is labelled with its source. If you find a place where this prompt contradicts the paper, stop and report — do not choose silently.

---

## Decisions locked on 21 Oct 2026

| # | Decision | Source |
|---|---|---|
| PL1 | Host = the paper's problem P4 (Eq. 23), solved exactly (below). The Plato-code solver (your F1: constraints applied to log-variables) is **not** the host. | paper |
| PL2 | Aggregation = the environment's FedAsync sequential mixing, same parameters as every arm. Polaris contributes selection and G_i tracking only. | paper ("plug-and-play … combined with various aggregation methods") |
| PL3 | p_i = n_i / Σ_{all 30} n_j, fleet-static (≈ 1/30), from reported partition sizes. Not batch-relative (your F4). | paper (§2: Σ_{i=1}^{N} p_i = 1) |
| PL4 | A (Plato λ) = 1; α = 10; initial G = 0.5; initial τ = 0.01; stored τ = staleness + 0.1; β = 1. | code defaults (paper silent / "not sensitive 0.01–10") |
| PL4b | Staleness enters the objective **linearly**: A·p_i²·q_i·τ_i·G_i. If the Plato code uses τ², record it as a code-vs-paper difference; the host uses the paper's linear term. | paper |
| PL5 | Dedicated RNG `default_rng([seed, round, k, SALT_POLARIS])`; one draw per free slot from q renormalised over the remaining candidates. | ours (determinism) |
| PL6 | Arm `polaris`, S1, seed 4126 first. Comparators: S1 random, gpu, observed, location_only, wen, wen_fresh (existing runs). | lab request |
| PL7 | Per-decision side log `polaris_decisions.csv`; dispatch-log `score` = q_i of the pick. | ours |
| PL8 | Solve once per selection call; for extra picks in the same call (first cycle, k = 3) renormalise q over the remaining pool. | code |
| — | Exception to the α = 1 lock (`wen_config.py:77`) for `polaris` only. | PL4 |

## The selection problem, exactly

At each selection call, over the **eligible pool** P (S1: idle vehicles):

- f1_i = p_i² · G_i²
- f2_i = A · p_i² · τ_i · G_i
- KKT of min Σ f1_i / q_i + Σ f2_i q_i + μ(Σ q_i − 1): q_i(μ) = √(f1_i / (f2_i + μ)), with μ > −min_i f2_i.
- Find μ by bisection so that Σ_{i∈P} q_i(μ) = 1 (Σ q is strictly decreasing in μ). Tolerance |Σq − 1| < 1e-12.
- Edge cases: G_i = 0 → q_i = 0, the vehicle is never sampled in that call (log it); all f1 and f2 equal → uniform q.
- The paper solves over all N clients; in async only the idle pool can be dispatched. Solve over the pool (code does the same) — record this in the ADR.

## G_i and τ_i (what the server observes)

- **G_i**: L2 norm of the client's model update (client weights − the global weights it started from) over layers whose name contains `conv` (Plato filter, your F5). Updated each time the client's update is merged. Clients never merged: initial G = 0.5 before the first aggregation; after it, α · mean of the norms in the latest aggregation (Plato rule). Reuse the existing `FedAsyncPolarisAggregationStrategy` G-tracking; do not write a second one.
- **τ_i**: observed staleness of the client's last merged update (PLATO staleness = merges during the vehicle's whole task) + 0.1; initial 0.01. No estimator of time, position or tier is used — Polaris sees none of these.
- Polaris reads only reported or observed quantities. Extend `tests/test_wen_separation.py` to cover the Polaris path.

## How `polaris` differs from arms we already have (write this into the ADR)

| arm | τ source | quality term | selection rule |
|---|---|---|---|
| `observed` (ours, ablation) | observed staleness | G_i, α = 1 | ε-greedy on V = G / √(1 + τ) |
| `gpu` (Ours) | GPU estimator T̂ → τ̂ | G_i, α = 1 | ε-greedy on V |
| `polaris` | observed staleness + 0.1 | G_i, α = 10 | sample from P4's optimal q |

---

## Step 0 — Checks (no code change)

1. Report from the Plato code, with file:line, whether the staleness term uses τ or τ², and how G of unseen clients is set after the first aggregation. If either differs from this prompt, list it; do not stop unless it contradicts the **paper**.
2. Confirm which code path the existing `FedAsyncPolarisAggregationStrategy` uses for G_i and that it can return per-client G to a new selector.
3. Confirm the disk guard and checkpoint cleaner cover a new run alongside any Study 2 run still in progress.

## Step 1 — Lock the predictions

Copy `predictions_v3_5_addendum.md` (supplied) to `docs/predictions_v3_5_addendum.md`, commit, record the hash. Extend the manifest guard: no `arm = polaris` run without it. Manifests also record the source-tree hash (Prompt I Step 0b).

## Step 2 — Implement

1. `PolarisSelection` as a host in the `WenSelector` position: eligibility → (Stage 0, off in S1) → Polaris. New module under `wen_env/`; register `arm = polaris`; default-off keys; unknown values raise.
2. Solver as above (closed form + bisection). No cvxopt or MOSEK dependency for the host.
3. Optional, **implement and unit-test only, do not run**: `polaris_plato` — the Plato-code probability rule reproduced exactly (log-variable constraints), for a later documented variant. Requires cvxopt in a separate venv; skip if it complicates the main venv and say so.
4. Logging:
   - `polaris_decisions.csv` per selection call: round, pool ids, G_i, τ_i, p_i, f1_i, f2_i, μ, q vector, max q, effective pool size 1 / Σ q², picks and their q.
   - dispatch log: `score` = q of the pick; all standard columns, clock checks, `global_version`.
5. ADR (next free number): Polaris host — fidelity rule, PL1–PL8 with sources, the code-vs-paper differences (F1 solver, F4 batch-relative weights, τ vs τ² if found, F3 global RNG), the pool-vs-N choice, and the comparison table above.

## Step 3 — Tests

1. The bisection solution matches a correct general-purpose solve of P4 (e.g. scipy SLSQP on q with Σq = 1, q ≥ 1e-12) on 200 random pools (size 20–30, G ∈ [0.1, 5], τ ∈ [0.01, 20]): max |Δq| < 1e-6, objective within 1e-9 relative.
2. Σ q = 1 and q ≥ 0 on every logged call.
3. Monotonicity on synthetic pools: raising one client's G raises its q; raising its τ lowers its q.
4. Identical states → uniform q.
5. Determinism: two 100-round runs, seed 4126, identical dispatch logs and `polaris_decisions.csv`.
6. Separation test covers the Polaris path.
7. Online = replayed clock on a 100-round run.
8. (If implemented) `polaris_plato` reproduces Plato's `_calculate_selection_probability` output on 10 fixed inputs.

## Step 4 — Smoke (200 rounds, seed 4126, S1)

Report: s/round, dispatch share per tier, distinct vehicles, Gini, mean upload distance, and per 50-round window: median max q, median effective pool size, G range, τ range. **Do not read accuracy.** If the median effective pool size is below 3 for the whole smoke, stop and report (that would mean q is concentrated, which the paper formulation should not do).

## Step 5 — Run

S1 `polaris`, seed 4126, 4,000 rounds max, stop at 90% — identical budget and stop rule to every S1 run. Manifest with predictions v3.5 hash, source hash, seed. Seeds 4127 / 4128 only on explicit go-ahead.

## Step 6 — Report

`evidence/polaris_s1_s4126.md`:

1. **S1 table, seed 4126:** random, gpu (Ours), observed, location_only, wen, wen_fresh, polaris. Columns: TTA@80 / 85 / 90%, s/round (whole run, before / after 80%), energy to 80%, dispatch share per tier (rounds 1–500 and last 500), mean upload distance, Gini, distinct vehicles, image coverage of aggregated updates at r500 / r1000 / end (Prompt F Step 4 definition).
2. **Polaris internals over time:** median max q and effective pool size per 500 rounds; G and τ ranges; correlation (Spearman) of per-vehicle dispatch count with mean G, mean τ, tier, V_i.
3. **P34–P37:** comparison, seed 4126 direction, status "single seed — not evaluated". No re-wording.
4. Curves: accuracy vs online clock, S1, all arms above.
5. Step 0 answers and every code-vs-paper difference found.

Mechanical only — no conclusions beyond the P-table.

## Stop conditions

- Step 1 commit missing → no `polaris` run.
- Any Step 3 test fails → stop.
- Smoke effective pool size < 3 throughout → stop and report.
- A contradiction between this prompt and the paper → stop and report.
- Online ≠ replayed clock → stop.
- Free disk < 20 GB for > 15 minutes → stop launching and report.
- Seed 4126 done → stop.

## Deliverable message

Step 0 answers, ADR number, test results, smoke table, the S1 table, Polaris internals, P34–P37 directions, and every deviation and code-vs-paper difference.
