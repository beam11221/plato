# Prompt E — Wen-faithful selector (async adaptation) and Study 1 runs (RunPod)

Continues from Prompt C (main runs v1, 36 runs, `evidence/main_runs_v1.md`) and Prompt D (`cpu_cal`). Same rules: do not edit `plato/` core; keep the ground-truth / reported / estimated separation; locked values only; any locked value the code cannot honour → stop and report.

This prompt implements the `wen` arm (Wen et al., Information Sciences 664, 2024, RA-MaOVSM) inside our async FedAsync setup and runs Study 1: **Ours vs Wen-faithful vs random**, with none of our components inside Wen or random. **Not in this prompt:** Stage 0 for random or Wen (Study 2), Wen + GPU transplant.

If Prompt D runs are still in progress, do not disturb them; share the disk guard and checkpoint cleaner.

---

## Decisions locked on 18 Oct 2026 (W1–W10)

| # | Decision |
|---|---|
| W1 | Async: Wen's objectives and selection rule are used to pick one vehicle per free slot in FedAsync. No sync rounds. |
| W2 | N = 30 (15 Xavier / 10 Orin / 5 Thor, existing fleet), C = 3. |
| W3 | Wen's optimiser (RVEA-PDCG) is replaced by **exact enumeration**: the candidate set is ≤ 30, so the exact non-dominated set is computed directly. Normalisation: **min–max within the non-dominated set**, per objective; if an objective's range is 0, its normalised value is 0 for every member; a single-member set is picked directly. Pick = smallest mean of the 4 normalised values; ties → lower `client_id`. Deterministic — **no ε, no RNG**. |
| W4 | Objectives (all minimised), per idle eligible candidate k — see Step 2. |
| W5 | Constraints Eq. 10 with limits rescaled to our workload: T_max, E_max = **90th percentile** of F1, F2 (see Step 1). Fallback: if no candidate is feasible, ignore both constraints for that decision only. |
| W6 | Ground truth = the same environment truth as every arm (GPU training time with jitter, Wen Eq. 5–6 upload with Rayleigh |χ|² ~ Exp(1), GPU energy P_tier·T_train + p·T_up). Wen's **estimates** are CPU-based as below. |
| W7 | S2: eligibility = idle and inside coverage [400, 1600] at dispatch (environment rule, same as every arm); `coverage_gated` upload (out-of-coverage finish = wasted dispatch). **No Stage 0.** |
| W8 | Metrics: TTA at 80%, 85%, 90% (online clock), s/round, energy to 80% (ground truth), waste (S2). Stop rule = `target_accuracy = 0.9`, max 4,000 rounds — identical to the 36 main runs. |
| W9 | Random baseline = existing async `random` runs (S1, S2 Stage 0 off, 3 seeds). Not rerun. |
| W10 | Same local training (ResNet-18, E = 1, same batch and lr) and the same FedAsync aggregation (staleness mixing, same parameters) as every arm. Only the selector differs. |

---

## Step 0 — Lock the predictions

Copy `predictions_v3_2_addendum.md` (supplied with this prompt) to `docs/predictions_v3_2_addendum.md`, commit, record the hash. Extend the `wen_manifest.py` guard: no run with `arm = wen` may start without this commit.

## Step 1 — Constraint limits (W5), computed once before any run

1. Population: every vehicle i (all 30) × every trace timestamp t at 1 s steps over the first 30,000 s of the SUMO bundle (`bundle_wen_closed_v1`).
2. For each (i, t), with **estimates only** (|χ|² = 1):
   - T̂_com,i = b·|S_i|·ρ / f_tier(i), b = 24,576, ρ = 1,250, f = 2.2 / 2.2 / 2.6 GHz, |S_i| = the client's own sample count.
   - Wen's predicted upload position x̂ = Loc_x(i, t) + T̂_com,i · V_i, wrapped on the 2,000 m closed loop with the **same wrap convention the existing gpu-arm x̂ code uses** (state which function you reuse).
   - d = √((x̂ − 1000)² + 10²); T̂_up from Wen Eq. 5–6 (B = 2 MHz, p = 23 dBm, N = −114 dBm, φ = 3, D = 357,566,784 bits).
   - F1 = T̂_com,i + T̂_up; F2 = e·ρ·b·|S_i|·f² + p·T̂_up, e = 1e-28, p = 0.2 W.
3. T_max = 90th percentile of F1, E_max = 90th percentile of F2 over all (i, t). Write the exact values into the config and every `wen` manifest. The same limits are used in S1 and S2.
4. Report a per-tier table (Xavier / Orin / Thor and fleet): F1 range, F2 range, share cut by T_max only, by E_max only, by both, and total cut — over all positions, and separately over positions inside [400, 1600].
5. Sanity: planning estimate (uniform upload distance 0–1,000 m) ≈ 34.3 s and ≈ 36.4 J; T_max cuts only Xavier/Orin, E_max cuts only Thor; total cut ≈ 20%. The **rule** (q = 90) is locked, not these numbers: small differences are expected and are not a reason to stop. Stop only if the result changes the qualitative picture — E_max lands where it excludes **every** Thor position (Thor F2 range lies entirely above E_max), or either limit cuts no candidate at all.

## Step 2 — Implement the `wen` selector

New module under `wen_env/` (e.g. `wen_selector.py`); register `arm = wen`. At each free slot (C = 3; at the start fill the slots one at a time, removing each picked vehicle before the next pick):

1. Candidates = idle and eligible vehicles (S1: all idle; S2: idle and inside coverage).
2. For each candidate k, with values available at decision time:
   - **F1** = T̂_com,k + T̂_up,k, as in Step 1, using the vehicle's position at the decision time.
   - **F2** = e·ρ·b·|S_k|·f_k² + p·T̂_up,k.
   - **F3** = the last training loss vehicle k **reported** (mean training loss of its last local epoch, the value the client sends back). Lower is preferred (Wen). A never-trained vehicle gets the mean of the losses reported so far by the fleet; if no vehicle has reported yet, F3 is the same constant for every candidate.
   - **F4** = KL(P_k ‖ Q) = Σ_c P_k(c)·ln(P_k(c) / 0.1), P_k = vehicle k's label histogram / |S_k| (reported once at registration), Q = CIFAR-10 test distribution (uniform 0.1). 0·ln 0 = 0; no smoothing.
3. Feasible = F1 ≤ T_max and F2 ≤ E_max. If the feasible set is empty → use all candidates and set `fallback = true`.
4. Non-dominated set over (F1, F2, F3, F4), all minimised (k dominates j if ≤ on all four and < on at least one).
5. Min–max normalise each objective within the non-dominated set (range 0 → 0); score = mean of the four; pick the minimum; ties → lower `client_id`.

Wen's estimates never read ground truth: no true fading, no true training time, no GPU data. F3 and F4 come only from what the client reports.

**Per-decision log** (in addition to the existing dispatch-log columns): `arm, n_candidates, n_feasible, n_cut_T_only, n_cut_E_only, n_cut_both` (each also split by tier), `fallback, n_pareto, pick_rank_in_pareto` (rank by score), the pick's `F1..F4` and normalised values, `F1..F4` of **every** candidate (a side file per run is fine), Wen's predicted `x_hat_upload` for the pick, and the pick's true upload position.

ADR (next free number): "Wen-faithful selector, async adaptation" — W1–W10, why exact enumeration replaces RVEA-PDCG, why the constraint limits are rescaled (Wen's 30 s / 20 J reject every vehicle under our workload; Step 1 table as evidence), and the known limitation that Wen's position prediction uses CPU-scale T̂_com.

## Step 3 — Tests

1. F1, F2 for one vehicle per tier at three positions (x = 200, 1000, 1700) match a hand calculation to 1e-9 relative; print the table.
2. F4 for three clients matches a hand calculation; the fleet's F4 range is reported.
3. Non-dominated filter on a synthetic 10-point case with a known answer; normalisation edge cases (range 0, single member).
4. Determinism: two 100-round runs with seed 4126 give identical dispatch logs.
5. F3: a never-trained vehicle gets the fleet-mean prior; before any report all F3 are equal and normalise to 0.
6. Ground-truth energy and time for a `wen` dispatch equal the GPU formulas (same as every arm).
7. No estimate path reads ground truth (grep / unit test on the call graph, as in earlier prompts).

## Step 4 — Smoke (plumbing, 200 rounds, seed 4126, S1 and S2)

Report: s/round, dispatch share per tier, fallback rate, constraint cuts per tier, distribution of `n_pareto`, number of distinct vehicles dispatched, mean |x̂_upload − x_upload|, mean true upload distance. **Do not read accuracy.**

## Step 5 — Runs

| Scenario | Arm | Stage 0 | Seeds |
|---|---|---|---|
| S1 | wen | off | 4126, 4127, 4128 |
| S2 | wen | off | 4126, 4127, 4128 |

6 runs, may run in parallel. Each: 4,000 rounds max, stop at 90%, `results/<exp_id>/manifest.yaml` with config hash, fleet name, bundle sha256, predictions v3.2 commit hash, T_max / E_max, seed. Checkpoint cleaner and disk guard (< 20 GB → do not launch) as in Prompt C Step 4b.

## Step 6 — TTA@85% / @90% for existing runs (log-only, no new runs)

From the accuracy logs, compute TTA@80%, @85%, @90% (online clock) for every existing run: the 36 main runs (Prompt C) and `cpu_cal` (if finished). A run that did not reach a threshold → report "not reached" with its last accuracy.

## Step 7 — Report

`evidence/wen_faithful_v1.md`:

1. **Study 1 table**, per scenario, mean ± SD over 3 seeds — S1 rows: random, gpu (ε = 0.2), wen, plus location_only and cpu for reference; S2 rows: random, gpu (ε = 0.2, Stage 0 off), wen. Columns: TTA@80 / 85 / 90%, s/round (whole run, before / after 80%), energy to 80%, dispatch share per tier (rounds 1–500 and last 500 rounds), waste fraction (dispatches, GPU-s, J; S2), mean true upload distance, mean |x̂ − x| at upload.
2. **Constraint table** (realised over all `wen` decisions, per scenario): share of candidates cut by T only / E only / both per tier, fallback rate, mean `n_feasible`, mean `n_pareto`.
3. **Participation**: per-vehicle dispatch counts for wen vs random vs gpu (min, max, number of vehicles with 0 dispatches, Gini).
4. **P-table P15–P21**: comparison, per-seed direction, held / not held by the v3 rule. No re-wording.
5. Accuracy-vs-time curves (online clock), one panel per scenario, random / gpu / wen, mean with min–max band.
6. Step 6 TTA table for all existing runs.

Mechanical only — no conclusions beyond the P-table.

## Stop conditions

- Step 0 commit missing → no `wen` run.
- Any Step 3 test fails → stop.
- Step 1 qualitative check fails (Thor fully excluded by E_max, or a limit cuts nothing) → report and wait (do not run).
- Online ≠ replayed clock in any run → stop all runs.
- `TraceEndError` → stop that run, report, continue the others.
- Free disk < 20 GB for > 15 minutes → stop launching and report.

## Deliverable message

T_max / E_max exact values and the Step 1 per-tier table, test results, smoke table, ADR number, the Study 1 table, the constraint and participation tables, the P15–P21 table, the Step 6 TTA table, and every deviation from this prompt.
