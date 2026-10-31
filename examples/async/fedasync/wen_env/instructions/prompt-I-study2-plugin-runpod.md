# Prompt I — Study 2: plug-in study, seed 4126 (RunPod)

Continues from Prompt H (`evidence/wen_fresh_v1.md`, seed 4126) and the audit (`audit_bundle_66a2ed82_20261018.zip`). Same rules: do not edit `plato/` core; keep the ground-truth / reported / estimated separation; locked values only; stop and report on any conflict.

**Question:** are our two components plug-ins that help other selectors — (a) Stage 0, (b) the GPU-aware estimator — when inserted into hosts that are otherwise run **as proposed**? Hosts: `random` and `wen_fresh`. Do not modify a host's own logic.

**Scope of this prompt: seed 4126 only.** Seeds 4127 / 4128, and `wen_fresh` seeds 4127 / 4128 for Study 1b, come later on explicit go-ahead.

---

## Decisions locked on 19 Oct 2026

| # | Decision |
|---|---|
| S2-1 | Wen host = `wen_fresh` (fresh F3, Prompt H). |
| S2-2a | GPU plug-in replaces only Wen's F1 / F2: F1 = T̂_train(GPU) + T̂_up with x̂ = x + V_k·T̂_train(GPU); F2 = P_tier·T̂_train + p·T̂_up (P = 40 / 75 / 130 W, p = 0.2 W). F3 (fresh), F4, Pareto, min–max, pick rule, tie-break, determinism unchanged. |
| S2-2b | Constraint limits under the GPU plug-in: re-apply the W5 rule — T_max, E_max = 90th percentile of the **new** F1, F2 over all vehicles × trace samples (planning estimate ≈ 13.57 s, ≈ 113.5 J; T_max cuts only far Xavier, E_max cuts only far Xavier). The CPU host keeps 34.334 s / 36.424 J. |
| S2-2c | Wen-sys (F1 + F2 only) is **not** in this prompt (optional, later). |
| S2-3 | Pipeline, every cell, every scenario: **(1) eligibility → (2) Stage 0 (if on) → (3) host's scoring and constraints → (4) host's selection.** Eligibility is the environment rule, applied to every arm: the vehicle is idle, and in S2 it is inside coverage [400, 1600] at dispatch time. Stage 0 never sees a non-eligible vehicle. |
| S2-4 | Stage 0 always uses **our** estimator, whatever the host: T̂ = T̂_train(GPU) + T̂_up (|χ|² = 1, at the predicted finish position), residence (1,600 − x)/V_k ≥ T̂, Ê = P_tier·T̂_train + p·T̂_up ≤ E_max = 600 J. Fallback when Stage 0 rejects every eligible candidate: the same rule the existing `gpu + Stage 0` arm uses (state which). |
| Naming | "Ours" = gpu + Stage 0 in every scenario. Stage 0 is a no-op in S1 (coverage is the whole loop) — verified by Step 0c, not assumed. |

---

## Step 0 — Checks before any new code

**0a. Eligibility in every existing S2 run.** From the dispatch logs of every S2 run so far (random, gpu, gpu + Stage 0, cpu + Stage 0, wen, wen_fresh; all seeds), report per run: number of dispatches, number with x_dispatch outside [400, 1600], number dispatched to a vehicle that was not idle. Expected: 0 and 0. Any non-zero → stop and report.

**0b. Audit-G housekeeping** (skip any item already done; say which, with commit hash):
1. Commit all run-path code (`wen_env/`, `gpu_tiers*/`, `sumo_*`, configs); add a source-tree hash to every manifest from now on.
2. 200-round replay with the current code for one existing run per arm family (S1 gpu, S1 cpu, S2 gpu_stage0, S1 cpu_cal, S1 wen, S1 wen_fresh), seed 4126: dispatch-log rows identical. Report any mismatch and stop.
3. Add `tests/test_wen_separation.py` (import-graph and estimator call-graph check: no estimator or selector — including `wen_fresh`'s fresh-loss path and Stage 0 — reads ground-truth fields).

**0c. Stage 0 is a no-op in S1.** Run S1 gpu with Stage 0 on for 200 rounds, seed 4126: Stage 0 rejection count = 0, dispatch log identical to S1 gpu (Stage 0 off) for the same rounds.

## Step 1 — Lock the predictions

Copy `predictions_v3_4_addendum.md` (supplied) to `docs/predictions_v3_4_addendum.md`, commit, record the hash. Extend the manifest guard: no Study 2 run without it.

## Step 2 — Implement

1. **Stage 0 as a plug-in** usable by any host (`random`, `wen_fresh`): a filter between eligibility and the host, per S2-3 / S2-4. Reuse the existing Stage 0 code; do not duplicate the estimator.
2. **GPU plug-in for Wen**: an option on `wen_fresh` that swaps F1 / F2 to the GPU family (S2-2a).
3. **GPU limits**: compute T_max / E_max with the GPU F1 / F2 exactly as Prompt E Step 1 (all 30 vehicles × trace samples at 1 s over the first 30,000 s, estimates only, same wrap). Write the values into the config and manifests, and report the per-tier cut table (all positions and positions inside [400, 1600]).
4. Arm names: `random_s0`, `wen_fresh_s0`, `wen_fresh_gpu`, `wen_fresh_gpu_s0`.
5. Per-decision log additions: `n_eligible`, `n_stage0_rejected` (by reason: time / energy, and by tier), `stage0_fallback`, and for the GPU plug-in the same F1..F4 / constraint columns as `wen`.
6. ADR (next free number): plug-in study — pipeline order, Stage 0 estimator independence from the host, GPU limits by the W5 rule.

**Qualitative check on the GPU limits — stop and report if:** E_max excludes every position of any tier, or either limit cuts no candidate.

## Step 3 — Tests

1. Each plug-in switched off reproduces its host's dispatch log for 100 rounds (seed 4126): `random_s0` with Stage 0 off = `random`; `wen_fresh_gpu` with the plug-in off = `wen_fresh`.
2. Stage 0 inside a host rejects exactly the vehicles the existing `gpu + Stage 0` Stage 0 would reject for the same eligible set (unit test on logged states).
3. Stage 0 never receives a non-eligible vehicle (assertion + test).
4. GPU F1 / F2 for one vehicle per tier at x = 500, 1,000, 1,500 match a hand calculation to 1e-9 relative.
5. Determinism: two 100-round runs of `wen_fresh_gpu`, seed 4126, identical logs.
6. Full suite passes.

## Step 4 — Smoke (200 rounds, seed 4126, all 5 cells)

Per cell: s/round, tier share, Stage 0 rejection rate (by reason and tier), fallback rate, waste (S2), constraint cuts per tier, distinct vehicles, mean |x̂ − x|, mean upload distance. Do not read accuracy.

## Step 5 — Runs (seed 4126 only), in this priority order

| Priority | # | Arm | Scenario | Comparator (existing) |
|---|---|---|---|---|
| 1 | 1 | `random_s0` | S2 | S2 random |
| 1 | 2 | `wen_fresh_s0` | S2 | S2 wen_fresh |
| 2 | 3 | `wen_fresh_gpu` | S1 | S1 wen_fresh |
| 2 | 4 | `wen_fresh_gpu` | S2 | S2 wen_fresh |
| 3 | 5 | `wen_fresh_gpu_s0` | S2 | S2 wen_fresh_gpu, S2 wen_fresh_s0 |

- Launch priority 1 first, then priority 2. **Launch cell 5 last**, after cells 1–4 have finished.
- When cells 1 and 2 finish, write the interim report (Step 6, Study 2a part) and continue with the rest without waiting.
- 4,000 rounds max, stop at 90%, manifests with predictions v3.4 hash, source hash, limits, seed. Checkpoint cleaner and disk guard as before.

## Step 6 — Report

`evidence/study2_plugin_s4126.md` (interim file after cells 1–2, final after cell 5):

1. **Study 2a table (S2):** random vs random_s0; wen_fresh vs wen_fresh_s0; gpu (Stage 0 off) vs gpu + Stage 0 for reference. Columns: TTA@80 / 85 / 90%, s/round, energy to 80%, waste (dispatches / GPU-s / kJ), Stage 0 rejection rate (by reason, by tier), fallback rate, tier share, mean upload distance, dispatch Gini.
2. **Study 2b table (S1 and S2):** wen_fresh vs wen_fresh_gpu, with gpu and random for reference. Same columns plus mean |x̂ − x|, constraint cuts per tier, Thor share.
3. **Cell 5 (S2):** the 2 × 2 for Wen: {CPU, GPU} × {Stage 0 off, on} — TTA@80 / 85 / 90%, waste, s/round, energy.
4. **GPU limits** and the per-tier cut table (Step 2.3), and the realised cuts in the runs.
5. **P26–P33:** comparison, seed 4126 direction, status "single seed — not evaluated". No re-wording.
6. Curves: accuracy vs online clock, one panel per comparison group.
7. Step 0a / 0b / 0c results.

Mechanical only — no conclusions beyond the P-table.

## Stop conditions

- Step 0a finds any non-eligible dispatch → stop.
- Step 0b replay mismatch → stop.
- Step 0c not a no-op → stop.
- Step 1 commit missing → no Study 2 run.
- GPU-limit qualitative check fails → stop before Step 5.
- Any Step 3 test fails → stop.
- Online ≠ replayed clock → stop all runs.
- Free disk < 20 GB for > 15 minutes → stop launching and report.
- Seed 4126 done → stop (no other seeds).

## Deliverable message

Step 0a / 0b / 0c results, GPU limits and per-tier cut table, test results, smoke table, ADR number, the Study 2a / 2b / cell-5 tables, P26–P33 directions, and every deviation.
