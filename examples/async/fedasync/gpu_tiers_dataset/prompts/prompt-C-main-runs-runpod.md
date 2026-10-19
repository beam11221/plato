# Prompt C — Selection rule, energy estimates, and the main runs that do not need Wen's selector (RunPod)

Continues from Prompt B v2 (environment model, `examples/async/fedasync/wen_env/`, ADR-0005…0012, smoke run passed). Same rules: do not edit `plato/` core; keep the ground-truth / reported / estimated separation; any locked value the code cannot honour → stop and report.

This prompt (1) changes the selection rule, (2) fixes the cpu arm's energy estimate, (3) runs the cells of the run plan that do not need Wen's selector. **Wen-faithful and Wen + GPU are not part of this prompt** (rows 4, 5, 11, 12 of the run plan — a later prompt).

The pre-registered predictions are v3 in `claude/roadmap-wen-environment.md` (copy them into `docs/predictions_v3.md` before any full run, commit, and record the commit hash in every run manifest). They are locked: do not edit them after a full run starts.

---

## Decisions locked on 16 Oct 2026

### D1 — Selection rule: ε-greedy, ε = 0.2

Applies identically to every arm that uses our score V_i = G_i / √(1 + τ̂_i): `gpu`, `cpu`, `gpu_oracle`, `observed`, `location_only`.

At each free slot (one pick per decision; C = 3):
1. Candidates = idle vehicles that are eligible (S1: all idle; S2: idle, in coverage, and — if Stage 0 is on — feasible).
2. Draw u ~ U(0,1) from an RNG keyed by `(seed, round, k)` (same `k` convention as the fading key, so a held round does not replay the draw).
3. If u < ε: pick uniformly among the candidates (second draw from the same keyed stream).
4. Else: pick the candidate with the highest V_i; ties go to the lower `client_id`.

`random` is the ε = 1 endpoint and must go through the same code path with ε = 1 (not a separate selector), so that eligibility and RNG handling are identical.

Ablation switches (gpu arm only, S1): `selection = "egreedy"` with `epsilon = 0.0` (pure top-1), and `selection = "proportional"` (the previous behaviour: sample ∝ V_i without replacement). Default for every other cell: `egreedy`, ε = 0.2.

Why (for the ADR): the smoke run showed proportional sampling collapses to random once upload compresses V to ≈ 0.45–0.56 (gpu −1.3% s/round vs random). ε-greedy depends only on the ranking of V, so its selectivity does not shrink with V's spread; ε keeps every candidate's per-decision probability ≥ ε / |candidates|. Precedent for a reserved exploration fraction: Oort (OSDI '21), exploration ε vs exploitation (1 − ε).

### D2 — Energy estimate per arm family (Stage 0 energy check)

Each arm estimates energy with the model its time estimate comes from:
- GPU family (`gpu`, `gpu_oracle`, and `location_only` / `observed` if ever combined with Stage 0): `Ê = P_tier × T̂_train + p × T̂_up`, P = 40 / 75 / 130 W, p = 0.2 W.
- CPU family (`cpu`): Wen Eq. 7 + Eq. 8: `Ê = e · ρ · b · |S_i| · f_tier² + p × T̂_up`, with e = 1e-28, ρ = 1,250, b = 24,576, f = 2.2 / 2.2 / 2.6 GHz → training part ≈ 24.8 / 24.8 / 34.6 J.
- E_max stays 600 J (non-binding by default).
- **Ground-truth energy is unchanged** (GPU model for every arm) — only the estimate the cpu arm uses for its own Stage 0 check changes.

Report the per-tier Ê for both families and confirm no arm is blocked by E_max.

---

## Step 1 — Implement D1, D2 and logging

Add to the per-dispatch log: `selection_rule, epsilon, coin_u, explored (bool), n_candidates, rank_of_pick (1 = top V), V_max, V_min, V_pick, E_hat`.

ADRs (next free numbers): ε-greedy selection rule (with the smoke-run evidence and Oort precedent); per-family energy estimate; disjoint Dirichlet sampler with Sinkhorn balancing (mean TV 0.063 from the Dirichlet draw — flagged in the Prompt B report as undocumented). Also replace the ADR-0005 placeholder with the SUMO ADR text (supplied separately as `ADR-0001-wen-closed-environment.md` from the MacBook; keep a note of the original name).

## Step 2 — Tests

1. ε = 1 reproduces the `random` arm's dispatch log exactly for the same seed.
2. ε = 0 always picks rank 1.
3. Over 10,000 synthetic decisions with 27 candidates, ε = 0.2 gives P(rank 1) = 0.8 + 0.2/27 within ±1%, and every candidate's frequency ≥ 0.2/27 − 0.3%.
4. `proportional` reproduces the Prompt B smoke-run gpu log for seed 4126.
5. cpu `Ê` matches the D2 numbers; no tier is blocked at E_max = 600 J.
6. Ground-truth energy for a cpu-arm dispatch equals the GPU formula.

## Step 3 — Re-smoke (plumbing, 200 rounds, seed 4126)

Cells: S1 × {random, gpu (ε = 0.2), gpu (ε = 0), gpu (proportional), cpu}; S2 × {gpu + Stage 0, cpu + Stage 0}.
Report: s/round, dispatch share per tier, share of explored picks, mean `rank_of_pick` among exploit picks, `V_max − V_min` distribution, cpu + Stage 0 rejection rate. **Do not read accuracy.** This checks the rule works; it is not the result.

## Step 4 — Pilot gate (seed 4126, full budget)

Run S1 `random` and S1 `gpu` (ε = 0.2) for **4,000 rounds**. Then check, and **stop and report if any fails**:

| Gate | Pass condition |
|---|---|
| Plateau | both arms' accuracy flat over the last 500 rounds (slope of a linear fit < 0.5 points per 1,000 rounds) |
| Target | both arms cross 80% before round 3,500 |
| Trace | last dispatch time < 30,000 s (no `TraceEndError`) |
| Clock | online = replayed clock |

If plateau or target fails, do **not** change the round budget, the target or any setting — report the curves and stop. Those are thesis decisions.

## Step 4b — Storage cleaner (start before Step 4, keep running through Step 5)

The pod has a **200 GB** volume; 4,000-round runs fill it with model checkpoints. Run a checkpoint cleaner in the background for the whole of Steps 4–5. If the repo already has the cleaner used in earlier stages, reuse it and say which file; otherwise write `wen_env/tools/ckpt_cleaner.py`.

Rules:
- **Deletes only** model checkpoint files (global and client weights, optimizer state) under the runs' checkpoint directories. Match by path pattern and extension; print the patterns in the report.
- **Never deletes**: logs, dispatch logs, accuracy CSVs, `results/`, manifests, `evidence/`, configs, the fleet, the SUMO bundle, or anything outside the checkpoint directories.
- Does not touch files a running process may still need: keep the **most recent checkpoint of each run that is still running**, and only delete files whose mtime is older than **10 minutes**.
- After a run finishes, delete all of its checkpoints (no final model is needed — evaluation is logged every round).
- Loop every 5 minutes; append each deletion (path, size, time) to `logs/ckpt_cleaner.log`.
- Disk guard: before launching any new run, check free space; if **< 20 GB**, do not launch — wait for the cleaner, and if still < 20 GB after 15 minutes, stop and report.
- Must survive the session: run under `nohup` or `tmux` and record the PID / session name in the report.

Test it once in a dry-run mode (list what would be deleted, delete nothing) on the Step 3 smoke outputs before enabling deletion.

## Step 5 — Main runs, one seed first

**Phase A — seed 4126 only.** Run every cell in the table below with seed 4126 (the two Step 4 runs are reused, not rerun). Then do Step 6 on this single seed and **stop and report**. Do not start other seeds until told to.

**Phase B — seeds 4127, 4128** — only after explicit go-ahead in the planning chat.

Predictions v3 stay locked across both phases: the Phase A results must not lead to any edit of `docs/predictions_v3.md`, any setting, or the cell list.

| # (run plan) | Scenario | Arm | Rule | Stage 0 |
|---|---|---|---|---|
| 1 | S1 | random | ε = 1 | off |
| 2 | S1 | cpu | ε = 0.2 | off |
| 3 | S1 | gpu | ε = 0.2 | off |
| 6 | S1 | observed | ε = 0.2 | off |
| 7 | S1 | gpu_oracle | ε = 0.2 | off |
| 15 | S1 | gpu | ε = 0 | off |
| 16 | S1 | gpu | proportional | off |
| 8 | S2 | random | ε = 1 | off |
| 9 | S2 | gpu | ε = 0.2 | off |
| 10 | S2 | gpu | ε = 0.2 | on |
| 13 | S2 | cpu | ε = 0.2 | on |

11 cells × 3 seeds = 33 runs in total; Phase A is 11 runs (2 already done in Step 4). Runs may execute in parallel: the clock is computed online from simulated quantities, so host load no longer contaminates it — confirm by checking online = replayed for every run.

Each run: 4,000 rounds, `results/<exp_id>/manifest.yaml` with config hash, fleet name, bundle manifest sha256, predictions commit hash, seed.

## Step 6 — Analysis (mechanical, against predictions v3)

Produce `evidence/main_runs_v1.md` (Phase A: `evidence/main_runs_seed4126.md`) with:

**Phase A (one seed):** report every number below for seed 4126 alone, and in the P-table give the direction for that seed with the status **"single seed — not evaluated"**. The held / not-held verdict needs all 3 seeds and is only filled in after Phase B.

1. Per cell, mean ± std over 3 seeds (Phase A: the single value): TTA@80% (online clock), plateau (mean accuracy of the last 200 rounds), s/round whole run, s/round before and after the 80% crossing, energy to reach 80% (ground truth), dispatch share per tier (rounds 1–500, 1,500–2,000, 3,500–4,000), explored-pick share; S2 cells add waste fraction (dispatches, GPU-s, J) and Stage 0 rejection rate.
2. A table P1–P13 (P12 marked "not run") with: the comparison, per-seed direction, held / not held by the v3 rule (same direction in all 3 seeds). No re-wording of predictions.
3. Accuracy-vs-time curves (online clock), one panel per scenario, mean with min–max band.

Do not write conclusions beyond the P-table — interpretation happens in the planning chat.

---

## Stop conditions

- Any Step 2 test fails → stop.
- Any Step 4 gate fails → stop and report; do not adjust budget, target or settings.
- `TraceEndError` in any run → stop that cell, report, continue the others.
- Online ≠ replayed clock in any run → stop all runs.
- E_max blocks any arm → stop (D2 should prevent it).
- Free disk < 20 GB for more than 15 minutes → stop launching runs and report.
- Phase A finished → stop and report (do not start Phase B).

## Deliverable message

Reply with: ADR list, test results, Step 3 re-smoke table, cleaner setup (file, patterns, PID / tmux session, dry-run listing, total space freed), Step 4 gate table with curves, then the Phase A file path and its single-seed P-table, plus every deviation from this prompt.
