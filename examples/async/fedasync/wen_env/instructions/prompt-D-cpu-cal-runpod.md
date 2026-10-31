# Prompt D — cpu_cal ablation and log-only mechanism tables (RunPod)

Continues from Prompt C (main runs v1 complete: 36 runs, `evidence/main_runs_v1.md`). Same rules: do not edit `plato/` core; keep the ground-truth / reported / estimated separation; locked values only; stop and report on any conflict.

## Step 0 — Lock the predictions first

Copy `predictions_v3_1_addendum.md` (supplied with this prompt) to `docs/predictions_v3_1_addendum.md`, commit, and record the hash. No `cpu_cal` run may start before this commit; extend the `wen_manifest.py` guard to require it for `arm = cpu_cal`.

## Step 1 — Implement `cpu_cal`

- New arm name `cpu_cal`, S1 only. Same code path as `cpu`; the only difference is ρ.
- Compute ρ_cal at fleet build time, not by hand:
  `ρ_cal = mean_i(T_train_tier(i)) / mean_i(b·|S_i| / f_tier(i))`
  using the datasheet training time without jitter (2.7838 / 0.35690 / 0.18559 s at |S| = 1,667; use each client's own |S_i|), b = 24,576, f = 2.2 / 2.2 / 2.6 GHz. Expect ≈ 85. Write the exact value into the config and the run manifest.
- Energy estimate for `cpu_cal` uses the CPU family (Wen Eq. 7 + 8) with ρ_cal, as D2 in Prompt C. Ground truth unchanged.
- ADR (next free number): `cpu_cal` as a CPU-favourable ablation — purpose, formula, and that it uses fleet-mean information unavailable to a deployed system.

## Step 2 — Tests

1. `cpu_cal` with ρ = 1,250 reproduces the `cpu` dispatch log for seed 4126 (30-round check).
2. Mean of T̂_cpu_cal_train over the fleet equals the mean datasheet T_train to 1e-9 relative.
3. Per-tier T̂_cpu_cal_train ≈ 1.58 / 1.58 / 1.34 s (report exact values).
4. No tier blocked at E_max = 600 J.

## Step 3 — Runs

S1 `cpu_cal` × seeds 4126, 4127, 4128. Everything else identical to the main runs, **including the stop rule** (`target_accuracy = 0.9`), so the arm is comparable to the existing 36 runs. Checkpoint cleaner and disk guard as in Prompt C.

## Step 4 — Log-only mechanism tables (no new runs)

From the dispatch logs of every S1 run (all 13 arms incl. `cpu_cal`) and every S2 run, per arm, mean ± SD over 3 seeds:

1. **Energy per dispatch** (ground truth, J) — all dispatches, and split before / after the 80% crossing.
2. **Mean upload distance** d at training finish (m), and the share of dispatches whose training finishes inside x ∈ [400, 1600].
3. **Mean true T_train and mean true T_up per dispatch** (s).
4. **Dispatch share per tier** in rounds 1–500 and in the last 500 rounds of each run (replaces the empty 3,500–4,000 window).
5. **Position-estimate error:** mean |x̂_finish − x_finish| (m) per arm, where x̂ is the arm's own predicted finish position.

## Step 5 — Report

`evidence/cpu_cal_v1.md`:
- the main-runs table (Prompt C Step 6 columns) with `cpu_cal` added as a row;
- the P14 table (P14a–d), per-seed direction and held / not held under the v3 rule;
- the Step 4 tables.

Mechanical only — no conclusions beyond the P14 table.

## Stop conditions

- Step 0 commit missing → no `cpu_cal` run.
- Any Step 2 test fails → stop.
- Online ≠ replayed clock → stop.

## Deliverable message

ρ_cal exact value, per-tier T̂_cpu_cal_train, test results, the `cpu_cal` row, the P14 table, the Step 4 tables, and every deviation.
