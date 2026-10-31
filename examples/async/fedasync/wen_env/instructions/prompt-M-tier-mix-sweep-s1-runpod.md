# Prompt M — Mechanism logging + replay (Part A), then tier-mix sweep in Scenario 1 (Part B) (RunPod)

Continues from Prompt K (`evidence/s1_final.md`) and Prompt L (`evidence/s2_final.md`). Same rules: do not edit `plato/` core; ground-truth / reported / estimated separation; locked values only; stop and report on any conflict.

**Order: Part A first (logging, replay, mechanism report). Part B (sweep) starts only after every Part A replay has reproduced its original dispatch log.** The new logging stays on for Part B.

**Principle: change as little as possible.** Logging must be behaviour-neutral (no RNG draw, no change to any decision). Anything not listed here stays exactly as in the S1 runs of Prompt K; if a value is not mentioned, use the S1 default and say so in the report.

---

## Step 0 — Back up before anything else

1. Commit every run-path and reporting file (including `wen_env/study2_report.py`); push the repo to its GitHub remote. Report the commit hash and the remote URL. If there is no remote, say so and wait — do not create one.
2. Create a tarball of `evidence/`, `docs/`, configs, manifests and all run logs (dispatch, merge, decisions, candidates, accuracy CSVs; **no checkpoints, no datasets**). Report path, size, sha256.

---

# Part A — Mechanism logging and replay (S1, M0 fleet, seed 4126)

## Why

The Results chapter must explain *how* each selector decides, with plots from our own logs:
- **Ours:** the trade-off between predicted time τ̂ and data value G_i in V = G / √(1 + τ̂), and what ε-greedy adds.
- **Polaris:** the same G-versus-staleness idea, but with observed staleness and probabilistic sampling (q is almost uniform here).
- **Wen et al.:** which objectives actually decide its picks (F3 own-data loss and F4 KL dominate; F1 speed barely matters; F2 energy acts as a tier filter), and the F3 feedback loop.

Today the `gpu` arm logs only V_max / V_min / V_pick per decision, so these plots cannot be drawn.

## A1 — Logging to add (behaviour-neutral)

Common columns in every candidate file: `run, arm, seed, round, decision_id, slot_k, client, tier, x_dispatch, V_k, picked (bool)`.

**Passive common axes for all three arms.** For every candidate, also log what Ours' scorer *would* see, computed by a separate logger (never fed back to the selector): `tau_hat_ours` (from GPU T̂_train + T̂_up at the predicted finish position, same code as the `gpu` arm), `T_hat_train_gpu, T_hat_up, x_hat_finish_gpu, G_ours` (Ours' G_i tracker with α = 1), `V_ours`. This puts all arms on the same (τ̂, G) axes. For arms that do not track G (Wen), the logger keeps its own passive G tracker fed from merged updates (same formula as Ours).

**Ours (`gpu`) → `gpu_candidates.csv`**, per candidate per decision:
`T_hat_train, T_hat_up, x_hat_finish, d_hat, T_hat_total, T_round_used, tau_hat, G, G_source (measured / prior), n_merged_client, V, rank_by_V, coin_u, explored (bool), is_max_G, is_min_tau_hat`.
Plus **`gpu_g_updates.csv`**, per merge: `round, client, update_norm_conv, G_new, staleness`.

**Polaris (`polaris`) → `polaris_candidates.csv`** (long format, per candidate per call; keep the existing `polaris_decisions.csv`):
`G_i, G_source (seen / alpha-proxy / initial), tau_i (observed + 0.1), p_i, f1_i, f2_i, mu, q_i, rank_by_q, q_renormalised_at_pick, draw_u`.

**Wen et al. (`wen_fresh`) → extend `wen_candidates.csv`** so that each row has:
`F1, F2, F3, F4` (raw), `F3_global_version`, `feasible_T, feasible_E, fallback`, `in_pareto`, `nF1, nF2, nF3, nF4` (normalised within the Pareto set; blank if not a member), `score` (mean of nF), `rank_by_score`, `is_min_F1 … is_min_F4` (among all candidates), `T_hat_com_cpu, x_hat_upload_cpu`.
Plus **`wen_f3_by_version.csv`** if not already produced (`wen_fresh_eval.csv` has F3 for all 30 per version — confirm and reuse).

**Ground truth for picked vehicles** stays in the dispatch log (true T_train, x_finish, T_up, energy). Do not add truth to candidate files.

## A2 — Tests

1. For each of `gpu`, `polaris`, `wen_fresh`: a 100-round run with logging on reproduces the existing seed-4126 dispatch log row for row (and `wen_decisions.csv` / `polaris_decisions.csv`).
2. Separation test (`tests/test_wen_separation.py`) still passes; the passive logger is not imported by any selector.
3. `V == G / sqrt(1 + tau_hat)` on every `gpu_candidates.csv` row; the picked row is rank 1 whenever `explored = false`.
4. Polaris: Σ q = 1 per call; Wen: the picked row has the minimum `score` among `in_pareto` rows.

## A3 — Replay (full runs, seed 4126 only, S1, M0 fleet)

| arm | stop |
|---|---|
| `gpu` (Ours) | same stop rule as the original (90 % or 4,000 rounds) |
| `polaris` | same |
| `wen_fresh` | same (4,000 rounds) |
| `random` *(optional, if capacity: passive τ̂ / G only, for the baseline map)* | same |

Each replay must reproduce its original dispatch log **for the whole run**, and online = replayed clock. Any mismatch → stop and report the first differing row.

## A4 — Mechanism report `evidence/mechanism_s1_s4126.md` with plots

Mechanical; plots saved as PNG next to the report.

**Ours**
1. Candidate map: x = τ̂, y = G, grey = candidates, colour = picked (exploit) / picked (explored); iso-V curves G = V·√(1 + τ̂) for the decision's V_pick; three panels: rounds 1–200, 1,000–1,200, last 200 before 90 %.
2. Share of exploit picks that are the min-τ̂ candidate, the max-G candidate, both, neither — per 500 rounds.
3. Static counterfactual on logged decisions: pick by τ̂ only, by G only, by V (exploit) — pick-change share and dispatch Gini.
4. Per-tier G over time (mean, IQR) and per-tier τ̂ over time; tier share over time.
5. Per-vehicle dispatch count vs mean G and vs mean τ̂ (Spearman).

**Polaris**
6. Candidate map on the same axes (x = passive τ̂_ours, y = G), colour = q_i; plus a second map with x = observed τ_i.
7. q distribution per call vs uniform 1 / |pool|; effective pool size over time; spread (max/min) of G and τ per call.
8. Per-vehicle dispatch count vs mean G, mean τ, mean τ̂_ours (Spearman).

**Wen et al.**
9. Candidate map on the same axes (x = passive τ̂_ours, y = passive G), colour = Wen's score, picked highlighted.
10. Share of picks that are the min of F1 / F2 / F3 / F4, per 500 rounds.
11. F3 of the top-5 vehicles vs the rest over time; F4 rank of the top-10 most-picked vehicles.
12. F2 by tier (box plot) under the CPU model; per-tier constraint cuts.
13. Static counterfactual: drop F1, F2, F3, F4 one at a time; F1 + F2 only; F3 + F4 only — pick-change share and Gini.

**Cross-arm**
14. One figure with three panels (Ours, Polaris, Wen) of the same candidate set axes (τ̂, G) showing where each arm's picks fall; and a table: mean τ̂ and mean G of picked vehicles vs all candidates, per arm.

---

# Part B — Tier-mix sweep (starts after A3 is verified)

## Why

The 15 / 10 / 5 (Xavier / Orin / Thor) mix is our own choice; reviewers asked whether S1 results — especially Wen losing to random — depend on it. Only the tier counts change.

## What changes

| id | Xavier / Orin / Thor |
|---|---|
| M1 | 10 / 10 / 10 |
| M2 | 5 / 10 / 15 |
| M3 | 2 / 8 / 20 |
| T1 | 15 / 0 / 15 |
| T2 | 5 / 0 / 25 |
| T3 | 15 / 15 / 0 |

M0 = the existing 15 / 10 / 5 S1 runs (reference). Tier assignment (locked): contiguous ID blocks in the order Xavier → Orin → Thor, as in M0 (e.g. M2: IDs 1–5 Xavier, 6–15 Orin, 16–30 Thor). A tier with 0 vehicles is absent.

## What stays fixed (S1 defaults)

- Vehicle IDs, per-ID speeds (seed 20261015 in ID order), SUMO bundle, per-ID data partitions (**the data each ID holds is identical to M0**).
- Tier specifications, η = 0.2, jitter, upload and energy models, RSU, 100 % coverage, `always_accept`.
- Ours: `gpu`, ε = 0.2, α = 1, Stage 0 off (no-op in S1). Polaris: exactly Prompt J. Wen et al.: `wen_fresh` exactly Prompt H, with **T_max / E_max recomputed per mix by the W5 rule** (q = 90 over that mix's 30 vehicles × trace samples, Prompt E Step 1 method). Random: ε = 1 path.
- FedAsync, C = 3, 4,000 rounds, stop at 90 %, eval every 10 rounds, metrics as `s1_final.md`.
- **Part A logging stays on** in every sweep run.

## B0 — Fleets and limits

1. One fleet file per mix; report per-ID tier, per-tier count, per-tier mean speed, and a check that each ID's partition hash equals M0's.
2. Wen limits per mix + per-tier cut table. If a tier present in the mix is cut at **every** position, report and wait.

## B1 — Lock the predictions

Copy `predictions_v3_7_addendum.md` to `docs/`, commit, record the hash; extend the manifest guard to every sweep run. Manifests also record mix id and fleet-file hash.

## B2 — Implement and test

A `fleet_mix` config key (default M0) selecting the fleet file; nothing else. Tests: estimator tier = ground-truth tier for every ID in every mix; partitions unchanged; absent tiers never appear in logs; with `fleet_mix = M0` the 100-round logs equal the Part A replays.

## B3 — Runs, seed 4126 only

| mix | arms |
|---|---|
| M1, M2, M3, T1, T2, T3 | random, gpu (Ours), polaris, wen_fresh |

24 runs. Cheaper arms first; wen_fresh as capacity allows (disk guard, cleaner, watchdog 30-min no-merge alert). **Stop after seed 4126 and report.** Seeds 4127 / 4128 only on explicit go-ahead.

## B4 — Report `evidence/tier_sweep_s4126.md` (mechanical only)

1. Per mix (M0 seed 4126 first), in the format of `claude/s1-results-at-85.md`, for 80 / 85 / 90 %: TTA, energy to X %, rounds to X %, s/round to X % (TTA ÷ rounds), max accuracy; per arm.
2. Selection behaviour per mix: tier share (rounds 1–500, last 500), Gini, distinct vehicles, image coverage r500 / r1000 / end, mean |x_upload − 1000|; Wen limits, realised cuts, tier shares.
3. Cross-mix summary: Ours' gain vs Random and vs Polaris at 80 / 85 % (TTA and energy, %), Wen vs Random, and the fleet-heterogeneity index (CV over the 30 vehicles of datasheet T_train + mean T_up at Random's M0 mean upload distance).
4. Mechanism plots 1, 6 and 9 of A4 for M3 and T1 (the most and least heterogeneous mixes).
5. P-table: P44–P47 per mix and P48; seed-4126 directions, status "single seed — not evaluated".
6. Curves: accuracy vs online clock, one panel per mix, four arms.
7. Wall times, deviations.

---

## Stop conditions

- Step 0: no remote → report and wait.
- A2 test fails, or an A3 replay differs from its original anywhere → stop (do not start Part B).
- B0 partition mismatch or a tier wholly cut → stop and report.
- B1 commit missing → no sweep run.
- Online ≠ replayed clock in any run → stop all runs.
- Free disk < 20 GB for > 15 minutes → stop launching and report.
- Seed 4126 of Part B done → write the report and stop.

## Deliverable message

Step 0 results (commit, remote, tarball); A1 file list and columns; A2 tests; A3 replay checks; A4 report path and plot list; B0 fleets and Wen limits; B2 tests; B4 per-mix tables, cross-mix summary, P-table; every deviation.
