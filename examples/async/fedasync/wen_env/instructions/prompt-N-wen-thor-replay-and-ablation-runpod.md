# Prompt N — Why Wen et al. excludes Thor (Part 1, log-only replay), then Wen diagnostic ablations (Part 2, S1 seed 4126) (RunPod)

Continues from Prompt F (`evidence/wen_concentration_v1.md`), Prompt H (`evidence/wen_fresh_v1.md`) and Prompt K (`evidence/s1_final.md`). Same rules: do not edit `plato/` core; ground-truth / reported / estimated separation; locked values only; write new files only, never rewrite an existing report; stop and report on any conflict.

**Relation to Prompt M.** Part 1 is CPU-only analysis on existing logs and may run alongside Prompt M. Part 2 adds new arms behind default-off config keys; it must not change any Prompt M run path, and it starts only after Prompt M Part A replays are verified and GPU capacity allows (disk guard, cleaner, 30-min no-merge watchdog as usual).

## Step 0 — Lock and inventory

1. Copy `predictions_v3_8_addendum.md` to `docs/`, commit, record the hash. No replay or run before this commit.
2. For S1 `wen_fresh` seeds 4126 / 4127 / 4128 and S1 `wen_fresh_gpu` seeds 4126 / 4127 / 4128, report for `wen_candidates.csv`:
   - rows per decision (mean), and whether **infeasible candidates are logged with raw F1–F4** (expected: ≈ 28 rows per decision vs n_feasible ≈ 22);
   - which columns exist, in particular candidate position (`x_dispatch`), decision time, `V_k`, tier, `feasible_T`, `feasible_E`.
3. **If candidate position is not logged:** reconstruct it from the SUMO bundle (`bundle_wen_closed_v1`, kinematic constant speed, position wrap) at the decision's simulated time. Gate: recomputing CPU F1 and F2 from the reconstructed positions with the `wen_fresh` code must reproduce every logged F1 and F2 to 1e-9 relative. Any mismatch → stop and report the first row.
4. Gate: replaying R0 (below) reproduces every logged pick of all three `wen_fresh` runs. Any mismatch → stop.

---

# Part 1 — Static replay on logged candidates (no training)

## Rules for every replay

- Same candidate sets, same decision order, same F3 and F4 values as logged (F3 is the fresh value of the logged model version, so it is defined for every candidate, trained or not).
- Change only what the replay id says. Selection rule unchanged: feasibility filter → exact non-dominated set → min–max normalise within the Pareto set → pick the minimum mean (ties → lower id) → feasibility fallback as in `wen_faithful.py`.
- F1 / F2 functions: use the **exact** functions of `wen_fresh` (CPU family) and `wen_fresh_gpu` (GPU family) and only swap which function fills each slot. In the GPU family, keep its own coupling: upload position from the GPU-predicted finish, GPU energy = tier power × GPU T̂_train + p·T_up, exactly as `wen_fresh_gpu` computes it.
- "q90 / q99" limits: the W5 rule (Prompt E Step 1): quantile over 30 vehicles × trace samples of the F1 or F2 used in that replay. "∞" = constraint off.
- State in the report that static replay ignores feedback (the model trajectory, and so F3, would differ in a real run).

## Replays (S1, `wen_fresh`, seeds 4126 / 4127 / 4128)

| id | time model (F1) | energy model (F2) | T_max | E_max | objectives | purpose |
|---|---|---|---|---|---|---|
| R0 | CPU | CPU | q90 | q90 | F1–F4 | must reproduce the logged picks |
| R1 | CPU | CPU | q90 | **q99** | F1–F4 | layer 2, relaxed |
| R2 | CPU | CPU | q90 | **∞** | F1–F4 | layer 2 removed |
| R3 | CPU | CPU | ∞ | ∞ | F1–F4 | no constraints at all |
| R4 | CPU | CPU | ∞ | ∞ | **F1 + F2 only** | layer 3 alone (normalisation, no data objectives) |
| X1 | CPU | **GPU** | q90* | q90* | F1–F4 | energy model swapped only |
| X2 | **GPU** | CPU | q90* | q90* | F1–F4 | time model swapped only |
| X3 | GPU | GPU | q90* | q90* | F1–F4 | check: limits must equal the `wen_fresh_gpu` manifest (T_max 13.564085609369194 s, E_max 113.50862433236982 J) |
| X1∞ / X2∞ | as X1 / X2 | | ∞ | ∞ | F1–F4 | swapped model without constraints |

\* recomputed for that model combination. Report every limit value.

**Gate for X3:** recomputed limits equal the manifest values to 1e-9 relative, and F1 / F2 recomputed on the `wen_fresh_gpu` candidate files equal the logged values. Otherwise stop.

## Outputs per replay × seed

1. Per-tier rates: share of that tier's candidate rows that are feasible, in the Pareto set, picked; per-tier share of picks (Xavier / Orin / Thor). Random's realised whole-run tier shares at the same seed on the same table.
2. Gini of replayed pick counts; number of distinct vehicles picked.
3. **Thor diagnostics** (R0–R4, X2, X2∞): over decisions with ≥ 1 Thor in the Pareto set — count of such decisions; share where a Thor is picked; mean nF1, nF2, nF3, nF4 of the best-scored Thor and of the best-scored non-Thor; the P51 inequality share; Pareto-set F1 range split into "tier gap" (best Thor F1 vs best non-Thor F1) and "within-non-Thor spread".
4. **Xavier diagnostics** (X1, X1∞): the same table with Xavier in place of Thor.
5. **Client 26** (Thor; F4 rank 6 in seed 4126, rank 1 in seed 4128): candidate rows, feasible, in Pareto, picked, mean nF1–nF4 when in Pareto, per replay.
6. F2 by tier (box plot) and the q90 / q99 / logged E_max lines, CPU and GPU families.

## Report `evidence/wen_thor_replay_v1.md`

Mechanical. Step 0 inventory and gates; one table per output above (seeds as rows); the box plot; P49–P53 table (seed directions, v3 verdict). No interpretation beyond "consistent / not consistent with layer 1 / 2 / 3".

## Decision rule for the conditional Part 2 run (pre-set, no waiting)

If R1 gives a Thor share of picks ≥ 0.05 in **any** seed, run `wen_fresh_q99` in Part 2. Otherwise skip it and say so.

---

# Part 2 — Diagnostic ablations of Wen et al. (S1, seed 4126 only)

## Why

Prompt F / H counterfactuals are static. The claim "the F3 feedback loop, static F4 and the lack of exploration cause the collapse" needs real runs. These arms are **diagnostic ablations**, not comparators: the comparator stays `wen_fresh` as proposed.

## Arms (everything else exactly as `wen_fresh` S1 seed 4126: limits, FedAsync, C = 3, 4,000 rounds, stop at 90 %, eval every 10 rounds, Prompt M logging if it is on)

| arm | change | note |
|---|---|---|
| `wen_fresh_noF3` | objectives F1, F2, F4 (non-dominated filter and normalisation on three objectives) | fresh F3 not needed for selection: keep computing it **only if** cost allows, as a passive log; otherwise disable it and say so (saves ≈ 17 h) |
| `wen_fresh_noF4` | objectives F1, F2, F3 | |
| `wen_fresh_eps` | with probability ε = 0.2, pick uniformly among **Wen-feasible** idle candidates; otherwise the normal Wen pick | ε coin from its own seeded RNG stream (same mechanism as the `gpu` arm's `coin_u`); must not consume any other RNG; log `coin_u, explored` |
| `wen_fresh_q99` *(conditional, Part 1 rule)* | E_max = q99 (CPU family), T_max = q90 | |

Config keys `wen_objectives` (default `[F1,F2,F3,F4]`), `wen_epsilon` (default 0), `wen_emax_quantile` (default 90), all default-off.

## Tests

1. With all keys at default, a 100-round run reproduces the `wen_fresh` S1 seed-4126 dispatch log, `wen_decisions.csv` and `wen_candidates.csv` row for row.
2. `wen_fresh_eps` with ε = 0 reproduces the same logs (proves the coin does not disturb other RNG).
3. `noF3` / `noF4`: replaying the decisions with the logged values reproduces every pick; the dropped objective never enters normalisation.
4. Separation test still passes.
5. Online = replayed clock.

## Report `evidence/wen_ablation_s4126.md`

- Table in the format of `s1_final.md` 2.1a / 2.1b for: Random, Ours, `wen_fresh` (existing seed-4126 runs) and every new arm — TTA / energy / rounds to 80 / 85 / 90 %, s/round whole and before / after 80 %, max accuracy, tier shares (r1–500, last 500), Gini, distinct vehicles, image coverage r500 / r1000 / end.
- **TTA decomposition** for every arm: rounds to 80 % × s/round before 80 %, each as a ratio to Random.
- Rolling Gini (250-round windows) and top-5 vs rest F3 over time where F3 is logged.
- P54–P57: seed-4126 directions, status "single seed — not evaluated".
- Curves: accuracy vs online clock, all arms. Wall times, deviations.

**Stop after seed 4126 and report.** Seeds 4127 / 4128 only on explicit go-ahead.

---

## Stop conditions

- v3.8 not committed → nothing runs.
- Step 0 gate 3 or 4 fails, or the X3 gate fails → stop Part 1, report the first mismatch.
- Part 2 test fails → do not launch Part 2 runs.
- Online ≠ replayed clock in any run → stop all runs.
- Free disk < 20 GB for > 15 minutes → stop launching and report.
- Any Part 2 change alters a Prompt M run path → stop and report.

## Deliverable message

Step 0 commit hash, inventory and gates; Part 1 per-tier tables, Thor / Xavier / client-26 diagnostics, P49–P53 table, and the conditional-run decision; Part 2 tests, study table, TTA decomposition, P54–P57; every deviation.
