# Prompt H — `wen_fresh`: Wen-faithful with a fresh F3 (Study 1b, RunPod)

Continues from Prompt E (`evidence/wen_faithful_v1.md`) and Prompt F (`evidence/wen_concentration_v1.md`). Same rules: do not edit `plato/` core; keep the ground-truth / reported / estimated separation; locked values only; stop and report on any conflict.

**If Prompt G (audit bundle) has not finished, finish it first**, so that the audit reflects the code used for Study 1. Prompt H then adds code on top.

## Step 0 — Confirm the current `wen` F3 prior (no code change)

From the code used by the 6 `wen` runs, report with file:line:
1. The exact definition of the never-trained prior: mean over the **latest** report of each vehicle that has reported, or over **all** reports in history, or something else.
2. Whether the prior is recomputed at every decision.
3. Why never-trained Thor vehicles show mean F3 ≈ 0.34 in the Prompt F per-vehicle tables, while vehicles picked once early stay at 0.79–0.95 — reproduce one such value from the logs.

Write this into `evidence/wen_fresh_v1.md` (section 0). Do not change `wen`.

## Step 1 — Lock the predictions

Copy `predictions_v3_3_addendum.md` (supplied) to `docs/predictions_v3_3_addendum.md`, commit, record the hash. Extend the manifest guard: no `arm = wen_fresh` run without this commit.

## Step 2 — Implement `wen_fresh`

New arm `wen_fresh`, same selector code path as `wen`; the only difference is the F3 source:
- **F3_k** = mean cross-entropy of the **current global model** on vehicle k's full local dataset (its disjoint partition), eval mode, no augmentation, same normalisation transforms as training.
- Recompute for **all 30 vehicles** every time the global model changes (each merge), so every decision uses losses from the model version current at that decision. Cache by global-model version; do not recompute between merges.
- No prior; no use of reported training losses.
- The evaluation is **not charged** to the online clock or to energy. Log its wall time separately (`fresh_eval_s` per merge).
- Everything else identical to `wen`: F1, F2, F4, T_max = 34.334345905647005 s, E_max = 36.423846640394224 J, fallback, exact Pareto, min–max normalisation, pick rule, tie-break, determinism.

Same per-decision log and per-candidate side file as `wen`, plus `global_version` used for F3.

ADR (next free number): `wen_fresh` — why (Prompt F staleness finding), the definition, the difference from Wen's Algorithm 3 (pre-training loss of the global model vs post-training loss of each candidate), and that it is not charged to the clock (best case for Wen).

## Step 3 — Tests

1. With F3 replaced by the stale-loss source, the `wen_fresh` code path reproduces the `wen` S1 seed 4126 dispatch log for 100 rounds.
2. F3 for 3 vehicles at one global version matches a direct evaluation to 1e-6.
3. Cache: F3 changes only when `global_version` changes.
4. The online clock of a 100-round `wen_fresh` run is unaffected by `fresh_eval_s` (online = replayed).
5. Determinism: two 100-round runs, seed 4126, identical logs.
6. Report the mean `fresh_eval_s` per merge and the projected extra wall time per 4,000-round run.

## Step 4 — Smoke (200 rounds, seed 4126, S1 and S2)

Same table as Prompt E Step 4, plus the rolling Gini over rounds 1–200 and the F3 range across the fleet at rounds 50, 100, 200. Do not read accuracy.

## Step 5 — Runs

| Scenario | Arm | Stage 0 | Seeds |
|---|---|---|---|
| S1 | wen_fresh | off | 4126, 4127, 4128 |
| S2 | wen_fresh | off | 4126, 4127, 4128 |

4,000 rounds max, stop at 90%, manifest with predictions v3.3 hash, T_max / E_max, seed. Checkpoint cleaner and disk guard as before.

## Step 6 — Extra log-only analysis on the existing `wen` runs (no new runs)

Extend the Prompt F static counterfactual with two more replays on the logged `wen` decisions: **drop F3 and F4 together** (pick on F1 + F2 only) and **drop F1 and F2 together** (pick on F3 + F4 only). Report pick-change share and counterfactual Gini per scenario and seed, as in Prompt F Step 3.

## Step 7 — Report

`evidence/wen_fresh_v1.md`:
0. Step 0 prior definition.
1. Study table as in Prompt E Step 7 (S1: random, gpu, wen, wen_fresh, location_only, cpu; S2: random, gpu, wen, wen_fresh): TTA@80 / 85 / 90%, s/round, energy to 80%, tier share, waste (S2), mean upload distance, mean |x̂ − x|.
2. Concentration as in Prompt F for wen_fresh next to wen, random, gpu: whole-run Gini, rolling Gini, onset rounds, per-vehicle table (one seed per scenario is enough in the main text; all seeds in an appendix), Spearman correlations, F3 over time (top-5 vs rest), label coverage.
3. Constraint table for wen_fresh.
4. P22–P25 table, per-seed direction, held / not held. No re-wording.
5. Curves: accuracy vs online clock, random / gpu / wen / wen_fresh, one panel per scenario.
6. Step 6 counterfactual table.
7. Extra wall time spent on fresh evaluation.

Mechanical only — no conclusions beyond the P-table.

## Stop conditions

- Step 1 commit missing → no `wen_fresh` run.
- Any Step 3 test fails → stop.
- Projected extra wall time > 8 h per run → report and wait before Step 5.
- Online ≠ replayed clock → stop all runs.
- Free disk < 20 GB for > 15 minutes → stop launching and report.

## Deliverable message

Step 0 answer, test results, smoke table, ADR number, the study table, concentration comparison, P22–P25, the Step 6 counterfactual, extra wall time, and every deviation.
