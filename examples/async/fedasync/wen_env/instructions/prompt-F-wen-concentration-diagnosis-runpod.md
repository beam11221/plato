# Prompt F — Wen-faithful participation collapse: log-only diagnosis (RunPod)

Continues from Prompt E (`evidence/wen_faithful_v1.md`). **No new training runs.** Everything below is computed from existing logs: the 6 `wen` runs (S1, S2 × seeds 4126 / 4127 / 4128), their `wen_decisions.csv` and per-candidate side files, and the dispatch / accuracy logs of the S1 and S2 `random` and `gpu` runs. Do not edit `plato/` core or any run output; write new files only.

## Why

S1 `wen` has Gini 0.816 (random 0.069, gpu 0.125), one vehicle takes ≈ 25% of all dispatches, and no seed reaches 90%. We need to know **which objective drives the collapse** before Study 2. Hypotheses, stated before looking (not pre-registered predictions):
- H1 — F3 feedback loop: a picked vehicle's reported loss falls, so it is picked again.
- H2 — F4 is static, so the vehicles with the lowest KL are always favoured.
- H3 — F1 / F2 (position + tier) concentrate picks on whichever vehicles sit at the favoured position.

## Step 1 — Per-vehicle table (each wen run)

One row per vehicle: `client_id, tier, |S_i|, V_i, F4 value, F4 rank (1 = lowest), dispatch count, dispatch share, rank by dispatch count, mean F1 / F2 / F3 when a candidate, mean normalised F1..F4 when picked`. Same dispatch-count columns for `random` and `gpu` at the same seed.

Report Spearman correlation (per seed) between dispatch count and: F4 rank, mean F3, mean F1, mean F2, V_i, tier.

## Step 2 — Time course

1. Rolling Gini of dispatch counts in windows of 250 rounds (wen, random, gpu; S1 and S2). Report the round at which the top vehicle's cumulative share first exceeds 10% and 20%.
2. For the top-5 vehicles by dispatch count vs the rest: reported loss (F3) over rounds (mean per 100-round window).
3. Per decision, which objective the pick is the minimum of among all candidates (F1-min, F2-min, F3-min, F4-min; several can be true). Share per 500-round window.

## Step 3 — Static counterfactual on logged decisions

For every logged wen decision, replay the selector on the **logged candidate set and logged F values** with one objective removed at a time (drop F1, drop F2, drop F3, drop F4; constraints unchanged) and with all four kept (must reproduce the logged pick exactly — a test; stop if it does not). Report per scenario and seed:
- share of decisions whose pick changes when each objective is dropped;
- Gini of the counterfactual pick counts (on the logged decisions; this ignores feedback, say so in the report).

This is a static replay: it shows which objective decides each pick, not what a rerun would do.

## Step 4 — Label coverage of the aggregated updates

Each dispatch contributes the label histogram of the dispatched vehicle. For wen, random and gpu (S1, S2, each seed), compute the cumulative label distribution of all **aggregated** updates (count-weighted by |S_i|, staleness weights ignored) at rounds 250, 500, 1,000, 2,000, 4,000 (or the run's last round), and its KL to uniform 0.1. Also report the share of the 50,000 training images that have been in at least one aggregated update by those rounds.

## Step 5 — Consistency fix for the earlier report

`wen_faithful_v1.md` Step 6 lists S1 gpu and S1 cpu at seed 4126 (the disk-full hangs at rounds 1,957 / 1,929) as "not reached 90%". By the earlier decision these two count as reaching 90% in `main_runs_v1.md` and `cpu_cal_v1.md`. Add a footnote to the Step 6 table in a new file (do not rewrite `wen_faithful_v1.md`) giving both readings, and use the earlier decision (counted as 90%, last accuracy 0.8975 / 0.8901) in every table in this prompt.

## Step 6 — Report

`evidence/wen_concentration_v1.md`: Steps 1–4 tables, the rolling-Gini and F3-over-time plots (one panel per scenario, wen vs random vs gpu), the counterfactual table, the label-coverage table, Step 5 footnote, and the replay test result. Mechanical only — state which hypotheses the numbers are consistent with, no further interpretation.

## Stop conditions

- Step 3 replay with all four objectives does not reproduce every logged pick → stop and report the first mismatch.
- Any per-candidate side file missing for a run → report which, continue with the rest.

## Deliverable message

Step 1 correlations, Step 2 onset rounds, the Step 3 counterfactual table, the Step 4 coverage table, the replay test result, and every deviation.
