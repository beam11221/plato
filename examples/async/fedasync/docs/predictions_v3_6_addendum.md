# Pre-registered predictions v3.6 — Scenario 2 main comparison and Polaris plug-in (locked 23 Oct 2026, before any S2 `polaris` run)

Adds S2 Polaris arms to v3 … v3.5, which are unchanged. Commit before the first S2 `polaris` / `polaris_s0` run; record the hash in every S2 manifest from now on.

## Scenario 2 (locked)

Identical to S1 except coverage = [400, 1600] m (60 % of the 2,000 m loop, RSU radius 600 m), `coverage_gated` uploads (finish outside coverage = wasted dispatch). Pipeline: eligibility (idle, inside coverage at dispatch) → Stage 0 (if on; our estimator) → host → selection.

## Arms

Main: random, **Ours (gpu + Stage 0)**, Polaris, Wen et al. (`wen_fresh`).
Plug-in: random + Stage 0, wen_fresh + Stage 0, wen_fresh + GPU, wen_fresh + GPU + Stage 0, **Polaris + Stage 0**. Ablation: Ours without Stage 0 (gpu).

## Metrics (locked)

TTA@80 / 85 / 90 %; energy to 80 / 85 / 90 % = ground-truth E_train + E_up of **every** dispatch finished by that time, failed dispatches included; **successful training ratio (STR)** = updates delivered and merged / vehicles dispatched (Tu et al. 2024, arXiv 2410.10451: |N^r| / |S^r|). STR is the headline participation metric and replaces "waste" in the thesis text; the failed part is also reported as GPU-s and kJ.

**Mapping to earlier predictions (no re-wording of locked text):** "waste fraction (dispatches) lower" in P20, P26, P28 and P33 is evaluated as the identical statement "STR higher" (STR = 1 − waste fraction).

## Note on information already seen

Seed-4126 S2 values for random, Ours, gpu, wen_fresh and the Study 2 cells are already known (`study2_plugin_s4126.md`, `wen_fresh_v1.md`). The predictions below are about the 3-seed verdict; P41 is stated although seed 4126 is already close (Ours 1,109 s vs wen_fresh 1,145 s).

## Predictions (v3 rule: same direction in all 3 seeds)

- **P38:** Ours has lower TTA@80% than Polaris (S2).
- **P39:** Ours has a higher successful training ratio than Polaris (S2).
- **P40:** Ours uses less energy to reach 80% than Polaris (S2, failed dispatches included).
- **P41:** Ours has lower TTA@80% than wen_fresh (S2).
- **P42:** Polaris + Stage 0 has a higher successful training ratio than Polaris.
- **P43:** Polaris + Stage 0 has lower TTA@80% than Polaris.

Open S2 predictions carried over and evaluated with the same seeds: P25 (gpu without Stage 0 vs wen_fresh, TTA@80), P26–P28 (Stage 0 plug-in into random and wen_fresh), P33 (wen_fresh + GPU + Stage 0 waste ≤ wen_fresh + GPU).

## Reported, not predicted

All pairwise directions at 80 / 85 / 90 % on TTA, energy and STR; Polaris vs random; failed GPU-s and kJ; uplink airtime to 80 / 85 / 90 %; mean staleness of merged updates; Gini, distinct vehicles, image coverage; Stage 0 rejection rate per tier and reason.
