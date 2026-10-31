# Pre-registered predictions v3.8 — why Wen et al. excludes Thor, and what drives its collapse (locked 31 Oct 2026, before any Prompt N replay or run)

Adds to v3 … v3.7, which are unchanged. Commit before Prompt N Part 1 starts; record the hash in every Prompt N manifest and report.

## Why

In S1, `wen_fresh` gives Thor a whole-run share of 0.000–0.002 although Thor has the highest CPU frequency (2.6 GHz vs 2.2 GHz). Working explanation, in three layers:
1. **Physics of the CPU model.** Thor's T_com is 15 % shorter (F1 −12 % with upload), but E_com ∝ f² makes it 40 % more expensive (F2 +36 %).
2. **Our calibration.** E_max = q90 of F2 over the fleet lands inside Thor's F2 range (Thor = 16.7 % of the fleet, highest F2), so 10 / 16.7 ≈ 60 % of Thor candidates are cut before ranking.
3. **Wen's normalisation.** Inside the Pareto set the tier gap fills almost the whole F2 range (Thor nF2 ≈ 1, Xavier/Orin ≈ 0), while F1's range is largely filled by position spread among Xavier/Orin, so Thor's time advantage is diluted. Evidence already in hand: client 26 (Thor) has F4 rank 1 in seed 4128 and is dispatched 4 times.

Layer 2 is ours, layers 1 and 3 are Wen's. Prompt N separates them.

## Part 1 — static replay predictions (S1, `wen_fresh`, seeds 4126 / 4127 / 4128; v3 rule: same direction in all 3 seeds)

Replay ids are defined in Prompt N.

- **P49:** Under R1 (E_max = q99, T_max = q90, CPU model), Thor's share of replayed picks is below 0.05.
- **P50:** Under R2 (E_max = ∞, T_max = q90, CPU model), Thor's share of replayed picks is below Random's realised whole-run Thor share at the same seed.
- **P51:** Under R2, among decisions with at least one Thor in the Pareto set, the energy penalty exceeds the time advantage in more than half of them: (nF2 of the best-scored Thor − nF2 of the best-scored non-Thor) > (nF1 of the best-scored non-Thor − nF1 of the best-scored Thor).
- **P52:** Under X1 (CPU time + GPU energy, limits recomputed at q90), Xavier's share of replayed picks is below 0.05.
- **P53:** Under X2 (GPU time + CPU energy, limits recomputed at q90), Thor's share of replayed picks is below 0.05.

P52 and P53 together test the claim "the energy objective alone removes whichever tier it prices highest, whatever the time model".

## Part 2 — dynamic ablation (S1, seed 4126 only; single seed — directions reported, not evaluated)

Diagnostic ablations of Wen et al., not comparators. Reference values: `wen_fresh` seed 4126 (Gini 0.819, TTA@80 % 1,967 s, max accuracy 0.8683); Random seed 4126 (TTA@80 % 1,419 s).

- **P54:** `wen_fresh_noF3` has lower dispatch Gini than `wen_fresh`.
- **P55:** `wen_fresh_noF4` has lower dispatch Gini than `wen_fresh`.
- **P56:** `wen_fresh_eps` (ε = 0.2) has lower TTA@80 % than `wen_fresh`.
- **P57 (conditional run only):** if `wen_fresh_q99` runs, its Thor share is below Random's Thor share.

## Reported, not predicted

Per-tier feasible / in-Pareto / picked rates for every replay; Thor's mean nF1 and nF2 when in the Pareto set; client 26 in every replay; Gini of replayed picks; whether any Part 2 arm reaches Random's TTA@80 % or 90 % accuracy; rounds to 80 % and s/round for every Part 2 arm.

Context (not predictions): static replay ignores feedback. If Thor is picked early in a real run, its F3 can fall and Thor could join the rich-get-richer loop, so a static replay can underestimate a dynamic effect. That is why Part 2 has a conditional q99 run.
