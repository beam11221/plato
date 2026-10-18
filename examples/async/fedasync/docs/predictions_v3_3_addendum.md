# Pre-registered predictions v3.3 — addendum (locked 18 Oct 2026, before any wen_fresh run)

Adds the `wen_fresh` arm to v3 (`f2bb90a1`), v3.1 (`098e1dd1`) and v3.2 (`66a2ed82`), which are unchanged. Commit this file before the first `wen_fresh` run and record its hash in every `wen_fresh` manifest.

## Why

The Prompt F diagnosis (`evidence/wen_concentration_v1.md`) showed that `wen`'s F3 (last reported loss) is refreshed only when a vehicle is picked again. Vehicles left out early keep losses of 0.79–0.95 from the early, poor global model for the whole run, while the most-picked vehicles report ≈ 0.01. Wen et al.'s Algorithm 3 recomputes f3 every round on the current global model, so the stale F3 is our adaptation, and it may amplify the lock-in. `wen_fresh` removes the staleness.

## Arm: `wen_fresh` (S1 and S2, Stage 0 off)

Identical to `wen` (W1–W10, same T_max = 34.334 s, E_max = 36.424 J, same seeds 4126 / 4127 / 4128, same stop rule) except **F3**:
- F3_k = mean cross-entropy loss of the **current global model** on vehicle k's full local dataset (eval mode, no augmentation), recomputed whenever the global model changes, for every vehicle.
- No prior is needed: every vehicle has a fresh value from the first decision on.
- The computation is **not charged to the clock or to energy** — a best case for Wen.
- Known difference from Wen: Wen measures the loss after local training on each candidate; we measure the loss of the global model before training (no candidate trains unless picked).

## Predictions (v3 rule: same direction in all 3 seeds; mean ± SD reported; magnitudes not predicted)

- **P22 Staleness amplified the collapse:** wen_fresh has a lower whole-run dispatch Gini than wen, in S1 and in S2.
- **P23 Collapse persists:** wen_fresh has a higher whole-run dispatch Gini than random, in S1 and in S2.
- **P24 Time-to-accuracy (S1):** gpu has lower TTA@80% than wen_fresh.
- **P25 Time-to-accuracy (S2):** gpu (Stage 0 off) has lower TTA@80% than wen_fresh.

## Reported, not predicted

wen_fresh vs wen on TTA@80 / 85 / 90% and on whether S1 reaches 90%; wen_fresh vs random on TTA; label coverage; onset of concentration.

Context (not a prediction): a fresh pre-training loss still measures how well the global model already fits a vehicle's data, so the most-trained vehicles should still have the lowest F3, and F4 is still static. We therefore expect a weaker but present collapse.
