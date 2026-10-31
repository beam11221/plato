# Pre-registered predictions v3.7 — tier-mix sweep in Scenario 1 (locked 30 Oct 2026, before any sweep run)

Adds the tier-mix sweep to v3 … v3.6, which are unchanged. Commit before the first sweep run; record the hash in every sweep manifest.

## Why

The 15 / 10 / 5 (Xavier / Orin / Thor) fleet is our own choice. Reviewers asked whether the results — in particular Wen et al. losing to random — depend on it. The sweep changes **only the number of vehicles per tier**; everything else stays at the Scenario 1 defaults.

## Mixes (N = 30, Scenario 1)

| id | Xavier / Orin / Thor | purpose |
|---|---|---|
| M0 | 15 / 10 / 5 | existing S1 (reference, not rerun) |
| M1 | 10 / 10 / 10 | balanced |
| M2 | 5 / 10 / 15 | reversed |
| M3 | 2 / 8 / 20 | very few Xavier, mostly Thor |
| T1 | 15 / 0 / 15 | two tiers, no Orin, equal halves |
| T2 | 5 / 0 / 25 | two tiers, no Orin, mostly Thor |
| T3 | 15 / 15 / 0 | two tiers, no Thor — equal CPU frequency (2.2 GHz), the mix most favourable to Wen's CPU model |

Held fixed: vehicle IDs, per-ID speeds, per-ID data partitions, tier specifications, selection parameters (ε = 0.2, α = 1), Wen's q = 90 rule (limits recomputed per mix), Polaris settings (Prompt J), budget 4,000 rounds, stop at 90 %. Tier labels are assigned to contiguous ID blocks in the order Xavier → Orin → Thor, as in M0.

## Arms per mix

Random, Ours (`gpu`; Stage 0 is a no-op in S1), Polaris, Wen et al. (`wen_fresh`).

## Predictions (v3 rule: same direction in all 3 seeds; evaluated separately for each mix M1, M2, M3, T1, T2, T3)

- **P44 (per mix):** Ours has lower TTA@80% than Random.
- **P45 (per mix):** Ours has lower TTA@80% than Polaris.
- **P46 (per mix):** Ours has lower TTA@80% than Wen et al.
- **P47 (per mix):** Ours uses less energy to reach 85% than Random.
- **P48 (across mixes):** Ours' relative TTA@80% gain over Random, (Random − Ours) / Random per seed, is larger in T1 (15 / 0 / 15) than in M3 (2 / 8 / 20).

## Reported, not predicted

All thresholds 80 / 85 / 90 % for TTA, energy, rounds and s/round; Wen vs Random and Polaris vs Random in each mix; whether Wen reaches 90 %; Wen's limits and per-tier cuts; tier shares; Gini; image coverage; Ours' gain vs a fleet-heterogeneity index (coefficient of variation of the fleet's true T_train + mean T_up).

Context (not predictions): Wen's collapse (F3 / F4, deterministic) does not depend on the tier mix, so Wen is expected to stay below Random in most mixes; its CPU-model mis-ranking matters least in T3. Ours' advantage should shrink as the fleet becomes uniformly fast (M3, T2) toward the position-only gain (location_only ≈ −11 % s/round in M0), and be largest in T1.
