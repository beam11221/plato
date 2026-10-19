# ADR-0013: Selection rule is ε-greedy on V_i, with ε = 0.2

Status: accepted 16 Oct 2026 (Prompt C, decision D1). Implemented in `wen_env/wen_egreedy.py`.

## Context

The Prompt B smoke run used the earlier rule: sample a client with probability proportional to V_i = G_i / √(1 + τ̂_i), without replacement.

Once upload time is in T̂, every vehicle's T̂ falls between about 8 and 12 s. That compresses V to about 0.45–0.56 (gpu arm, S1: s(τ̂) = 0.53 / 0.53 / 0.48 for Thor / Orin / Xavier). Sampling in proportion to numbers that close is almost uniform.

In the smoke run, gpu was only 1.3% faster per round than random (3.72 s against 3.77 s per round). The scoring rule had stopped being able to express its own ranking.

## Decision

The same procedure runs for every arm that scores with V_i (`gpu`, `cpu`, `gpu_oracle`, `observed`, `location_only`), once per free slot (C = 3).

1. **Candidates.**
   - S1: every idle vehicle.
   - S2: idle vehicles in coverage that, when Stage 0 is on, are also feasible.
2. **Coin.** u ~ U(0,1), drawn from `default_rng([seed, round, k, salt])`. `k` counts decisions already taken under that round number, the same convention as the fading key, so a round held by a lost update does not replay its draw.
3. **Explore.** If u < ε, pick uniformly among the candidates, using a second draw from the same stream.
4. **Exploit.** Otherwise pick the candidate with the highest V_i. Ties go to the lower `client_id`.

**ε values.**
- Every other cell uses ε = 0.2.
- `random` is ε = 1 and goes through the same class (`WenSelector` with no scorer), so eligibility, slot counting and RNG handling are identical to the other arms.
- The S1 gpu ablations are ε = 0 (pure top-1) and `proportional` (the previous rule, delegated unchanged to the Polaris selector).

**Scores within one selection call.** V is computed once per call. When one call fills several slots (the first cycle, or a refill), the later picks reuse that snapshot minus the clients already picked.

## Why

- **Ranking, not spread.** ε-greedy depends only on how the candidates rank by V. Its selectivity therefore does not shrink as V's spread shrinks.
- **Exploration floor.** ε keeps every candidate's probability per decision at or above ε / |candidates|. That preserves the exploration that the G_i prior and the observed arm rely on.
- **Precedent.** Oort (Lai et al., OSDI '21) reserves an exploration fraction ε of each round, with 1 − ε for exploitation.

## Logging

Every dispatch row records:
- `selection_rule`, `epsilon`;
- `coin_u`, `explored`;
- `n_candidates`, `rank_of_pick` (1 = highest V);
- `V_max`, `V_min`, `V_pick`.

## Checks (`wen_env/tests/test_wen_egreedy.py`)

- ε = 0 always picks rank 1, and ties go to the lower id.
- Over 10,000 decisions among 27 candidates, ε = 0.2 picks rank 1 with probability 0.8 + 0.2/27 (±1%), and every candidate is picked at least 0.2/27 − 0.3% of the time.
- On real runs, ε = 1 reproduces the random arm's dispatch log, and `proportional` reproduces the Prompt B gpu smoke log.
