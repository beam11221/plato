# ADR-0014: Each arm estimates energy with its own time model

Status: accepted 16 Oct 2026 (Prompt C, decision D2). This supersedes the "E_max blocks the cpu arm" note in ADR-0010.

## Context

Under ADR-0010 every arm estimated energy as `P_tier × T̂_train + p × T̂_up`. For the cpu arm that gave 931 / 1,746 / 2,561 J (Xavier / Orin / Thor), because P_tier was multiplied by a 20–23 s CPU training estimate. All three are above E_max = 600 J, so "cpu + Stage 0" would have rejected every vehicle and fallen back on every cycle.

## Decision

An arm estimates energy with the model its time estimate comes from.

**GPU family** (`gpu`, `gpu_oracle`; also `location_only` and `observed` if they are ever combined with Stage 0):

```
Ê = P_tier × T̂_train + p × T̂_up        P = 40 / 75 / 130 W,  p = 0.2 W
```

**CPU family** (`cpu`), Wen Eq. 7 + Eq. 8:

```
Ê = e · ρ · b · |S_i| · f_tier² + p × T̂_up
    e = 1e-28,  ρ = 1,250,  b = 24,576,  f = 2.2 / 2.2 / 2.6 GHz
```

The training part comes to 24.8 / 24.8 / 34.6 J at |S| = 1,667.

**Unchanged.** E_max stays at 600 J. Ground-truth energy is still the GPU model for every arm, because only the cpu arm's estimate changes. Every dispatch logs `E_hat`.

## Per-tier Ê (|S| = 1,667, worst case at d = 1,000 m, where T̂_up ≈ 11.45 s)

| Tier | GPU family Ê | CPU family Ê | Blocked at 600 J? |
|---|---|---|---|
| Xavier | 111.4 + 2.3 ≈ 114 J | 24.8 + 2.3 ≈ 27 J | no |
| Orin | 26.8 + 2.3 ≈ 29 J | 24.8 + 2.3 ≈ 27 J | no |
| Thor | 24.1 + 2.3 ≈ 26 J | 34.6 + 2.3 ≈ 37 J | no |

`tests/test_wen_egreedy.py` sweeps x over the whole road for both families and every tier, and asserts Ê < 600 J.

## Consequence

Under the CPU model, Thor looks the most expensive vehicle (because of f²), while the ground truth makes Xavier the most expensive. This is the "wrong signal" the cpu arm is meant to carry.
