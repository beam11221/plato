# ADR-0007: Upload time follows Wen Eq. 5–6

Status: accepted 14 Oct 2026, implemented 16 Oct 2026 (Prompt B, Steps 3 and 4).

## Decision

```
Tr   = B · log2(1 + h · p / (ε + N)),    h = d^(−φ) · |χ|²
T_up = D / Tr
d    = sqrt((x − 1000)² + 10²)
```

| Constant | Value | Source |
|---|---|---|
| B | 2 MHz per vehicle | Wen Table 2 |
| p | 23 dBm (0.1995 W) | Wen Table 2 |
| N | −114 dBm | Wen Table 2 |
| φ | 3 | Wen Table 2 |
| ε | 0: orthogonal sub-channels, no co-channel interference | decided 14 Oct |
| \|χ\|² | ~ Exp(1), one draw per upload (Rayleigh) | decided 14 Oct |
| D | ResNet-18 parameter count × 32 bits = 11,173,962 × 32 = 357,566,784 bits, read from the model | — |
| Download | ignored | Wen §3.3 |

**Where d is taken.** d is evaluated at the end of training, as in Wen's model. Under S2, coverage is checked over the whole upload interval `[t_dispatch + T_train, t_finish]`. A partial upload is never accepted.

**How the fading draw is seeded.** The RNG is keyed by `(seed, client_id, round, k)`. Here `k` counts earlier dispatches of the same client under the same round number. It is 0 except after an S2 cycle whose only update was lost: that holds the round counter, and without `k` the redispatch would replay the identical draw.

## Check (`tests/test_wen_physics.py`)

With D = 11.17 M × 32 and |χ|² = 1, the prompt's table is reproduced within ±0.1 s.

| d (m) | 10 | 100 | 250 | 500 | 1,000 |
|---|---|---|---|---|---|
| T_up, D = 11.17 M × 32 (s) | 5.03 | 6.99 | 8.27 | 9.60 | 11.45 |
| T_up, real D (s) | 5.03 | 6.99 | 8.27 | 9.61 | 11.45 |

## Consequence

Upload is most of every task's time: about 5–11 s against 0.19–2.8 s of training. The experiment is therefore communication-bound, as the pre-registered predictions assume.
