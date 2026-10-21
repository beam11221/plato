# Polaris in S1 — seeds 4126, 4127, 4128 (Prompt J, extended on user go-ahead of 21 Oct)

Per-seed tables:
- `polaris_s1_s4126.md`, which is the full report with deviations and the Step 0 answers;
- `polaris_s1_s4127_tables.md`;
- `polaris_s1_s4128_tables.md`.

All three `polaris` runs reached 90%, and online = replayed clock in each (max diff 0.0 s, no order residuals). `wen_fresh` was not run at seeds 4127 and 4128 (Prompt I: those seeds await a go-ahead), so it appears in the seed-4126 table only.

| seed | polaris: round at 90% | TTA@80 / 85 / 90% (s) | s/round | energy to 80% (kJ) | Gini |
|---|---|---|---|---|---|
| 4126 | 2,020 | 1580 / 2452 / 7646 | 3.785 | 27.8 | 0.050 |
| 4127 | 2,340 | 1903 / 2924 / 8803 | 3.762 | 32.9 | 0.049 |
| 4128 | 2,060 | 1988 / 2976 / 7739 | 3.757 | 36.1 | 0.053 |

## P34–P37 (v3 rule: the same direction in all 3 seeds)

| P | comparison | 4126 (gpu vs polaris) | 4127 | 4128 | seeds as predicted |
|---|---|---|---|---|---|
| P34 | gpu has lower s/round than polaris | 3.244 vs 3.785 | 3.253 vs 3.762 | 3.234 vs 3.757 | 3 / 3 |
| P35 | gpu has lower TTA@80% than polaris | 1158 vs 1580 | 1237 vs 1903 | 1550 vs 1988 | 3 / 3 |
| P36 | gpu uses less ground-truth energy to reach 80% than polaris | 19.4 vs 27.8 kJ | 21.2 vs 32.9 kJ | 25.9 vs 36.1 kJ | 3 / 3 |
| P37 | polaris has a lower whole-run dispatch Gini than gpu | 0.114 vs 0.050 | 0.132 vs 0.049 | 0.128 vs 0.053 | 3 / 3 |

## Polaris internals, all seeds

- **Effective pool 1/Σq².** It stays at 26.9–27.9 in every 500-round window, against about 27–28 candidates, and the median max q is 0.041–0.049. q never concentrates.
- **Spearman correlation of per-vehicle dispatch counts.** Against mean G, mean τ, tier and mean V, |ρ| ≤ 0.28 with p ≥ 0.13 in every seed.
