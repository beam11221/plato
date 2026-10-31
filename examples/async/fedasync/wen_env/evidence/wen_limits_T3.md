# Wen constraint limits, cpu family F1 / F2 (W5 rule; Prompt E Step 1; Prompt M B0.2, mix T3)

Population: 900,000 (vehicle, 1 s trace sample) pairs, t = 1..30,000 s.
T_max = P90(F1) = **34.399097287533074 s**; E_max = P90(F2) = **27.006107925574867 J**.
Qualitative check: PASS.

## All positions

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 0.0% | 0.6% | 10.1% | 10.7% |
| orin | 28.293–34.728 | 25.777–27.076 | 0.8% | 0.2% | 9.1% | 10.1% |
| fleet | 28.293–34.728 | 25.777–27.076 | 0.4% | 0.4% | 9.6% | 10.4% |

## Positions inside [400, 1600] at decision time

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 0.0% | 1.0% | 15.8% | 16.7% |
| orin | 28.293–34.728 | 25.777–27.076 | 1.3% | 0.3% | 14.1% | 15.6% |
| fleet | 28.293–34.728 | 25.777–27.076 | 0.6% | 0.6% | 14.9% | 16.2% |
