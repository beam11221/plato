# Wen constraint limits, cpu family F1 / F2 (W5 rule; Prompt E Step 1; Prompt M B0.2, mix M3)

Population: 900,000 (vehicle, 1 s trace sample) pairs, t = 1..30,000 s.
T_max = P90(F1) = **33.69977461802514 s**; E_max = P90(F2) = **36.799328319679084 J**.
Qualitative check: PASS.

## All positions

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 30.0% | 0.0% | 0.0% | 30.0% |
| orin | 28.307–34.728 | 25.792–27.076 | 30.0% | 0.0% | 0.0% | 30.0% |
| thor | 24.714–31.147 | 35.603–36.908 | 0.0% | 15.0% | 0.0% | 15.0% |
| fleet | 24.714–34.728 | 25.792–36.908 | 10.0% | 10.0% | 0.0% | 20.0% |

## Positions inside [400, 1600] at decision time

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 38.9% | 0.0% | 0.0% | 38.9% |
| orin | 28.307–34.728 | 25.792–27.076 | 41.0% | 0.0% | 0.0% | 41.0% |
| thor | 24.714–31.147 | 35.603–36.908 | 0.0% | 20.6% | 0.0% | 20.6% |
| fleet | 24.714–34.728 | 25.792–36.908 | 13.5% | 13.7% | 0.0% | 27.2% |
