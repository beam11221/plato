# Wen constraint limits, cpu family F1 / F2 (W5 rule; Prompt E Step 1; Prompt M B0.2, mix M1)

Population: 900,000 (vehicle, 1 s trace sample) pairs, t = 1..30,000 s.
T_max = P90(F1) = **34.235525358332396 s**; E_max = P90(F2) = **36.68189617270728 J**.
Qualitative check: PASS.

## All positions

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 15.0% | 0.0% | 0.0% | 15.0% |
| orin | 28.307–34.728 | 25.792–27.076 | 15.0% | 0.0% | 0.0% | 15.0% |
| thor | 24.714–31.135 | 35.603–36.888 | 0.0% | 30.0% | 0.0% | 30.0% |
| fleet | 24.714–34.728 | 25.792–36.888 | 10.0% | 10.0% | 0.0% | 20.0% |

## Positions inside [400, 1600] at decision time

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 22.1% | 0.0% | 0.0% | 22.1% |
| orin | 28.307–34.728 | 25.792–27.076 | 23.2% | 0.0% | 0.0% | 23.2% |
| thor | 24.714–31.135 | 35.603–36.888 | 0.0% | 36.1% | 0.0% | 36.1% |
| fleet | 24.714–34.728 | 25.792–36.888 | 15.1% | 12.0% | 0.0% | 27.2% |
