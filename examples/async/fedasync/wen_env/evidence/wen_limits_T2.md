# Wen constraint limits, cpu family F1 / F2 (W5 rule; Prompt E Step 1; Prompt M B0.2, mix T2)

Population: 900,000 (vehicle, 1 s trace sample) pairs, t = 1..30,000 s.
T_max = P90(F1) = **32.40893691778983 s**; E_max = P90(F2) = **36.82174170391514 J**.
Qualitative check: PASS.

## All positions

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 60.0% | 0.0% | 0.0% | 60.0% |
| thor | 24.714–31.147 | 35.603–36.908 | 0.0% | 12.0% | 0.0% | 12.0% |
| fleet | 24.714–34.728 | 25.792–36.908 | 10.0% | 10.0% | 0.0% | 20.0% |

## Positions inside [400, 1600] at decision time

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 61.3% | 0.0% | 0.0% | 61.3% |
| thor | 24.714–31.147 | 35.603–36.908 | 0.0% | 17.0% | 0.0% | 17.0% |
| fleet | 24.714–34.728 | 25.792–36.908 | 10.2% | 14.1% | 0.0% | 24.3% |
