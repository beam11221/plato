# Wen constraint limits, cpu family F1 / F2 (W5 rule; Prompt E Step 1)

Population: 900,000 (vehicle, 1 s trace sample) pairs, t = 1..30,000 s.
T_max = P90(F1) = **34.334345905647005 s**; E_max = P90(F2) = **36.423846640394224 J**.
Qualitative check: PASS.

## All positions

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 12.1% | 0.0% | 0.0% | 12.1% |
| orin | 28.293–34.728 | 25.777–27.076 | 11.9% | 0.0% | 0.0% | 11.9% |
| thor | 24.714–31.135 | 35.603–36.888 | 0.0% | 60.0% | 0.0% | 60.0% |
| fleet | 24.714–34.728 | 25.777–36.888 | 10.0% | 10.0% | 0.0% | 20.0% |

## Positions inside [400, 1600] at decision time

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 18.7% | 0.0% | 0.0% | 18.7% |
| orin | 28.293–34.728 | 25.777–27.076 | 17.7% | 0.0% | 0.0% | 17.7% |
| thor | 24.714–31.135 | 35.603–36.888 | 0.0% | 62.8% | 0.0% | 62.8% |
| fleet | 24.714–34.728 | 25.777–36.888 | 15.3% | 10.5% | 0.0% | 25.7% |
