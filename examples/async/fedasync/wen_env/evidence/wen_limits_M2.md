# Wen constraint limits, cpu family F1 / F2 (W5 rule; Prompt E Step 1; Prompt M B0.2, mix M2)

Population: 900,000 (vehicle, 1 s trace sample) pairs, t = 1..30,000 s.
T_max = P90(F1) = **34.062392404031414 s**; E_max = P90(F2) = **36.76129724789274 J**.
Qualitative check: PASS.

## All positions

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 20.0% | 0.0% | 0.0% | 20.0% |
| orin | 28.307–34.728 | 25.792–27.076 | 20.0% | 0.0% | 0.0% | 20.0% |
| thor | 24.714–31.147 | 35.603–36.908 | 0.0% | 20.0% | 0.0% | 20.0% |
| fleet | 24.714–34.728 | 25.792–36.908 | 10.0% | 10.0% | 0.0% | 20.0% |

## Positions inside [400, 1600] at decision time

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 28.307–34.728 | 25.792–27.076 | 27.4% | 0.0% | 0.0% | 27.4% |
| orin | 28.307–34.728 | 25.792–27.076 | 31.4% | 0.0% | 0.0% | 31.4% |
| thor | 24.714–31.147 | 35.603–36.908 | 0.0% | 25.1% | 0.0% | 25.1% |
| fleet | 24.714–34.728 | 25.792–36.908 | 15.0% | 12.5% | 0.0% | 27.6% |
