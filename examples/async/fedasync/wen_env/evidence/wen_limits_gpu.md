# Wen constraint limits, gpu family F1 / F2 (W5 rule; Prompt I Step 2.3)

Population: 900,000 (vehicle, 1 s trace sample) pairs, t = 1..30,000 s.
T_max = P90(F1) = **13.564085609369194 s**; E_max = P90(F2) = **113.50862433236982 J**.
Qualitative check: PASS.

## All positions

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 7.814–14.235 | 112.359–113.643 | 0.0% | 0.0% | 20.0% | 20.0% |
| orin | 5.387–11.808 | 27.757–29.058 | 0.0% | 0.0% | 0.0% | 0.0% |
| thor | 5.215–11.636 | 25.118–26.402 | 0.0% | 0.0% | 0.0% | 0.0% |
| fleet | 5.215–14.235 | 25.118–113.643 | 0.0% | 0.0% | 10.0% | 10.0% |

## Positions inside [400, 1600] at decision time

| tier | F1 range (s) | F2 range (J) | cut by T only | cut by E only | cut by both | total cut |
|---|---|---|---|---|---|---|
| xavier | 7.814–13.170 | 112.359–113.430 | 0.0% | 0.0% | 0.0% | 0.0% |
| orin | 5.387–10.433 | 27.757–28.782 | 0.0% | 0.0% | 0.0% | 0.0% |
| thor | 5.215–10.240 | 25.118–26.123 | 0.0% | 0.0% | 0.0% | 0.0% |
| fleet | 5.215–13.170 | 25.118–113.430 | 0.0% | 0.0% | 0.0% | 0.0% |
