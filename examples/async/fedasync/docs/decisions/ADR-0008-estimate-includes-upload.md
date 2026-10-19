# ADR-0008: Every estimator scores with T̂ = T̂_train + T̂_up

Status: accepted 14 Oct 2026, implemented 16 Oct 2026 (Prompt B, Step 5).

## Decision

Every non-random arm estimates a vehicle's whole task. The estimate is built only from what the vehicle reports at decision time: position x, speed V_k, tier spec, and |S_i|.

```
x̂     = wrap(x + V_k · T̂_train)
T̂_up  = Wen Eq. 5–6 at d(x̂), with |χ|² = 1
T̂     = T̂_train + T̂_up
Ê     = P_tier · T̂_train + p · T̂_up
```

| Arm | T̂_train |
|---|---|
| gpu | the tier formula of ADR-0006, without jitter |
| cpu | b·\|S\|·ρ / f (ADR-0009): 23.3 / 23.3 / 19.7 s |
| gpu_oracle | the true jittered training time. This is the only ground truth any arm receives, and it is passed in as an explicit callable. |
| location_only | constant: the fleet mean of the gpu estimates. The switch is implemented; the arm has not been run yet. |
| observed | no T̂. τ̂ is the last delivered PLATO staleness. Durations now include upload, so that staleness counts the merges during the whole task (training and upload). |

**T_round and τ̂ keep their definitions, applied to T̂.**
- The queue kernel gives `T_round = mean(T̂) / C`.
  - At startup it is taken over the fleet at t = 0.
  - Afterwards it is recomputed live over the busy set, whose T̂ are frozen at dispatch.
- τ̂ is the existing two-phase residual count.
- The score is `V_i = G_i / sqrt(1 + τ̂)`.

## Separation

The ground truth (jitter, fading, future position, true durations) lives in `wen_truth`, `wen_fleet` and `wen_trace`. The estimator, selection and reported modules never import those (`tests/test_wen_truth_and_selection.py`). `ReportedVehicle` has no field that could carry ground truth.

## Consequence

τ̂ and the gpu-versus-cpu comparison now include position. Because the upload term is shared by every arm, gpu versus cpu isolates the speed signal. gpu versus random does not.
