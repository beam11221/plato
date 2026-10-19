# Prompt B, Steps 1–2: measured training time, and the fleet on the trace

## Step 1: FLOPs and training time

`wen_env/measure_flops.py` produced the raw measurement in `flops.json`.

| | Value |
|---|---|
| Counter | fvcore `FlopCountAnalysis`. It counts multiply-adds (**MACs**); FLOPs = 2 × MACs. |
| Model | PLATO `resnet_18` (`plato.models.resnet.Model`, BatchNorm, CIFAR stem), 32×32, batch 1, eval mode |
| MACs, forward | 556,651,520 (conv 555,417,600; batch norm 1,228,800; linear 5,120) |
| **FLOPs_fwd** | **1,113,303,040**, which is +1.2% against 1.1 GFLOP (stop band ±20%: no stop) |
| FLOPs_train per image | 3 × FLOPs_fwd = 3,339,909,120 |
| Parameters | 11,173,962, giving D = 357,566,784 bits (state_dict elements 11,183,582, including BatchNorm buffers) |

The datasheet T_tier below is computed with η = 0.2 and E = 1, and no jitter.

| Tier | TFLOPS | \|S\| = 1,667 | \|S\| = 1,666 | Roadmap value at 1.1 GFLOP |
|---|---|---|---|---|
| Xavier | 10 | 2.7838 s | 2.7822 s | 2.75 s |
| Orin | 78 | 0.35690 s | 0.35669 s | 0.35 s |
| Thor | 150 | 0.18559 s | 0.18548 s | 0.18 s |

The true time is T_tier × exp(eps_i), with eps_i persistent per client.
- σ = 0.149166 gives a CV of 15%.
- The eps values come from the fleet `n30_x15-o10-t5_phys_p1667_j0p149_s4126` (seed 4126), which was rebuilt for this prompt with the locked σ and the measured W.

## Step 2: bundle and mapping

**Integrity.**
- The zip `bundle_wen_closed_v1.zip` is **not on this machine**; the bundle arrived extracted. The pinned zip sha256 is therefore only checked when a zip is supplied.
- In its place, `manifest.yaml` is pinned (sha256 `f07a1880…43d2`), and all 19 per-file sha256 values it lists are verified. They all match.

**Speeds.** The desired speeds equal `default_rng(20261015).uniform(17, 34, 30)` in ID order. They range from 17.077 to 32.590 m/s.

**Mapping.** Client i ↔ `veh{i−1:02d}`. The tiers form contiguous blocks, asserted against the fleet file.

| Tier | Clients | Mean V_k |
|---|---|---|
| Xavier | 1–15 (veh00–veh14) | 26.77 m/s |
| Orin | 16–25 (veh15–veh24) | 24.86 m/s |
| Thor | 26–30 (veh25–veh29) | 28.13 m/s |

These match the expected 26.8 / 24.9 / 28.1 m/s and were not reshuffled.

| client | veh_id | tier | V_k (m/s) | \|S_i\| | exp(eps_i) | T_train_true (s) |
|---|---|---|---|---|---|---|
| 1 | veh00 | xavier | 21.775 | 1667 | 0.9491 | 2.6420 |
| 2 | veh01 | xavier | 26.988 | 1667 | 0.8760 | 2.4386 |
| 3 | veh02 | xavier | 25.073 | 1667 | 0.8303 | 2.3115 |
| 4 | veh03 | xavier | 24.017 | 1667 | 0.8874 | 2.4703 |
| 5 | veh04 | xavier | 17.077 | 1667 | 0.8816 | 2.4542 |
| 6 | veh05 | xavier | 30.007 | 1667 | 1.1573 | 3.2217 |
| 7 | veh06 | xavier | 17.371 | 1667 | 1.1021 | 3.0680 |
| 8 | veh07 | xavier | 32.043 | 1667 | 0.7643 | 2.1276 |
| 9 | veh08 | xavier | 30.561 | 1667 | 0.9768 | 2.7193 |
| 10 | veh09 | xavier | 31.865 | 1667 | 1.1575 | 3.2221 |
| 11 | veh10 | xavier | 32.590 | 1667 | 0.7966 | 2.2177 |
| 12 | veh11 | xavier | 26.913 | 1667 | 0.8937 | 2.4879 |
| 13 | veh12 | xavier | 32.390 | 1667 | 1.0310 | 2.8701 |
| 14 | veh13 | xavier | 24.665 | 1667 | 0.9792 | 2.7260 |
| 15 | veh14 | xavier | 28.275 | 1667 | 0.8811 | 2.4528 |
| 16 | veh15 | orin | 20.993 | 1667 | 0.7805 | 0.2786 |
| 17 | veh16 | orin | 23.041 | 1667 | 1.1169 | 0.3986 |
| 18 | veh17 | orin | 25.581 | 1667 | 0.9499 | 0.3390 |
| 19 | veh18 | orin | 30.583 | 1667 | 1.2874 | 0.4595 |
| 20 | veh19 | orin | 17.697 | 1667 | 0.9807 | 0.3500 |
| 21 | veh20 | orin | 25.656 | 1666 | 1.2660 | 0.4516 |
| 22 | veh21 | orin | 17.609 | 1666 | 0.9533 | 0.3400 |
| 23 | veh22 | orin | 31.711 | 1666 | 0.8478 | 0.3024 |
| 24 | veh23 | orin | 31.505 | 1666 | 1.1289 | 0.4027 |
| 25 | veh24 | orin | 24.197 | 1666 | 0.9014 | 0.3215 |
| 26 | veh25 | thor | 21.515 | 1666 | 0.9633 | 0.1787 |
| 27 | veh26 | thor | 26.642 | 1666 | 0.8225 | 0.1526 |
| 28 | veh27 | thor | 32.137 | 1666 | 1.1810 | 0.2190 |
| 29 | veh28 | thor | 28.417 | 1666 | 0.8851 | 0.1642 |
| 30 | veh29 | thor | 31.919 | 1666 | 1.1346 | 0.2104 |

**Position lookup.**
- Positions are interpolated linearly within one lap. They are never interpolated across a wrap; across one they are extrapolated kinematically from the sample on the same lap.
- The prompt asks for agreement with the global `(x0 + V_k t) mod 2000` within 0.2 m over the whole trace. **That does not hold for this bundle.** SUMO's wrap costs one 0.1 s step per lap (`wen_closed.py`), so the global form drifts by about V_k × 0.1 m per lap: about 435 m for veh04 and about 1.6 km of travel-equivalent for veh10 by t = 30,000 s.
- The test therefore checks the per-lap form, `x(t_lap_start) + V_k (t − t_lap_start)`, within 0.2 m on every lap of every vehicle. A separate test pins the global drift.
- Before the first sample (t ∈ [0, 1)), x is extrapolated backwards from the t = 1 sample with V_k.

**Trace end.** A position past t = 30,000 s raises `TraceEndError`, which is a stop condition. The trace is never looped.
