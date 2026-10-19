# Phase A analysis: Wen-aligned environment, seed 4126

**Date:** 16 Oct 2026.
**Data:** the 11 Phase A runs linked from `runs/wen_env/_phaseA/`.
**Mechanical numbers:** `main_runs_seed4126.md`.
**Curves:** `main_runs_seed4126_curves.png`.

> **Read this first.**
> - **One seed.** Every statement below is a single-seed observation, and no prediction is judged here; held / not held needs Phase B.
> - **Runs stop at 90% accuracy**, not at 4,000 rounds. There is therefore no true plateau, and late-run comparisons stop at about round 1,700–2,100.
> - **S1 gpu and S1 cpu are the runs that hung when the disk filled.** Their data is valid up to rounds 1,957 and 1,929 respectively, and their maxima (89.75% and 89.68%) are counted as 90%.
> - **Accuracy is noisy.** It is measured every 10 rounds and moves by about ±0.5 points between evaluations. Thresholds near the top (88–90%) are much noisier than 80% and 85%.

---

## 1. The environment and the two scenarios

### 1.1 What is the same in every cell

| Element | Setting |
|---|---|
| Road | One straight, one-way road, 2,000 m long with 2 lanes (Wen et al. 2024, §5.3.4). Vehicles leaving at x = 2,000 m re-enter at x = 0 with the same identity. The population is closed: the same 30 vehicles for the whole run. |
| Mobility | SUMO, kinematic. Each vehicle drives at one constant speed V_k drawn from U[17, 34] m/s (realised 17.1–32.6 m/s). Positions come from the SUMO trace (`bundle_wen_closed_v1`). |
| RSU | At x = 1,000 m (mid-road), 10 m off the road, so its distance to a vehicle is d = √((x − 1000)² + 10²). |
| Fleet | 30 vehicles: 15 Jetson Xavier, 10 Orin, 5 Thor (IDs 1–15, 16–25, 26–30). |
| Data | CIFAR-10, ResNet-18. A disjoint Dirichlet(α = 0.8) split: 1,667 or 1,666 images per vehicle, every image used exactly once. |
| Concurrency | C = 3 vehicles training at any moment (asynchronous FedAsync). |
| Training time (truth) | T_train = 3 × 1.113 GFLOP × \|S\| / (0.2 × TFLOPS) × a persistent ±15% per-vehicle jitter. That is about 2.6 / 0.36 / 0.19 s for Xavier / Orin / Thor. |
| Upload time (truth) | Wen Eq. 5–6: rate = B log₂(1 + d⁻³\|χ\|² p / N), with B = 2 MHz, p = 23 dBm, N = −114 dBm, and Rayleigh fading \|χ\|² ~ Exp(1) drawn per upload. d is taken where training finishes. The model is 357.6 Mbit, so T_up ≈ 5 s at d = 10 m and ≈ 11.5 s at d = 1,000 m. |
| One task | T_train + T_up. **Upload is about 85% of every task** (table in §4.2), so the system is communication-bound. |
| Round | One aggregation, i.e. one merged update. The clock is simulated online from the quantities above (no host time). |
| Stop rule | The run ends when test accuracy reaches 90%. |

### 1.2 S1: full coverage (Wen's own model)

- The RSU can hear every vehicle everywhere on the road. This is how Wen et al. model it: their system has no coverage boundary.
- **Every idle vehicle is a candidate** at every decision.
- **Every upload is accepted** (`always_accept`). Nothing is ever wasted.
- Position still matters through the upload rate. A vehicle that finishes training near x = 1,000 m uploads in about 5–6 s; one near either end of the road (d ≈ 1,000 m) takes about 11 s.

S1 asks one question: **which selection signal makes the fleet produce accurate models fastest, when nothing is lost?**

### 1.3 S2: partial coverage (our extension)

- The RSU only reaches **x ∈ [400, 1,600] m**, a 1,200 m zone (radius 600 m, inside the 300–1,000 m range of IEEE 802.11p).
- At 17–33 m/s a vehicle stays in the zone for 37–70 s per pass. About 18 of the 30 vehicles are inside it at any moment.
- **Candidates:** only idle vehicles that are inside the zone at decision time.
- **Upload policy `coverage_gated`:** the upload is accepted only if the vehicle stays inside the zone for the **whole** upload, from the end of training to the end of the upload. A vehicle that drives past x = 1,600 m before its upload completes loses the update.
- **Waste:** a lost update still cost GPU time and energy, and still occupied a slot for its full duration. It is counted as waste and does **not** create a round.
- **Stage 0 (optional feasibility filter):** before scoring, drop any candidate whose remaining time in the zone, R_i = (1,600 − x_i) / V_k, is shorter than its own estimated task time T̂_train + T̂_up, or whose estimated energy exceeds E_max = 600 J. Each arm applies Stage 0 with its own estimator. If nobody passes, the unfiltered pool is used and the event is logged as a fallback.

S2 asks two questions: **how much work is wasted when vehicles can drive out of range, and can a predictive filter (Stage 0) remove that waste without hurting learning?**

## 2. The arms

Every arm except `random` scores candidates with V_i = G_i / √(1 + τ̂_i):
- G_i is a gradient-bound estimate per vehicle, learnt from its updates.
- τ̂_i is the predicted staleness in rounds, derived from the estimated task time T̂_i.

Arms differ only in **how they estimate T̂** and in **the selection rule**.

| Arm | T̂_train | T̂_up | Rule |
|---|---|---|---|
| random | — | — | uniform (ε-greedy with ε = 1) |
| cpu | Wen Eq. 4 on the CPU clock: b·\|S\|·ρ / f ≈ 23.3 / 23.3 / 19.7 s | Wen Eq. 5–6 at the predicted finish position, \|χ\|² = 1 | ε-greedy, ε = 0.2 |
| **gpu** (the proposal) | GPU datasheet formula, no jitter: 2.78 / 0.36 / 0.19 s | same | ε-greedy, ε = 0.2 |
| gpu_oracle | the true jittered training time (upper bound) | same | ε-greedy, ε = 0.2 |
| observed | — (τ̂ = the vehicle's last observed staleness, Polaris-style) | — | ε-greedy, ε = 0.2 |
| gpu ε = 0 | as gpu | same | pure top-1 (ablation) |
| gpu proportional | as gpu | same | sample in proportion to V (the earlier rule; ablation) |

**ε-greedy.** With probability 0.2 the arm picks a random candidate; otherwise it picks the candidate with the highest V_i.

## 3. Headline numbers

| Cell | TTA@80% | TTA@85% | TTA@88% | s / round | Energy to 80% | S2 waste |
|---|---|---|---|---|---|---|
| S1 random | 1,419 s | 2,568 s | 3,990 s | 3.754 | 23.6 kJ | — |
| S1 cpu | 1,535 s | 2,837 s | 4,394 s | 3.724 | 26.6 kJ | — |
| **S1 gpu** | **1,158 s** | **2,089 s** | 4,151 s | **3.244** | **19.4 kJ** | — |
| S1 observed | 1,426 s | 2,962 s | 4,394 s | 3.681 | 23.1 kJ | — |
| S1 gpu_oracle | 1,168 s | 2,296 s | 3,608 s | 3.249 | 20.3 kJ | — |
| S1 gpu ε = 0 | 1,117 s | 2,100 s | 3,782 s | 3.145 | 18.9 kJ | — |
| S1 gpu proportional | 1,386 s | 2,530 s | 4,107 s | 3.740 | 25.0 kJ | — |
| S2 random | 1,813 s | 3,002 s | 4,009 s | 4.425 | 34.6 kJ | 23.1% |
| S2 gpu | 1,348 s | 2,254 s | 3,407 s | 3.573 | 24.8 kJ | 12.3% |
| **S2 gpu + Stage 0** | **1,109 s** | **2,003 s** | 3,467 s | **3.069** | **20.3 kJ** | **0.8%** |
| S2 cpu + Stage 0 | 1,548 s | 2,744 s | 4,252 s | 3.665 | 27.5 kJ | 0.5% |

**Every arm needs about the same number of rounds to reach 80%** (rounds 370–430). The TTA@80% differences are therefore almost entirely differences in **seconds per round**, not in learning per round.

## 4. S1: what happened, and why

### 4.1 gpu reaches 80% and 85% about 18% sooner than random; cpu does not

- **gpu against random:** −18.4% to 80% and −18.6% to 85%. Seconds per round are −13.6% over the whole run and −20.5% before the 80% crossing.
- **cpu against random:** 8% slower to 80%. Its seconds per round are almost the same (−0.8%).
- **Energy to 80%:** gpu 19.4 kJ against random 23.6 kJ and cpu 26.6 kJ.

### 4.2 The mechanism: gpu wins mostly by picking vehicles that are about to be close to the RSU

| S1 cell | Mean T_train | Mean T_up | Mean task | Mean d at upload | Dispatched while inside [400, 1,600] |
|---|---|---|---|---|---|
| random | 1.38 s | 9.88 s | 11.26 s | 492 m | 61% |
| cpu | 1.50 s | 9.65 s | 11.15 s | 462 m | 64% |
| **gpu** | **1.38 s** | **8.35 s** | **9.73 s** | **284 m** | **84%** |
| gpu_oracle | 1.36 s | 8.38 s | 9.74 s | 291 m | 83% |
| gpu ε = 0 | 1.41 s | 8.02 s | 9.43 s | 233 m | 91% |
| observed | 1.44 s | 9.59 s | 11.03 s | 465 m | 65% |
| proportional | 1.47 s | 9.75 s | 11.22 s | 479 m | 64% |

- **gpu's mean training time per task equals random's (1.38 s).** Over the whole run, gpu does not train on faster hardware on average.
- **What it does is upload from about 40% closer to the RSU.** The upload distance falls from 492 m to 284 m, which saves about 1.5 s per task. That gain appears in every tier: gpu's mean upload distance is 261–300 m for every tier, against about 490 m for every tier under random.
- **Why it can.** T̂ = T̂_train + T̂_up. Under the GPU estimate, T̂_train is small (0.19–2.8 s), so the position-dependent T̂_up dominates the ranking. A vehicle whose training will finish near the RSU gets a short T̂, hence a low τ̂ and a high V.
- **Why cpu cannot.** Its T̂_train (20–23 s) is larger than every upload time, so all candidates have nearly the same τ̂ and position barely moves the ranking. That is why cpu's upload distance (462 m) is close to random's.
- **The tier signal is there too, but early and smaller.** In rounds 1–500, gpu dispatched Xavier 32% of the time (random 43%; fleet share 50%) and Thor 30% (random 18%). By rounds 1,500–2,000 gpu dispatched Xavier 51%.

**Implication to verify in Phase B.** Most of gpu's S1 advantage comes from the **position** term that it shares with cpu, which the CPU estimate drowns out, rather than from knowing GPU speed. This is exactly what the location-only arm (P12, not run yet) would separate: T̂_train held constant, T̂_up kept. On these numbers it could recover most of gpu's gain. Running P12 deserves a higher priority.

### 4.3 After 80%, gpu's advantage in time shrinks, and it needs more rounds near the top

| | Rounds to 80% | Rounds to 85% | Rounds to 88% | TTA@88% |
|---|---|---|---|---|
| random | 380 | 680 | 1,060 | 3,990 s |
| gpu | 390 | 680 | 1,300 | 4,151 s |
| gpu_oracle | 390 | 740 | 1,140 | 3,608 s |
| gpu ε = 0 | 400 | 720 | 1,260 | 3,782 s |

- gpu's seconds per round stay lower after the 80% crossing (3.31 against 3.76), but it needed more **rounds** to reach 88%. At that threshold it lost its time advantage (4,151 s against 3,990 s).
- The P3 text anticipates this ("fair-rotation pay-back"): an arm that dispatches some vehicles less early has to catch up on their data later. The rising late Xavier share fits that reading.
- **This is the noisiest part of the curves**, at one seed and ±0.5-point evaluation noise. Treat it as a question for Phase B, not a finding.

### 4.4 The ablations

| Arm | Result |
|---|---|
| **gpu_oracle ≈ gpu** | 1,168 s against 1,158 s to 80%, with the same seconds per round (3.249 against 3.244). Knowing each vehicle's true training time, with ±15% jitter, adds nothing over the datasheet estimate, which is expected when upload dominates. |
| **ε = 0 (pure top-1)** | The fastest to 80% (1,117 s) and the lowest seconds per round (3.145). It is the most position-greedy (91% of dispatches inside the central zone, mean upload distance 233 m) and the most tier-skewed early (Xavier only 26% in rounds 1–500). Its plateau is equal to ε = 0.2's (0.878 against 0.879). |
| **proportional ≈ random** | Seconds per round 3.740 against 3.754, and mean upload distance 479 m against 492 m. This confirms the smoke-run reason for switching to ε-greedy (ADR-0013): sampling in proportion to a compressed V is close to uniform. |
| **observed ≈ random** | 1,426 s to 80% and 3.681 seconds per round. It is behind gpu in every phase (P8), consistent with history-based τ being a noisy guide once upload time depends on where the vehicle happens to be. |

## 5. S2: what happened, and why

### 5.1 Waste

| S2 cell | Waste (share of dispatches) | Wasted GPU-s | Wasted energy | Accepted updates / s |
|---|---|---|---|---|
| random | 23.1% | 819 | 37.5 kJ | 0.226 |
| gpu | 12.3% | 444 | 20.1 kJ | 0.280 |
| gpu + Stage 0 | 0.81% | 17 | 0.86 kJ | 0.326 |
| cpu + Stage 0 | 0.47% | 17 | 0.75 kJ | 0.273 |

- **random wastes about one dispatch in four.** It picks vehicles anywhere in the zone, including ones a few seconds from x = 1,600 m.
- **gpu's scoring alone halves the waste,** with no filter. Its preference for "finishes near the RSU" also favours vehicles with room left before the exit.
- **Stage 0 cuts waste to under 1%** for both estimators. The residual comes from fading: a draw that lengthens an upload past the exit, since Stage 0 plans with |χ|² = 1.

### 5.2 Speed

- **gpu + Stage 0 is the fastest cell of the whole experiment:** 1,109 s to 80% and 3.07 seconds per round. Two things combine:
  - almost no slot time is lost to waste;
  - it dispatches only vehicles that will stay inside, i.e. are close to the RSU (mean upload distance 179 m, mean upload 7.7 s, the lowest of all cells).
- **gpu without Stage 0** (1,348 s) is already 26% faster than random (1,813 s), whose waste costs it time.
- **cpu + Stage 0 rejects 73% of candidates** (gpu + Stage 0: 24%), falling back to the unfiltered pool in 33 of 1,899 cycles.
  - The cause: its T̂ (about 20–33 s) is about 3 times the true task time, so only vehicles with a long remaining dwell pass.
  - Those are mostly vehicles that just entered the zone near x = 400 m, far from the RSU. Its mean upload distance is 392 m, against 179 m for gpu + Stage 0.
  - Result: low waste, but long uploads and fewer dispatches per second (0.274 against 0.328). It reaches 80% in 1,548 s, slower than gpu even without Stage 0.

### 5.3 Learning quality

All four S2 cells cross 80% at rounds 380–430 and end at about 90%.

The Stage 0 "plateau" (mean of the last 200 rounds) is 0.8830 against 0.8841 without Stage 0. That is a 0.11-point difference, opposite to the P11 wording ("does not lower"), but far inside the evaluation noise, and the plateau metric itself is distorted by the 90% stop. Phase B decides this.

## 6. The predictions (direction at this seed only; not evaluated)

| P | Seed 4126 direction | Comment |
|---|---|---|
| P1 cpu closer to random than to gpu | as predicted | cpu −0.8% against gpu −13.6% in seconds per round |
| P2 gpu beats cpu | as predicted | seconds per round and TTA@80% |
| P3 gpu beats random (before 80%, whole run, TTA@80%) | as predicted | but the gain vanishes by 88% (§4.3) |
| P4 gpu plateau within ±1 point of random | as predicted | −0.6 points; the plateau is affected by the 90% stop |
| P5 energy to 80%: gpu < random | as predicted | 19.4 against 23.6 kJ |
| P7 gpu ≈ gpu_oracle | close | no direction predicted |
| P8 observed does not beat gpu | as predicted | gpu ahead in every phase |
| P9 Stage 0 reduces waste (gpu) | as predicted | 12.3% → 0.8%; the direction was already seen in the smoke run |
| P10 gpu + Stage 0 waste near zero; cpu + Stage 0 fewer dispatches per second | as predicted | 0.8%; 0.274 against 0.328 per second |
| P11 Stage 0 does not lower the plateau | **opposite, by 0.11 points** | inside the noise; the plateau metric is affected |
| P13 proportional ≈ random; ε = 0 gain ≥ ε = 0.2; ε = 0 plateau ≤ ε = 0.2 | ≈ / as predicted / as predicted | the "≈" needs the seed spread |
| P6, P12, and P9 for Wen-faithful | not run | |

## 7. What to look at when Phase B finishes

1. **The gpu − random gap in seconds per round and TTA@80%** across all 3 seeds: the core S1 claim.
2. **The rounds-to-88% pattern (§4.3):** whether gpu's late slowdown is systematic or noise.
3. **P11:** the sign of the Stage 0 plateau difference across seeds.
4. **Position against hardware.** The gpu advantage is driven by upload position more than by GPU speed. This strengthens the case for running **location-only (P12)** before writing claim C1 ("GPU vs CPU signal"). The fair comparison for "what GPU knowledge adds" is gpu against location-only, not gpu against random.
