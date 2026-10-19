# Roadmap — Wen-aligned closed-population environment (Oct 2026)

Agreed 13–14 Oct 2026. Replaces the open-population plan (p_return sweep). Status keys: [ ] todo · [~] under discussion · [x] decided.

## Decided

| Item | Value | Source |
|---|---|---|
| Population | Closed only (open dropped after advisor meeting) | advisor |
| Fleet N | 30 | Wen et al. 2024, Table 2 |
| Concurrency C | 3 (C/N = 10%) | Pisces (N=100/200/400, C=10/20/40), Polaris (N=200, up to 10%) |
| Road | Straight, one-way, 2,000 m, 2 lanes (overtaking) | Wen §5.3.4 (2000×20); lane count not given by Wen |
| Speed | 17–34 m/s (highway) | Wen Table 2, §5.3.4 |
| Scenario 1 | 100% coverage, `always_accept` | Wen (no coverage boundary in model) |
| Scenario 2 | Coverage 1,200 m centred (radius 600 m), `coverage_gated`, Stage 0 | IEEE 802.11p 300–1,000 m |
| RSU position | Roadside, at road midpoint | Wen §5.3.2 + Fig. 2 |
| Main comparator | Wen et al. (RA-MaOVSM) | — |
| Polaris / observed | Ablation only | advisor |
| Data quality | Keep G_i (do not switch to Wen's KL + loss) | scope discipline |
| Dataset / model | CIFAR-10 + ResNet-18 first | — |
| Tier mix | 15 Xavier / 10 Orin / 5 Thor | — |
| Wen parameters | b_k = 24,576 bits (32×32×3×8), \|S_k\| ≈ 1,667 (50,000/30), D_k(w) = ResNet-18 param count × 32 bits (read exact count from PLATO), ε_k = 0 (orthogonal sub-channels), Loc_y = 10 m, ρ_k identical for all vehicles (cancels from ranking; value still to pick from Wen refs [42]/[44]) | Wen leaves these unstated |
| Fading | Ground truth draws \|χ\|² ~ Exp(1) per upload; estimators use E = 1 | not known at decision time |
| Energy model | E = P_max(tier) × T_train + p × T_up; P_max = 40 / 75 / 130 W (Xavier / Orin / Thor, top power mode, consistent with TFLOPS figures). Per-task ≈ 575 / 134 / 139 J — slow tier costs most; Wen's E ∝ f² ranks the opposite way (supports "wrong signal is worse than none") | NVIDIA datasheets |
| E_max | Non-binding by default (e.g. 600 J), energy acts via F2 and as a reported metric; E_max sweep as separate sensitivity, never tuned on outcome | Wen's own constraints are effectively loose |
| Wen F3 (training loss) | Use last-reported loss per vehicle (prior for never-trained); same information class as our G_i. Literal "train-all-then-pick" kept only as optional sync ablation | fairness |
| Speed model | Each vehicle gets one desired speed ~ U[17, 34] m/s tied to its ID; SUMO sigma = 0 (constant speed); 2 lanes so fast vehicles overtake; verify realized ≈ desired from trace. Time-varying speed (sigma > 0) only as later sensitivity | Wen Table 1/2, Fig. 2 (distance = T_com·V_k) |

## 0. Blocking decisions / checks

- [ ] Units of sleep table vs Wen upload time (T_up ≈ 4.6–11.5 s for ResNet-18 over 5–1,000 m vs tier sleep means 1.07 / 1.78 / 14.37 s). Rescale with the same anchor if needed — energy figures above also depend on this
- [x] T̂ = T̂_train + T̂_up inside the training-time estimate (decided 14 Oct). Consequence: η (GPU) and ρ (CPU) no longer cancel from ranking — they set the weight of the training term against the upload term
- [x] ρ_k = 1,250 cycles/bit, identical for all vehicles (decided 15 Oct) — midpoint of [900, 1600] in Xiao et al. [44], reached via Wen's Table 2 citation list; used by our cpu arm, Wen-faithful and Wen + GPU's constraint alike. Calibrated ρ ≈ 427 proposed as a sensitivity cell (cpu arm, S1 and S2), not yet confirmed
- Background — ρ_k source check (14 Oct): Wen gives no value. Eq. 4 cites [35] (Li et al., TVT 2022) — arXiv version models upload energy only, no ρ. Table 2 cites [12, 14, 42, 44–46]; [44] Xiao et al. (T-ITS 2022) Table II: processing density [900, 1600] cycles/bit drawn per vehicle, capacitance 1e-27. With ρ = 1,250 Wen's own setting makes both constraints bind (T_max hit by slow CPUs: 37 s at 1 GHz; E_max hit by fast CPUs: 34 J at 3 GHz) — corrects the earlier "Wen's constraints are loose" reading, which assumed ρ = 20. Options: ρ = 1,250 (citable via Wen → [44]; our T̂_cpu ≈ 23 s at 2.2 GHz, ~3× fleet-mean truth) vs calibrated ρ ≈ 427
- [~] ρ_k: alternative rule = calibrate so the CPU estimator's fleet-mean T̂_train equals the true fleet-mean T_train (ρ ≈ 427 cycles/bit with b=24,576, |S|=1,667, f=2.2 GHz, fleet mean 7.96 s); apply the same fleet-mean calibration to η. Wen-scale ρ=20 gives T̂_cpu ≈ 0.37 s, which would turn the cpu arm into a distance-only selector
- [x] Stage 0 in Scenario 2 uses each vehicle's reported constant V_k (decided 14 Oct)
- [x] Carry-over confirmed: Dirichlet α = 0.8, G_i prior α = 1, σ = 0.149 (CV 15%), partition 1,667
- [~] Wen optimizer: RVEA-PDCG as published vs off-the-shelf RVEA / NSGA-III (pymoo); single solution picked by smallest normalized mean of objectives (Wen §5.3.2); population 50, 1,500 evaluations
- [~] Wen-async scalarization: per-vehicle score = normalized mean of F1, F2, F3, F4 (same rule Wen uses to pick from the Pareto set)
- [~] Wen + GPU runs in sync, async, or both
- [~] Equal-budget rule for sync vs async comparison (wall-clock budget and/or total GPU-seconds)
- [~] Upload vs training weight (15 Oct estimate, greedy-pick toy model, positions uniform, tier sleep means taken as seconds): at current scale (fleet-mean train 7.96 s) upload is ~63% of total time, but training spread (std 6.3 s) exceeds upload spread (std 3.0 s), so GPU ranking stays near-oracle (Spearman 0.93) while location-only is weak (0.25). As train scale shrinks to 0.1× (≈ one physical epoch on Xavier), location-only catches up (−34% vs gpu −38% vs random). True training time per local round (sleep units × local epochs) decides the regime — fix it with a physical justification, not by outcome
- [x] Extra arm: location-only (T̂_train constant, T̂_up shared) — the proper null for the speed signal, since random lacks location information
- [x] Tier CPU frequencies from code CSV: Xavier 2.2, Orin 2.2, Thor 2.6 GHz → with ρ = 1,250: T̂_cpu = 23.3 / 23.3 / 19.7 s (cpu ties Xavier with Orin, prefers Thor slightly)
- [x] Location-only = variant (b): our selector with constant T̂_train. Lowest priority; skip if main results already separate cpu / gpu / random (note: gpu vs cpu isolates the speed signal since both carry T̂_up; gpu vs random does not)
- [x] Anchor dropped (decided 15 Oct): training time from the physical formula T = 3 × FLOPs_fwd × |S| × E / (η × TFLOPS) with E = 1, η = 0.2 → Xavier 2.75 s, Orin 0.35 s, Thor 0.18 s (fleet mean 1.52 s). Toy-model consequence (greedy pick, 20k decisions): gpu −40% vs random (Spearman 0.87), location-only −30% (0.66), cpu −8% (0.04 — its ρ-based T̂ ≈ 20–23 s swamps the upload term, so it ignores location). Total-time ratio Xavier:Thor ≈ 1.25× (was 13×). Energy per task ≈ 110 / 26 / 23 J. Location-only is now the informative null for gpu; recommend raising its priority. Deferred (15 Oct): location-only priority and ρ sensitivity to be decided after the first batch of results
- Background — anchor / training-time scale (15 Oct). Facts: sleep in seconds, 1 local epoch, η currently 1.0, real training < 1 s, so sleep = anchor-scaled. Anchor was harmless while T_comm = 0 (clock linear in anchor); with physical T_up it now sets the compute:comm regime. Physical estimates (3 × 1.1 GFLOP fwd, 1,667 img/vehicle): E=1, η=1 → fleet mean 0.30 s (0.04× current); E=1, η=0.2 → 1.52 s (0.19×); E=5, η=0.2 → 7.62 s (0.96×); YOLOP rough (18.6 GFLOP fwd, ~1,167 img, η=0.2) → 18 s (2.3×), upload 3.6–8.1 s. Proposal: replace the free anchor with the physical formula, lock η and E with citations, and report both a comm-bound and a compute-bound point
- Section B (Wen optimizer, Wen-async scalarization, Wen + GPU sync/async, equal-budget rule) deferred until the comparator stage

## Run plan (CIFAR-10, 3 seeds each)

| # | Scenario | Arm | Selector | Speed signal | Stage 0 |
|---|---|---|---|---|---|
| 1 | S1 | random | random | — | off |
| 2 | S1 | cpu | ours | CPU-cycle + T̂_up | off |
| 3 | S1 | gpu (proposal) | ours | GPU-TOPS + T̂_up | off |
| 4 | S1 | Wen-faithful | Wen (sync) | CPU-cycle + T̂_up | off |
| 5 | S1 | Wen + GPU | Wen (sync) | GPU-TOPS + T̂_up | off |
| 6 | S1 | observed (ablation) | ours | history | off |
| 7 | S1 | gpu_oracle (ablation) | ours | per-vehicle truth | off |
| 8 | S2 | random | random | — | off |
| 9 | S2 | gpu | ours | GPU-TOPS + T̂_up | off |
| 10 | S2 | gpu + Stage 0 | ours | GPU-TOPS + T̂_up | on |
| 11 | S2 | Wen-faithful | Wen (sync) | CPU-cycle + T̂_up | off |
| 12 | S2 | Wen + Stage 0 | Wen (sync) | CPU-cycle + T̂_up | on |
| 13 | S2 | cpu + Stage 0 | ours | CPU-cycle + T̂_up | on |
| 14 | S1 | location-only (decided 15 Oct) | ours | constant T̂_train + T̂_up | off |

14 arms × 3 seeds = 42 runs. Location-only is the null for the speed signal: gpu vs location-only = what GPU adds; cpu vs location-only = whether CPU helps or hurts; location-only vs random = what position adds. Wen-async variants (Section B) add 2 arms × 3 seeds later.

## Pre-registered predictions (v2, 15 Oct 2026 — lock before any run)

Written for the locked setting: N = 30, C = 3, CIFAR-10 + ResNet-18, E = 1, η = 0.2 (Xavier 2.75 s / Orin 0.35 s / Thor 0.18 s training), upload per Wen Eq. 5–6 (≈ 5–11 s), T̂ = T̂_train + T̂_up for every non-random arm, ρ = 1,250, tier CPU 2.2 / 2.2 / 2.6 GHz, Dirichlet α = 0.8, G_i prior α = 1, σ = 0.149. This is a communication-bound regime: upload is most of each task's time.

Rule: a prediction holds if the direction is the same in all 3 seeds; report mean ± std. Magnitudes are not predicted. The toy-model numbers below (greedy pick, no G_i, no fair rotation) are context for why each direction is expected, not thresholds.

Primary readouts: seconds-per-round (mechanism; time-resolved before / after the 80% crossing) and TTA@80% (outcome). Plateau and energy are secondary.

### Scenario 1 — 100% coverage

- **P1 cpu carries no useful signal.** cpu is not faster than random on seconds-per-round or TTA@80% (tie within seed spread, or slower). Why: T̂_cpu ≈ 23 / 23 / 20 s swamps the 5–11 s upload term, so cpu ranks almost purely by CPU frequency, which ties Xavier with Orin (toy: −8% vs random, Spearman 0.04).
- **P2 gpu beats cpu** on seconds-per-round and on TTA@80%.
- **P3 gpu beats random on seconds-per-round before the 80% crossing.** On TTA@80% gpu ≤ random, with a tie within seed spread an acceptable outcome (closed pool; Stage 10 had gpu 6.9% behind random on TTA). Toy: −40% per pick, Spearman 0.87.
- **P4 Plateau** equal across all arms within ±1 point.
- **P5 Energy to reach 80%:** gpu < random; cpu not lower than random. Per-task training energy ≈ 110 / 26 / 23 J (Xavier / Orin / Thor), upload ≈ 1–2 J.
- **P6 Transplant:** Wen + GPU reaches 80% sooner than Wen-faithful, and has lower seconds-per-round.
- **P7 gpu ≈ gpu_oracle** (tier spread 15× vs 15% jitter; the upload term is shared, so the gap should be smaller than in earlier stages).
- **P8 observed does not beat gpu in any phase.** Changed from v1: history now records total time including upload, which depends on where the vehicle was at the time, so a vehicle's past total time is a noisy guide to its next one; gpu recomputes T̂_up from the current position at every decision. Expect observed ≈ gpu at best, worse early (k* ≈ 30–70 obs/client ≈ 300–700 rounds at N = 30, C = 3).
- **P12 Location-only (conditional — applies only if the arm is run, with the same seeds and config):** location-only beats random on seconds-per-round; gpu beats location-only on seconds-per-round. Toy: location-only −30%, gpu −40%, Spearman 0.66 vs 0.87.

### Scenario 2 — partial coverage (1,200 m), `coverage_gated`

- **P9 Stage 0 reduces waste** for both our selector (gpu) and Wen-faithful; Wen's Eq. 10 has no coverage term.
- **P10 gpu + Stage 0 waste is near zero** (exact position and constant V_k are known; residual comes only from 15% jitter and Rayleigh fading). **cpu + Stage 0 over-filters:** its T̂ is ~8× the true training time, so it rejects vehicles that would have finished — fewer admitted dispatches per unit time than gpu + Stage 0.
- **P11 Stage 0 does not lower the plateau** (adversarial data-to-GPU mapping test still pending).

### Changes from v1
- P1 softened from "cpu slower than random" to "cpu no faster than random", because tier CPU frequencies (2.2 / 2.2 / 2.6) give cpu a weak correct preference for Thor
- P5 energy figures recomputed for E = 1, η = 0.2 (were 575 / 134 / 139 J under the anchored sleep table)
- P8 reversed from "observed ≈ gpu at steady state" to "observed does not beat gpu", because history now includes position-dependent upload
- P12 added for location-only
- P10 adds the mechanism for cpu over-filtering (scale ~8× too high), not only wrong per-tier ordering

## 1. SUMO

Done 15 Oct — bundle `bundle_wen_closed_v1.zip` (sha256 18532ec2…8a24), all 8 gates pass. Two amendments approved by the author after Krauss failed G1/G3/G4: (1) **kinematic constant-speed motion** (TraCI setSpeedMode(0) + setSpeed(V_k); vehicles do not interact, x = x0 + V_k·t, matching Wen's model) with the closed loop as a position wrap; Krauss run kept as evidence that heterogeneous constant speeds platoon on a 2-lane 2 km loop — a limitation of Wen's assumption, to disclose. (2) G2 upper bound relaxed to 32.5 m/s; realized draw 17.08–32.59 m/s, seed untouched. Under kinematic motion the `lane` column and LC2013/Krauss parameters are inert — server must ignore `lane`. Speed draw reproduces as default_rng(20261015).uniform(17, 34, 30) in ID order; if tiers are contiguous ID blocks, mean speed Xavier 26.8 / Orin 24.9 / Thor 28.1 m/s (no effect in S1; ~2-point feasibility difference in S2) — confirm tier mapping in Prompt B Step 2. ADR numbered ADR-0001 on the SUMO side collides with the existing ADR-0001…0006 series; renumber or prefix.

- [ ] One-way 2,000 m, 2 lanes, lane max speed 34 m/s
- [ ] 30 vehicles, closed loop: re-enter at road start with the same ID
- [ ] Per-vehicle constant desired speed ~ U[17, 34] m/s tied to ID, sigma = 0
- [ ] RSU at roadside midpoint, Loc_y = 10 m
- [ ] Fleet file: 15 Xavier / 10 Orin / 5 Thor
- [ ] Export trace: position, speed, direction relative to RSU, per timestep
- [ ] Scenario 2 reuses the same trace; only the coverage predicate changes
- [ ] Pre-RunPod gate: realized speed covers 17–34 m/s; dwell-time histogram per scenario; Scenario 2 feasible-and-idle pool ≥ 2×C; ID → tier → sleep_s mapping matches fleet file

## 2. Implementation (PLATO)

Environment model (ground truth, shared by every arm)
- [ ] Upload time per Wen Eq. 5–6: d_k at training finish, φ=3, B=2 MHz, p=23 dBm, N=−114 dBm, Rayleigh χ_k; download ignored
- [ ] Energy: E = P_tier × T_train + p × T_up
- [ ] Upload policy: `always_accept` (S1), `coverage_gated` (S2)
- [ ] Clock reconstruction includes T_up

Estimators
- [ ] T̂_train GPU-TOPS (exists)
- [ ] T̂_train CPU-cycle per Wen Eq. 4 (b·|S|·ρ / f)
- [ ] T̂_up from Eq. 5–6 using position at decision time, shared by all arms

Stage 0
- [ ] v_max → 34 m/s
- [ ] Energy constraint E ≤ E_max
- [ ] Pluggable into Wen's selector

Wen selector
- [ ] Wen-faithful (sync FedAvg + many-objective search F1–F4 + Eq. 10 constraints)
- [ ] Wen-async port (score per vehicle, fill slots)
- [ ] Wen + GPU (T_com replaced by GPU estimate in F1 and Eq. 10)
- [ ] F4: clients report label histogram; server holds global test distribution

Metrics
- [ ] TTA@80%, waste, energy per accuracy, dispatch ratio, time-resolved seconds-per-round (before/after threshold)

## 3. Experiment plan

| Scenario | Experiment | Arms |
|---|---|---|
| 1: 100% coverage | Surgical | random, cpu, gpu (our selector) |
| 1: 100% coverage | Main comparator (2×2) | Wen-faithful, Wen + GPU, Ours + CPU, Ours |
| 2: Partial coverage | Stage 0 | Ours ±Stage 0, Wen ±Stage 0, random |
| Ablation | History-based | Polaris/observed, gpu_oracle |

- [ ] Pre-register predictions; Scenario 1 headline = beats cpu signal on TTA (Stage 10 closed: gpu was 6.9% behind random on TTA)
- [ ] ≥ 3 seeds per cell
- [ ] Round budget at N=30, C=3 (check plateau within 4,000 rounds)
- [ ] CIFAR-10 first, then CIFAR-100 on the new environment
- [ ] Adversarial data-to-GPU mapping, 1–2 runs in Scenario 2

## 4. Writing (Chapter 3)

- [ ] N=30 cites Wen; C cites Pisces + Polaris; Wen is synchronous so has no concurrency; cite FedBuff on concurrency ≠ participation fraction
- [ ] 2,000 m + 17–34 m/s cites Wen §5.3.4, with why not 200 m (at highway speed a 200 m road is crossed in 5.9–11.8 s, shorter than one Xavier training job; cannot host Scenario 2)
- [ ] 100% coverage follows Wen; partial coverage is our extension (Wen uploads succeed at any distance, though a vehicle travels up to ~1 km within T_max = 30 s)
- [ ] Claims: C1 GPU vs CPU signal; C2 Stage 0; C3 history-based in ablation (warm-up argument: k* ≈ 30–70 obs/client ≈ 300–700 rounds at N=30, C=3)

## Dropped

Open population; 5-environment p_return sweep; Stage 12 (G_i reset on re-entry); Exp 3.7 (p_recur = 0 estimator gate); open-population CIFAR-100 pilot.
