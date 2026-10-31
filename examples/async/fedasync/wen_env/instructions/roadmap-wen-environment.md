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
| 15 | S1 | gpu, ε = 0 (ablation) | ours, pure top-1 | GPU-TOPS + T̂_up | off |
| 16 | S1 | gpu, proportional (ablation) | ours, sample ∝ V | GPU-TOPS + T̂_up | off |

16 arms × 3 seeds = 48 runs (location-only still deferred). All 'ours' rows use ε-greedy, ε = 0.2, unless marked. Location-only is the null for the speed signal: gpu vs location-only = what GPU adds; cpu vs location-only = whether CPU helps or hurts; location-only vs random = what position adds. Wen-async variants (Section B) add 2 arms × 3 seeds later.

## Pre-registered predictions (v3, 16 Oct 2026 — lock before any full run)

Setting: as v2 (N = 30, C = 3, CIFAR-10 + ResNet-18, E = 1, η = 0.2, measured FLOPs_fwd 1.113 G → T_train 2.784 / 0.357 / 0.186 s, upload per Wen Eq. 5–6, T̂ = T̂_train + T̂_up, ρ = 1,250, CPU 2.2 / 2.2 / 2.6 GHz, Dirichlet α = 0.8 with the disjoint sampler, G_i prior α = 1, σ = 0.149), plus the **selection rule decided 16 Oct**:

> **ε-greedy, ε = 0.2**, applied identically to every arm that uses our score (gpu, cpu, gpu_oracle, observed, location_only): at each free slot, with probability 0.2 pick uniformly among idle eligible vehicles, otherwise pick the idle eligible vehicle with the highest V_i = G_i / √(1 + τ̂_i). The ε coin and the uniform pick use an RNG keyed by (seed, round, k). Ties at the top go to the lower client id. random = the ε = 1 endpoint.

Why the change: the smoke run showed proportional sampling on V ∝ 1/√(1+τ̂) collapses to random once upload compresses V to ~0.45–0.56 (gpu −1.3% vs random; toy model −1.4%). ε-greedy uses only the ranking of V, so its selectivity does not depend on how wide V is. Pure top-1 starves Xavier (toy: 2% of dispatches against 50% of the data); ε = 0.2 keeps every vehicle's per-decision chance ≥ 0.2/27.

Rule: a prediction holds if the direction is the same in all 3 seeds; report mean ± std. Magnitudes are not predicted; toy numbers (no G_i, no fair rotation) are context only.

### Scenario 1 — 100% coverage

- **P1 cpu is closer to random than to gpu:** cpu's seconds-per-round gain over random is less than half of gpu's. Why: T̂_cpu_train ≈ 23 / 23 / 20 s cannot separate Xavier from Orin and outweighs most of the upload signal (toy, ε = 0.2: cpu ≈ −6%, gpu ≈ −32%).
- **P2 gpu beats cpu** on seconds-per-round and TTA@80%.
- **P3 gpu beats random on seconds-per-round**, both before the 80% crossing and over the whole run, and **on TTA@80%**. If a whole-run gain is absent while the early gain is present, report it as the fair-rotation pay-back seen in Stage 3, not as a failure of the estimator.
- **P4 Plateau:** gpu within ±1 point of random.
- **P5 Energy to reach 80%:** gpu < random. Per-task training energy ≈ 107 / 29 / 26 J (Xavier / Orin / Thor); gpu dispatches Xavier far less often than random.
- **P6 Transplant:** Wen + GPU reaches 80% sooner than Wen-faithful and has lower seconds-per-round (Wen's own selection rule, unchanged by this decision).
- **P7 gpu ≈ gpu_oracle.**
- **P8 observed does not beat gpu in any phase** (its history is total time including position-dependent upload).
- **P12 Location-only (conditional — only if run, same seeds and config):** random < location-only < gpu in seconds-per-round gain (toy, ε = 0.2: ≈ −24% vs −32%).
- **P13 Selection-rule ablations (gpu arm only):** proportional sampling ≈ random on seconds-per-round (|Δ| within seed spread); ε = 0 has a seconds-per-round gain ≥ ε = 0.2; ε = 0 plateau ≤ ε = 0.2 plateau.

### Scenario 2 — partial coverage (1,200 m), `coverage_gated`

- **P9 Stage 0 reduces waste** for gpu and for Wen-faithful. *Disclosure:* the direction for gpu was already seen in the smoke run (1 seed, 200 rounds: 19.2% → 1.0%), so for gpu this is a confirmation, not a blind prediction; for Wen-faithful it is blind.
- **P10 gpu + Stage 0 waste is near zero.** cpu + Stage 0 admits fewer dispatches per unit time than gpu + Stage 0 (its T̂ is ~3× the true total time, so it rejects vehicles that would finish). Energy-cap behaviour of cpu + Stage 0 depends on the pending energy-estimate decision.
- **P11 Stage 0 does not lower the plateau.**

### Changes from v2
- Selection rule: ε-greedy (ε = 0.2) replaces proportional sampling for every arm using our score
- P1 restated as "closer to random than to gpu" (top-1 lets cpu exploit its weak correct preference for Thor)
- P3 strengthened: gpu predicted to beat random on whole-run seconds-per-round and on TTA@80% (v2 accepted a tie)
- P13 added for the selection-rule ablations
- P9 discloses that the gpu direction was seen in the smoke run

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

## Prompt B smoke run (16 Oct) — plumbing passes; one design issue surfaced

- FLOPs_fwd 1.113 GFLOP (+1.2%); T_train 2.784 / 0.357 / 0.186 s; D = 357.6 Mbit; upload table reproduced; online = replayed clock (0.0 s); 72/72 tests
- Upload share of task time 0.85–0.87 (communication-bound confirmed). gpu T̂ per tier 12.1 / 9.25 / 9.18 s vs true 12.3 / 9.8 / 9.5 s — the estimator is accurate
- **S1: random 3.77, cpu 3.73, gpu 3.72 s/round — gpu ≈ random (−1.3%).** Cause: the selector samples ∝ V_i (not top-K), and V = 1/√(1+τ̂) now spans only ~0.45–0.56, so sampling is near-uniform. Toy model reproduces it: proportional −1.4%, top-1 −40%, location-only top-1 −30%. The May design said top-K deterministic — check when / why it became proportional. Decision needed before the main run (selection rule), with the other rule kept as an ablation; predictions → v3
- 16 Oct: author agrees to move to top-K but wants exploration kept (reason for weighted sampling: let low-score clients still be picked). Toy model (no G_i): pure top-1 starves Xavier (2% of dispatches vs 50% of data; min client 0.13%) because some Orin/Thor is almost always near the RSU. Options: Oort cut-off sampling c = 95% (−37%, Xavier 5%) / c = 90% (−30%, 11%); ε-greedy top-1: ε 0.1 → −36% (X 7%), 0.2 → −32% (X 12%, min client 0.71%), 0.3 → −28% (X 17%). **Decided 16 Oct: ε-greedy, ε = 0.2** for every arm using our score; ε = 0 and proportional as ablations; random = ε = 1 endpoint
- S2: waste 19.2% → 1.0% with Stage 0; s/round 4.19 → 3.31 (−21%); feasible-idle median 12–13 ≥ 6
- **Decided 16 Oct:** cpu arm's Ê (was P_GPU × T̂_cpu = 931 / 1,746 / 2,561 J, blocked by E_max = 600 J) — each arm estimates energy with its own model family (cpu arm: Wen Eq. 7 + p·T̂_up ≈ 25 / 25 / 35 J + ~2 J), keeping E_max non-binding
- To document: disjoint Dirichlet sampler with Sinkhorn (mean TV 0.063 from the Dirichlet draw) needs its own ADR; ADR-0005 needs the SUMO ADR text copied over

## Finding (16 Oct): PLATO `noniid` sampler overlap

PLATO's `noniid` sampler seeds every client's torch generator with the same seed (dirichlet.py line 81 vs per-client seed on line 21) and samples each client independently from the whole set; at N = 20 × 2,500, α = 0.8, seed 4126 the fleet held only 8,453 distinct images (16.9%), with two clients sharing ~49% of their images on average. Every stage before the Wen environment used it. The Wen environment uses a disjoint per-class Dirichlet split with Sinkhorn equalisation (all 50,000 images used once; TV 0.063 from the Dirichlet draw). Disjoint per-class splitting is the standard meaning of a Dirichlet partition (Yurochkin et al. ICML 2019; FedMA, Wang et al. ICLR 2020). Consequences: between-arm comparisons within an old stage stand; absolute plateaus and cross-stage comparisons do not; the Stage 10 explanation "excluding Xavier is free because data is redundant" was likely produced by this overlap and must not be carried over — with disjoint data Xavier holds half of the distinct images, which raises the stakes of P4 / P11 and the adversarial data-to-GPU mapping test. 80% target was chosen against ~81–83% plateaus; new plateau ~90% — revisit only after Phase A, without changing the locked v3 target.

## Phase A results (seed 4126, 17 Oct) — directions mostly as predicted; mechanism is position, not tier

- S1 TTA@80%: gpu 1,158 s vs random 1,419 (−18.4%), cpu 1,535, observed 1,426, gpu_oracle 1,168, ε=0 1,117, proportional 1,386. All arms cross 80% at rounds 370–430 → differences are seconds-per-round
- **Mechanism:** gpu's mean T_train per task equals random's (1.38 s); its gain is uploading closer to the RSU (mean d 284 m vs 492 m; 84% dispatched inside [400,1600] vs 61%). Tier signal only early (Xavier 32% vs 43% in rounds 1–500). cpu cannot use position because ρ = 1,250 puts T̂_cpu_train (20–23 s) above every upload time
- S2: gpu + Stage 0 fastest cell (TTA@80% 1,109 s, −38.8% vs S2 random; waste 0.81%); cpu + Stage 0 rejects 73%, picks far vehicles (d 392 m), 33 fallback cycles
- Late phase: gpu needs more rounds to 88% (1,300 vs 1,060) and loses the time edge there — single seed, noisy; possibly data-coverage pay-back (disjoint data: Xavier holds half the distinct images)
- Protocol deviations: runs stopped at 90% accuracy (author's choice) → no plateau, P4/P11 not testable as written; S1 gpu and S1 cpu hung on disk-full at 89.75% / 89.68%; pilot plateau gate failed and was overridden
- Framing (17 Oct): the training-time → finish-position step is Wen's own (§3.3, Fig. 2: d_k = √((Locx_k ∓ T_com·V_k)² + Locy_k²), with T_com from the CPU model Eq. 4). So in Wen's model a wrong compute estimate propagates into a wrong distance and upload estimate. Expect P6 (Wen + GPU vs Wen-faithful) to act through both F1 and d_k
- Proposed before Phase B: (1) find why the disk filled despite the cleaner; (2) decide run budget so P4/P11 are measurable (fixed rounds for all cells, rerun seed 4126 under it); (3) run location-only (now essential: gpu − location-only = what GPU knowledge adds); (4) add cpu with scale-calibrated ρ ≈ 85 cycles/bit (fleet-mean T̂_cpu = true 1.54 s) so cpu vs gpu isolates tier ordering from scale; (5) optional compute-bound regime check (E = 5) for random / location-only / gpu; write predictions v3.1 for added arms before Phase B

## Main runs v1 — 3 seeds (17 Oct)

S1 s/round vs random (3.773): location-only −11.2%, gpu −14.0% (SD 0.01), ε=0 −14.7%, oracle −13.3%, proportional −1.4%, observed −2.2%, cpu −1.0%. TTA@80% vs random (1,733 ± 350 s): gpu −24%, location-only −17%, ε=0 −19%, cpu −5%, observed −6%. Energy to 80%: gpu 22.2 kJ, random 30.2, location-only 32.5 (Xavier 56% early — highest of any arm), cpu 28.7.
Decomposition: position ≈ 11 points of s/round; GPU tier knowledge adds ≈ 3 points of time (gpu < location-only in every seed) and ≈ 32% less energy than location-only. Tier effect is early: Xavier share rounds 1–500 gpu 30% vs random 46%; rounds 1,500–2,000 gpu 52% ≈ random 49%; after 80% gpu s/round 3.31 ≈ location-only 3.38.
S2: gpu + Stage 0 s/round −30% and TTA −27% vs random; waste 22.7% → 12.3% (gpu) → 0.9% (gpu + Stage 0); cpu + Stage 0 rejects 73%.
P-table (v3 rule): held P1, P2 (s/round), P3 (all three), P4, P5, P8, P9, P10, P12; not held P2 (TTA: seed 4128 tie 1,550 vs 1,549 s), P11 (seed 4126 −0.11 pt), P13 (ε=0 not reliably ≥ ε=0.2; proportional slightly faster than random beyond random's SD). P4/P11 measured on a 90%-stop "plateau".
17 Oct: author approved cpu_cal (Prompt D + predictions v3.1 addendum P14a–d written). Author's sequence afterwards: (1) holistic comparison Ours vs Wen-faithful vs random, Wen and random without any of our components; (2) plug Stage 0 into random and Wen, check final accuracy same/better and waste lower; (3) GPU-equation / compute-bound questions later. Open issues flagged for (1)–(2): Wen's own T_max = 30 s / E_max = 20 J reject every vehicle under our parameters (CPU T̂ ≈ 20–23 s + upload; CPU energy 25–35 J); "final accuracy" is undefined under the 90% stop rule — needs a fixed wall-clock budget; Stage 0 only acts in S2 (S1 has no exit); Stage 0 for random/Wen must use our GPU T̂ to be "our module".
Next (earlier list): cpu with calibrated ρ ≈ 85 (separates scale from tier order); energy per dispatch and mean upload distance per arm from existing logs; Wen comparator (Section B); optional compute-bound check (E = 5).

## Wen comparator — decision log (opened 17 Oct)

Goal (author): holistic comparison Ours vs Wen-faithful vs random, with Wen and random carrying none of our components. Decide in order; each item records options, choice, reason.

| # | Decision | Status |
|---|---|---|
| W1 | Aggregation / timing | **decided 17 Oct: async** — borrow Wen's equations and optimizer into our FedAsync framework |
| W2 | Participants | **decided 17 Oct: N = 30, C = 3**, same as all previous runs |
| W3 | Optimizer | **decided 18 Oct:** exact enumeration of RA-MaOVSM's decision space (≤ 27 single picks; ≤ 2,925 triples at start-up) → exact Pareto set → Wen's final rule (smallest normalised mean of F1–F4). Normalisation not specified by Wen (only "smallest normalized mean", §5.3.2, and Fig. 3 "value after normalization"); proposed min–max per objective over the Pareto set of each decision, (f − min)/(max − min), range 0 → 0, single-member set → that member. Consistent with RVEA's own scaling by the population's objective range (Cheng et al., IEEE TEVC 2016). Known property: min–max gives a tiny spread full weight | **decided 18 Oct** (exact enumeration + min–max over the Pareto set, confirmed) |
| W4 | Objectives, per candidate vehicle (single pick ⇒ set aggregates reduce to the vehicle's own value). F1 = T̂_com + T̂_up with T̂_com = b·\|S\|·ρ/f (ρ = 1,250) and d from Wen's Loc_x ± T̂_com·V_k; F2 = Wen Eq. 7 + 8 (CPU energy + p·T̂_up); F3 = vehicle's last-reported training loss, lower preferred (Wen), prior for a never-trained vehicle = fleet mean of reported losses (all equal at start ⇒ F3 inactive); F4 = KL(P_label ‖ Q) with P = vehicle's label histogram, Q = CIFAR-10 test distribution (uniform 0.1) | decided 18 Oct (F3 lower-is-better as Wen; never-trained prior = fleet mean; Q = CIFAR-10 test distribution, uniform 0.1 — all confirmed) |
| W5 | Constraints Eq. 10: Wen's T_max = 30 s / E_max = 20 J reject every vehicle under our workload (E_max rejects all; T_max passes X/O only within ~80 m of RSU, Thor within ~670 m). User chose to raise them (option C, 18 Oct). Proposed rule, fixed before any Wen run: each limit = q-th percentile of that objective over the static fleet × upload positions (estimates only, |x−1000| uniform 0–1000 m, Loc_y 10 m), q = 90 → T_max = 34.33 s, E_max = 36.44 J (80% of vehicle-positions feasible; X/O 88%, Thor 40%). Caveat: F2 is bimodal (X/O 25.8–27.1 J, Thor 35.6–36.9 J), so any E_max in 27.1–35.6 J bans Thor entirely (q = 75 → 27.01 J does). Fallback if no idle candidate is feasible: drop constraints for that decision. Optional sensitivity: q = 75. Per-tier check (analytic, uniform upload distance): limits are fleet-wide, not per tier; T_max binds only X/O (upload > ~880 m from RSU, cuts 12% of X/O), E_max binds only Thor (upload > ~400 m, cuts 60% of Thor); each constraint cuts 10% marginally, 20% jointly (disjoint). q = 95 would give joint 10% (34.53 s / 36.70 J; X/O 6%, Thor 30%). Prompt E must recompute q from the SUMO trace + Wen's predicted upload position and log per decision: n_candidates, n_feasible, rejected-by-T / -by-E / both per tier, fallback flag, F1–F4 of every candidate | decided 18 Oct: q = 90 (confirmed after per-tier check; marginal 10% per objective, joint 20%) |
| W6 | Ground truth for the Wen arm = same environment truth as every arm (GPU time, Wen upload, GPU energy); Wen's own estimates stay CPU-based. Question answered: how well Wen's selector works in a GPU world | decided 18 Oct |
| W7 | S2 handling: eligibility = in coverage at dispatch (environment rule); `coverage_gated` uploads; no Stage 0 in Wen-faithful (Stage 0 plug-in claim is tested separately in the Stage 2 plug-in study) | decided 18 Oct |
| W8 | Comparison basis: wall-clock (online clock) TTA@80%, energy to 80%, waste; same 90% stop rule; TTA reported at 80%, 85% and 90% (also recomputed from logs for the existing 36 runs + cpu_cal); s/round; mechanism logs as Prompt D Step 4 + W5 constraint logs | decided 18 Oct |
| W9 | Random baseline = existing async random (ε = 1, C = 3, FedAsync, S1 + S2 Stage 0 off, 3 seeds) — reused, no new runs; Wen's sync FedAvg baseline not used. Main comparison new runs = Wen-faithful S1 + S2 × 3 seeds = 6 runs | decided 18 Oct |
| W10 | Local training and aggregation details: same local config as all arms (ResNet-18, E = 1, same batch/lr); aggregation = the same FedAsync mixing as every arm (corrected from 'FedAvg ∝ |S_k|'); only the selector differs | decided 18 Oct |

Later (after the holistic comparison): Wen + GPU transplant (P6), Wen + Stage 0 (plugin study).

18 Oct decisions (after W1–W10):
- Study 1 (holistic: Ours vs Wen-faithful vs random, no components of ours in Wen/random) → Prompt E + predictions v3.2 addendum (P15–P21). New runs: Wen-faithful S1 + S2 × 3 seeds = 6. Random and gpu (S2 Stage 0 off) reused.
- Study 2 (Stage 0 plug-in: random ± Stage 0, Wen ± Stage 0) — prompt written only after Study 1 results. Stage 0 plugged into random / Wen uses **our** estimator (GPU T̂_train + T̂_up, V_k), not the host selector's estimates. Metric: TTA@80/85/90 + waste under the same 90% stop; "final accuracy at fixed wall-clock budget" deferred.
- Not yet run: S2 random + Stage 0, S2 Wen + Stage 0.
- Expected mechanism (written before runs, in predictions v3.2 context): Wen predicts the upload position with CPU-scale T̂_com ≈ 20–23 s while true training is 0.2–2.8 s, so it picks vehicles ~400–760 m upstream of the RSU; true upload distance should look like the cpu arm, not location-only.

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

## Study 1 results — Wen-faithful v1 (18 Oct, `evidence/wen_faithful_v1.md`, predictions v3.2 commit 66a2ed82)

- P15–P21 all held (3/3 seeds). T_max = 34.334 s, E_max = 36.424 J (trace-based, matches planning); fallback rate 0.
- S1 TTA@80%: gpu 1315 ± 207, random 1733 ± 345, wen 2772 ± 150 s. Rounds to 80%: gpu ≈ 440, random ≈ 463, wen ≈ 770, while wen s/round 3.63 vs random 3.77 → wen's loss is statistical (rounds), not system time. S1 wen never reached 90% (last 0.824 / 0.753 / 0.858; max ≤ 0.872 by 4,000 rounds).
- Participation collapse: S1 wen Gini 0.816 (random 0.069, gpu 0.125), one vehicle ≈ 999 of 4,000 dispatches, ~2 vehicles never picked. S2 wen Gini 0.657 (coverage forces rotation) → reaches 90% but at 12,924 s vs 6,869 (gpu) / 7,685 (random).
- Mechanisms seen: x̂ error 574 m (gpu 5 m) → true upload distance 513 m ≈ random 495 m (P15); Thor share 0.00 (E_max + highest F2); X/O split seed-dependent (Orin 0.67 ± 0.26) → wen's small s/round gain over random comes from which vehicles it locked onto, not from position.
- S2: wen waste 5.5% (lowest; gpu 12.3%, random 22.7%) because the misplaced x̂ pushes picks to the coverage entry (long residence) — right result, wrong reason. S2 TTA@80 wen 1608 between gpu 1416 and random 1817; crossover later (TTA@90 worst).
- Wen paper check: Algorithm 3 computes f3 by actually training each candidate on the current global model (fresh loss); we use the last reported loss (stale) because fresh f3 needs every candidate to train each decision. Wen reports accuracy for the first 30 federated rounds only (Fig. 4) — long-run concentration would not show there.
- Threat to the claim: part of gpu's win over wen comes from wen's determinism + low-loss feedback (not GPU-related). Next to separate: log-only concentration diagnosis (F3 vs F4 driver), Wen + GPU transplant (P6), optionally Wen + ε = 0.2.

18 Oct (after Study 1): author approved (1) log-only collapse diagnosis → Prompt F (`claude/prompts/prompt-F-wen-concentration-diagnosis-runpod.md`; per-vehicle table, rolling Gini, static drop-one-objective counterfactual, label coverage of aggregated updates, footnote reconciling the two disk-hang runs). (2) **Study 2 regrouped = plug-in study of our two components**: (a) Stage 0 (S2: random ± Stage 0, Wen ± Stage 0), (b) GPU-aware estimator plugged into Wen (Wen + GPU, S1 and S2; replaces Wen's CPU F1/F2, keeps F3/F4 and determinism). GPU estimator does not apply to random (no estimator to replace). Possible combined cell: Wen + GPU + Stage 0 (S2). Prompt for Study 2 written after Prompt F results.

## Prompt F results — Wen collapse diagnosis (18 Oct, `evidence/wen_concentration_v1.md`)

- Replay test PASS (23,472 decisions). Reading B adopted for the two disk-hang runs (counted as 90%).
- Drivers: F3 and F4 both consistent (dispatch vs F4 rank ρ −0.51…−0.83; vs mean F3 ρ −0.39…−0.95; dropping F3 changes 26–54% of picks, dropping F4 27–51%, both lower the static-replay Gini). F1/F2 are NOT drivers: dropping them raises Gini; picks are F1-min in only 1–9%; most-picked vehicles have worse F1 (ρ +0.77…+0.88) — data objectives override position.
- Onset immediate in S1 (top vehicle >20% from round ~20 in 2 seeds). Reading: at start F3 is inactive (all equal) → F4 picks the lowest-KL vehicles → their reported loss falls (top-5 F3 0.40 → 0.01) → F3 locks them in.
- **Validity threat found:** the rest of the fleet keeps F3 ≈ 0.79–0.95 for the whole run — stale losses from the early rounds that are never refreshed because those vehicles are never picked (random's round-level loss is 0.13–0.32 by r1000–2000). Wen's Algorithm 3 recomputes f3 every round by training each candidate on the current global model, so our stale F3 likely amplifies the lock-in. A fresh-F3 robustness arm is needed before Study 1 can be claimed.
- Label coverage: wen aggregated-label KL to uniform 0.008–0.034 vs ≤ 0.0098 for random/gpu; 63–80% of images covered by r250 (random/gpu 100%); 3/6 wen runs never cover all images (86.7–96.7%). S1 seed 4127 has the lowest coverage (86.7%) and the lowest final accuracy (0.753).
- Implication for Study 2(b): plugging the GPU estimator into Wen changes F1/F2, which do not drive the collapse, so Wen + GPU may show little TTA change; consider a system-objectives-only Wen variant to isolate the plug-in effect.

18 Oct: author confirmed F3 is reassigned after each merged training (never-trained prior = fleet mean), but refresh happens only when a vehicle is picked → stale F3. Approved Study 1b `wen_fresh` (F3 = current global model's loss on each vehicle's data, recomputed per global version, not charged to the clock) → Prompt H + predictions v3.3 (P22–P25). Prompt G name was already taken by the audit-bundle prompt. Prompt H Step 0 also asks the code to confirm the exact prior definition (Thor never-trained F3 ≈ 0.34 looked low). Next: Study 2 design discussion.

## Study 2 decision log (plug-in study)

| # | Decision | Status |
|---|---|---|
| S2-1 | Wen host version: decide after Prompt H — if P22 not held (fresh ≈ stale) use `wen`; if held use `wen_fresh` | decided 18 Oct (option c) |
| S2-2a | GPU plug-in = replace Wen's F1/F2 with our GPU family: F1 = T̂_train(GPU) + T̂_up with x̂ = x + V·T̂_train(GPU); F2 = P_tier·T̂_train + p·T̂_up. F3/F4, Pareto, normalisation, determinism unchanged | decided 18 Oct |
| S2-2b | Constraint limits under GPU estimates: keeping 34.33 s / 36.42 J bans every Xavier (F2 112–114 J) and never binds T; re-applying the W5 rule (q = 90 on the host's own estimates) gives T_max ≈ 13.57 s, E_max ≈ 113.5 J, cutting 20% of Xavier positions and no Orin/Thor | decided 18 Oct: re-apply q = 90 on the host's own estimates |
| S2-2c | (i) Wen + GPU is the plug-in result. (ii) Wen-sys (F1 + F2 only, CPU vs GPU, S1) is optional, lowest priority, diagnostic only — run last if compute is free. Principle (author): test Wen as proposed and report honestly if our component does not help; do not re-engineer Wen to suit it (that would turn it into our proposal). `wen_fresh` is consistent with this — it restores Wen's own per-round f3, it is not an improvement | decided 18 Oct |
| S2-3 | Cells (host = per S2-1): 2a Stage 0 (S2): random + Stage 0, Wen + Stage 0. 2b GPU plug-in: Wen + GPU (S1, S2). Combined: Wen + GPU + Stage 0 (S2) → completes {CPU, GPU} × {Stage 0 off, on} for Wen in S2. 5 cells × 3 seeds = 15 runs (+ 6 optional Wen-sys). Stage 0 order inside a host: eligibility → Stage 0 (our estimator) → host's own constraints → host's selection | cells decided 19 Oct (see below); filter order proposed |

## Study 1b results — wen_fresh, seed 4126 only (19 Oct, `evidence/wen_fresh_v1.md`, predictions v3.3 commit 77e2d7b7)

- Step 0: `wen` prior = running mean of every loss ever reported (1.86 at r8 → 0.13 at r4000); a vehicle picked once keeps its own early loss (e.g. 1.07 for ~3,500 rounds). 0 mismatches over 70,258 rows.
- Collapse NOT fixed: Gini S1 0.834 → 0.819, S2 0.679 → 0.662 (Δ ≈ −0.015, smaller than wen's seed SD 0.019); same top-5 vehicles (16, 19, 22, 5, 21 in S1); onset unchanged. Fresh top-5 F3 ≈ 0.04 vs rest ≈ 0.46 — the loop is real (global model fits the picked vehicles), not a staleness artefact. Audit-G H1 ("collapse driven largely by our F3 adaptation") is not supported by this seed.
- Fresh F3 locks out never-picked vehicles harder: S1 6 vehicles never picked (4 Thor), image coverage 80% (wen 100% by r2000). The stale running-mean prior had acted as accidental exploration.
- TTA improved: S1 TTA@80 2625 → 1967 s (outside wen's 3-seed range 2625–2925), still > random 1419 / gpu 1158; S1 still not 90% (last 0.850). S2 TTA@80 1145 s < gpu (Stage 0 off) 1348 and random 1813 → P25 direction failed in this seed; at 85% / 90% wen_fresh is worse (3108 / 10004 vs gpu 2254 / 6931). S2 waste 3.6% (gpu 12.3%), energy to 80% 19.8 vs 24.7 kJ. gpu + Stage 0 (same seed) TTA@80 1109.
- Step 6 static replay: F1 + F2 only → Gini 0.29–0.31 (S1), 0.44–0.47 (S2); F3 + F4 only → 0.88 (S1), 0.74–0.78 (S2).
- Cost: fresh evaluation ≈ 7.8 h extra per 4,000-round run (12.4 h total wall time).
- Open (from audit H2, now the main remaining objection): Wen selects 21–24 of 30 per round (C = 0.7–0.8); our async C = 3 picks 1 of ~28 — a deterministic rule concentrates at 10% participation by construction.

19 Oct decisions (author): (1) Wen host for Study 2 = `wen_fresh`. (2) Cell 5 Wen + GPU + Stage 0 included, run last. (3) Priority = Stage 0 plug-in first (random + Stage 0, wen_fresh + Stage 0, S2), then Wen + GPU (S1, S2), then cell 5. (4) Seed 4126 only first; other seeds later. (5) wen_fresh seeds 4127 / 4128 after Study 2 seed 4126 (Study 1b v3 verdicts pending until then). Naming: "Ours" = gpu + Stage 0 in every scenario (Stage 0 is a no-op in S1 — to be verified by a 200-round S1 test, rejection rate 0, identical dispatch log).
19 Oct: pipeline confirmed by author — eligibility (environment: idle + in coverage at dispatch, every arm, every S2 experiment) → Stage 0 → scoring/constraints → selection. Prompt I (Study 2, seed 4126) + predictions v3.4 (P26–P33) written; Prompt I Step 0a verifies eligibility held in every existing S2 run, 0b audit-G housekeeping (commit, source hash, replay, separation test), 0c Stage 0 no-op in S1.

## Study 2 interim — Study 2a, seed 4126 (20 Oct, `evidence/study2_plugin_s4126.md`)

- Step 0a: 0 dispatches outside coverage and 0 to a busy vehicle across all 24 S2 runs → eligibility held everywhere. 0b: 200-round replays identical for 6 arm families (commit 4fd265dd, clean) → provenance restored. 0c: Stage 0 in S1 rejected 0 of 5,602 checks, dispatch log identical → "Ours = gpu + Stage 0" in every scenario.
- random + Stage 0 vs random (S2): waste 23.1% → 1.4%; TTA@80 1813 → 1324 (−27%), @85 3002 → 2260, @90 7833 → 6242 (−20%); s/round 4.43 → 3.32; energy to 80% 34.6 → 25.7 kJ; Gini 0.057 → 0.079. P26, P27 direction as predicted.
- Decomposition at S2 seed 4126 (TTA@80 / @90): random 1813 / 7833; +Stage 0 1324 / 6242; gpu 1348 / 6931; gpu + Stage 0 1109 / 5279 → Stage 0 alone ≈ GPU scoring alone at 80%, larger at 90%; combined sub-additive.
- wen_fresh + Stage 0 vs wen_fresh: waste 3.6% → 0.2%; TTA@80 1145 → 1149 (same), @85 3108 → 2642, @90 10004 → 8387 (−16%); Gini 0.662 → 0.637. P28 direction as predicted.
- Stage 0 rejects ~24–25% of eligible vehicles, all by the time test; the energy test (E_max = 600 J) never binds in any cell — disclose.
- Smoke hint for 2b (not a result): wen_fresh + GPU fixes x̂ (|x̂ − x| ≈ 1 m vs ≈ 500 m) and upload distance (S1 273 vs 456 m), s/round 2.96 vs 3.46, but dispatches 0% Xavier (GPU F2 ≈ 112 J → nF2 = 1 in every Pareto set) and only 12–13 distinct vehicles → the exclusion flips from Thor to Xavier (half the fleet). Expect a lower accuracy ceiling.
- Deviation: cells 3–4 hung on EIO (MooseFS) at r3250 / r3030, rerun from scratch; cell 5 started alongside instead of after. Watchdog logs STALL but does not stop runs.

## 21 Oct — lab direction and Polaris comparator

Lab feedback (author, 21 Oct): anchor on Scenario 1; the GPU plug-in work in Study 2 is good; S2 later. Most urgent next step: **Polaris as a third main comparator in S1** (alongside random and Wen). Claude Code read-only study: `polaris_integration_findings.md` (Plato `examples/client_selection/polaris/`).

Paper check (Kang & Li, TCC 2024, P4 Eq. 23): min_q Σ p_i² G_i² / q_i + A·Σ p_i² q_i τ_i G_i, s.t. Σ q_i = 1, q_i > 0; p_i = aggregation weight with Σ p_i = 1 over all N clients (n_i / n_total); "plug-and-play … combined with various aggregation methods"; single hyperparameter insensitive over 0.01–10. Note: the paper's staleness term is linear in τ_i and G_i; Claude Code transcribed the code's term as w²·τ²·G·p — verify which the code uses.

Proposed fidelity rule: paper first; where the paper is silent, the authors' own Plato code (same group); where the code contradicts the paper, follow the paper and document the difference.

| # | Decision | Status |
|---|---|---|
| PL1 | Host = paper formulation P4 (q_i ∝ √(f1_i / (f2_i + μ)), μ by bisection; unit test vs a correct GP solve). Plato-code variant (F1: constraints on log-variables → concentrated p) documented, optional | proposed |
| PL2 | Aggregation = the environment's FedAsync mixing for every arm (paper: plug-and-play) | proposed |
| PL3 | p_i = fleet-static n_i / Σn (paper definition) | proposed |
| PL4 | Paper-silent parameters from Plato defaults: α = 10 (unseen-client G proxy; needs exception to the α = 1 lock), initial G = 0.5, initial τ = 0.01, +0.1 τ offset, A (λ) = 1; τ exponent per paper (linear) | proposed |
| PL5 | Dedicated RNG default_rng([seed, round, k, SALT]); one draw per free slot, renormalised over remaining candidates | proposed |
| PL6 | Scope: arm `polaris`, S1, seed 4126 first; comparators random, gpu (Ours), observed, wen_fresh. `polaris_gpu` (τ from our estimator) as the Study 2b analogue, next | proposed |
| PL7 | Per-decision log `polaris_decisions.csv` (pool, q vector, μ, G_i, τ_i, p_i, pick, q of pick); dispatch-log `score` = q_i | proposed |
| PL8 | Solve once per selection call, renormalise for extra picks in the same call | proposed |

Expectation (context, not yet a prediction): the paper formulation spreads q (median max q 0.075 in Claude Code's sweep), and our smoke/main runs showed probability-proportional sampling ≈ random in this comm-bound regime (gpu_prop s/round 3.721 vs random 3.773) → Polaris likely near random on s/round; any gain would come from G_i (rounds).
21 Oct: author confirmed the fidelity rule and PL1–PL8 → Prompt J (`claude/prompts/prompt-J-polaris-s1-runpod.md`) + predictions v3.5 (P34–P37: gpu < polaris on s/round, TTA@80, energy to 80%; polaris Gini < gpu). `polaris_plato` (code solver) implemented/tested only, not run.
