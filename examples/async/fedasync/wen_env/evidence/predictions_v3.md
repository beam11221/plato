# Pre-registered predictions v3 (locked 16 Oct 2026, before any full run)

Source: planning chat, copied verbatim from `claude/roadmap-wen-environment.md` (project doc). Commit this file before any run of ≥ 4,000 rounds and record the commit hash in every run manifest. Do not edit after the first full run starts.

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
