# Prompt B (v2, 16 Oct 2026) — Environment model for the Wen-aligned closed-population experiments (RunPod)

> v2 changes: Step 2 updated for the finished SUMO bundle (kinematic motion, inert `lane` column, bundle sha256, realized speed range, tier–speed report, trace-end guard); Step 10 imports the SUMO ADR under a non-colliding number.

You are working on the PLATO side of a master's thesis on GPU-aware client selection for asynchronous federated learning (FedAsync) in vehicular networks. This prompt rebuilds the **environment model** — training time, upload time, energy, coverage, clock — and the **estimators** the selectors use. It does **not** implement Wen et al.'s selector (that is a later prompt) and it does **not** run experiments beyond a short smoke test.

Rules that still hold from earlier stages:
- Do not edit `plato/` core. Work in the example folder (subclasses, callbacks, scripts).
- Keep the ClientProfile three-way separation: **ground truth** (simulator only) / **reported** (server-visible) / **estimated** (server-computed). Selectors may read only reported + estimated.
- Every number that is a design decision comes from the locked table below. If the code forces a different value, stop and report.

---

## Locked decisions (14–15 Oct 2026)

| Item | Value | Source |
|---|---|---|
| Fleet | N = 30; tiers 15 Xavier / 10 Orin / 5 Thor | Wen Table 2 (K = 30) |
| Concurrency | C = 3 | Pisces / Polaris (C/N = 10%) |
| Data | CIFAR-10, ResNet-18 (PLATO `resnet_18`, BatchNorm), Dirichlet α = 0.8, 50,000 images split so N × partition = 50,000 (20 clients × 1,667 + 10 × 1,666) | carry-over; avoids reusing images |
| G_i prior α | 1 | carry-over |
| Local epochs E | 1 | decided 15 Oct |
| GPU efficiency η | 0.2 for every tier | decided 15 Oct (placeholder until Jetson profiling) |
| Tier specs | read from the existing tier CSV: FP16 TFLOPS 10 / 78 / 150, CPU 2.2 / 2.2 / 2.6 GHz | code CSV |
| Per-client jitter | persistent lognormal on true training time, σ = 0.149166 (CV 15%), drawn once at fleet creation | locked at Stage 1 |
| **Anchor** | **removed** — no multiplicative anchor anywhere | decided 15 Oct |
| Upload model | Wen Eq. 5–6: Tr = B·log₂(1 + h·p/(ε + N)), h = d^(−φ)·\|χ\|², T_up = D / Tr | Wen et al. 2024 |
| Upload constants | B = 2 MHz per vehicle, p = 23 dBm, N = −114 dBm, φ = 3, ε = 0 (orthogonal sub-channels), \|χ\|² ~ Exp(1) per upload | Wen Table 2; ε and fading as decided |
| Model size D | ResNet-18 parameter count × 32 bits, **read from the model** | — |
| Geometry | RSU at x = 1,000 m, lateral offset Loc_y = 10 m; d = √((x − 1000)² + 10²) | Wen Fig. 2 |
| Download | ignored | Wen §3.3 |
| CPU-cycle model | T̂_cpu = b·\|S\|·ρ / f with b = 24,576 bits (32×32×3×8), ρ = 1,250 cycles/bit for all vehicles | Wen Eq. 4; ρ = midpoint of [900, 1600] in Xiao et al. T-ITS 2022, cited by Wen |
| Energy (ground truth) | E = P_tier × T_train + p × T_up; P = 40 / 75 / 130 W (Xavier / Orin / Thor top power mode), p = 0.2 W | NVIDIA datasheets; Wen Eq. 8 |
| E_max | 600 J (non-binding by default) | decided 14 Oct |
| Scenarios | **S1**: 100% coverage, `always_accept`. **S2**: coverage x ∈ [400, 1600], `coverage_gated`, Stage 0 available | decided 14 Oct |

**Estimate that now includes upload (decided 14 Oct):** every non-random arm scores with T̂ = T̂_train + T̂_up, and τ̂ is derived from that total.

---

## Step 0 — Read-only audit (report, then continue)

Before changing anything, report in `evidence/envmodel_step0.md`:

1. Where sleep / training times are generated today, and every place the anchor enters.
2. Whether PLATO consumes per-client sleep from a static pre-generated table (it did in May) — this prompt needs a **per-dispatch** duration, so name the hook you will override.
3. Where `T_round` and τ̂ are computed, and what `s(·)` receives.
4. Where the clock-reconstruction script lives and what it reads.
5. Where Stage 0 lives, and what speed term it uses today (it should be `v_max = 19.44`).

## Step 1 — Physical training time (replaces the anchor)

- Measure forward FLOPs of the actual model at 32×32, batch 1, with `fvcore` (or `thop` — **state which, and whether it reports MACs or FLOPs**; convert to FLOPs = 2 × MACs).
- `FLOPs_train_per_image = 3 × FLOPs_fwd`.
- `T_train_tier = FLOPs_train_per_image × |S_i| × E / (η × TFLOPS_tier × 1e12)`.
- True training time per client = `T_train_tier × jitter_i`. Estimators never see jitter.
- **Expected** with FLOPs_fwd ≈ 1.1 GFLOP and |S| ≈ 1,667: Xavier ≈ 2.75 s, Orin ≈ 0.35 s, Thor ≈ 0.18 s. **If your measured FLOPs_fwd differs from 1.1 GFLOP by more than 20%, stop and report the number before continuing** — every training time depends on it.
- Delete the anchor from the config path and add a test that fails if any anchor parameter is read.

## Step 2 — Mobility from the trace

The SUMO side is finished (see its ADR in the bundle). Two facts about the trace that differ from what you might assume:
- **Motion is kinematic, not car-following.** Each vehicle holds its constant speed `V_k` exactly (TraCI speed mode 0); vehicles do not interact and can pass through each other. Position is `x(t) = (x0 + V_k·t) mod 2000` up to a 0.1 m wrap clamp. This matches Wen's model.
- **The `lane` column is inert** under kinematic motion. Do not read it anywhere.

Steps:
- Load `bundle_wen_closed_v1.zip`. Its sha256 must be `18532ec2348ae53279b0cbad4d916712c23772c0a62ed7b70bcc7cd8cb5d8a24`; then verify every per-file sha256 in `manifest.yaml`. Refuse to run on any mismatch.
- Desired speeds span 17.08–32.59 m/s (not the full 17–34). Nothing here depends on the bounds, but report the per-vehicle list.
- Map client id ↔ `veh_id` deterministically (client i ↔ `veh{i:02d}`) and **assert** the tier assignment is the fleet file's. Report a table of mean `V_k` per tier. If tiers are contiguous ID blocks (veh00–14 Xavier, veh15–24 Orin, veh25–29 Thor), expect ≈ 26.8 / 24.9 / 28.1 m/s. Do **not** reshuffle speeds or tiers to balance this — report it.
- Position lookup `x(veh, t)`: linear interpolation between 1 s samples, never across the wrap. Add a test that the lookup matches the analytic `(x0 + V_k·t) mod 2000` within 0.2 m over the whole trace.
- The trace covers t ∈ [0, 30,000] s. If any dispatch would need a position beyond the trace end, **stop** — do not loop the trace (the SUMO ADR forbids it; request a longer trace instead).
- Reported attributes per vehicle at decision time: `x`, `V_k` (from `fleet_speeds.csv`), tier spec. Nothing else from the trace is server-visible.

## Step 3 — Clean simulated clock, computed online

The upload time depends on **where** the vehicle is, so the clock used to look up positions must be the clean one, during the run — not PLATO's clock and not a post-hoc reconstruction.

- Maintain an online C-slot clock in the server: when a report frees a slot, the next dispatch happens at `t_dispatch = t_finish_clean` of the report that freed it.
- For each dispatch compute, from ground truth: `T_train_true`; `x_finish = x(veh, t_dispatch + T_train_true)`; `d`; draw `|χ|²` from an RNG keyed by `(seed, client_id, round)`; `T_up_true`; `t_finish_clean = t_dispatch + T_train_true + T_up_true`.
- Feed `T_train_true + T_up_true` to PLATO as that dispatch's sleep (per-dispatch override in the example folder).
- The post-hoc reconstruction script must now **read the logged per-dispatch durations** instead of a static sleep table, and must reproduce the online clock exactly. Add a test asserting `max |t_reconstructed − t_online| < 1e-6 s` on the smoke run.
- Near-tie merge order is a known residual; log it as before, do not try to fix it here.

## Step 4 — Coverage policy and waste

- **S1 (`always_accept`)**: every upload is accepted.
- **S2 (`coverage_gated`)**: a vehicle may be dispatched only if `x_dispatch ∈ [400, 1600]`. The upload is accepted only if the vehicle stays inside [400, 1600] for the whole interval `[t_dispatch + T_train_true, t_finish_clean]`. Upload rate uses d at training finish (Wen's model); the coverage check uses the whole interval. Partial uploads are not accepted.
- A rejected upload is **waste**: log its GPU-seconds, upload-seconds and energy, and free the slot at `t_finish_clean`.

## Step 5 — Estimators (one module, shared by all arms)

All estimators read only reported attributes at `t_dispatch`.

- `T̂_up(x, V, T̂_train)`: predicted finish position `x̂ = x + V·T̂_train` (apply the closed-loop wrap), `d̂`, rate with `|χ|² = 1`.
- `gpu`: `T̂_train` = tier formula from Step 1 without jitter; `T̂ = T̂_train + T̂_up`.
- `cpu`: `T̂_train = b·|S_i|·ρ / f_tier` (expect ≈ 23.3 / 23.3 / 19.7 s); `T̂ = T̂_train + T̂_up`.
- `gpu_oracle`: true `T_train` (with jitter) + `T̂_up` (still `|χ|² = 1`).
- `location_only`: constant `T̂_train` = fleet mean of the `gpu` estimates; `T̂ = T̂_train + T̂_up`. Low priority — implement the switch, do not run it yet.
- `observed`: unchanged history-based estimator, but its history is now **total** time (train + upload). Document exactly what it stores.
- `T_round` and τ̂: keep the existing definition but apply it to the total `T̂`, identically for every arm. Report the new steady-state `T_round` and the per-tier `s(τ̂)` for `gpu` (previously 0.82 / 0.72 / 0.35 for Thor / Orin / Xavier).

## Step 6 — Stage 0 (S2 only)

- Feasible iff `R_i = (1600 − x_i) / V_k ≥ T̂_train + T̂_up` **and** `Ê_i ≤ E_max`, where each arm uses its own `T̂` and `Ê_i = P_tier × T̂_train + p × T̂_up`.
- Replace the old `v_max` with the vehicle's reported `V_k` (decided 14 Oct).
- Must be pluggable into any selector (Wen's selector will reuse it later).

## Step 7 — Logging

One row per dispatch: `run_id, scenario, arm, seed, round, client_id, veh_id, tier, t_dispatch, x_dispatch, V_k, T̂_train, T̂_up, T̂_total, τ̂, s(τ̂), G_i, g_is_prior, score, stage0_pass, T_train_true, x_finish, d_finish, chi2, T_up_true, t_finish_clean, accepted, E_train, E_up`.

## Step 8 — Tests (pytest)

1. Upload model with D = 11.17 M × 32 bits and `|χ|² = 1` reproduces: d = 10 m → 5.0 s, 100 m → 7.0 s, 250 m → 8.3 s, 500 m → 9.6 s, 1,000 m → 11.4 s (±0.1 s). Then report the same table with the real D.
2. Training times match Step 1 to 1e-9 relative; no anchor read anywhere.
3. Online clock == reconstructed clock (Step 3).
4. S1 accepts every upload; S2 gating on hand-built cases (exit during training, exit during upload, stays inside).
5. Stage 0 with `gpu_oracle` T̂ and `|χ|² = 1` never admits a vehicle that then fails coverage (fading may still cause rare failures — count them separately).
6. Same seed → identical dispatch log for the deterministic parts.
7. Selectors cannot import ground-truth fields (type-level check).

## Step 9 — Smoke run (plumbing only, 1 seed, 200 rounds)

Cells: S1 × {random, cpu, gpu}; S2 × {gpu, gpu + Stage 0}. Report, **without interpreting accuracy**:

- upload share of total task time (fleet mean); per-tier mean true total time
- per-tier mean `T̂` for gpu and cpu vs true
- S2: waste fraction with and without Stage 0; median feasible-and-idle pool vs 2 × C = 6
- energy per task by tier
- online-vs-reconstructed clock check result

## Step 10 — ADRs

The SUMO side wrote its decision record as `ADR-0001-wen-closed-environment.md`, which collides with the existing ADR-0001…0006 series here. Import it into this repo under the next free number (keep its content, add a line noting the original filename), then continue numbering for this prompt's ADRs.

Next free numbers: anchor removal and the physical training-time formula; upload model and constants (Wen Eq. 5–6) with ε and fading choices; T̂ includes upload; ρ = 1,250 and its citation chain; energy model and E_max; Stage 0 using reported V_k; N = 30 / C = 3 / partition split.

---

## Stop conditions

- FLOPs_fwd off by > 20% from 1.1 GFLOP → stop after Step 1 and report.
- Bundle sha256 mismatch → stop.
- Any dispatch needing a position beyond the 30,000 s trace → stop.
- Online and reconstructed clocks disagree → stop.
- S2 median feasible-and-idle pool < 6 in the smoke run → stop and report; do not widen coverage.
- Any locked value that the code cannot honour → stop and report.

## Deliverable message

Reply with: the Step 0 audit summary, measured FLOPs and resulting per-tier training times, the upload table with the real D, the smoke-run numbers, test results, the ADR list, and every deviation from this prompt.
