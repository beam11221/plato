# ADR-0019: Study 2 — Stage 0 and the GPU estimator as plug-ins

Status: implemented on 19 Oct 2026 (Prompt I, decisions S2-1 to S2-4). Runs are gated on the predictions v3.4 addendum commit (`0174b23d`).

## Context

Study 2 asks whether our two components help other selectors when they are inserted into those selectors' own pipelines:
- **Stage 0:** the feasibility filter with our estimator (ADR-0011).
- **The GPU-aware estimator:** ADR-0006 / ADR-0008.

The hosts are `random` and `wen_fresh` (ADR-0018). Each host runs as proposed, and its own logic is not modified.

## Decision

1. **Pipeline, for every cell and every scenario:**
   1. **Eligibility:** the environment rule. The vehicle is idle (not training, no report awaiting merge), and in S2 it is inside [400, 1600] at dispatch time. This is `WenCoverageGate.eligible` over PLATO's idle pool.
   2. **Stage 0** (if on).
   3. **The host's own scoring and constraints.**
   4. **The host's selection.**

   This is the layering `WenCoverageGate → WenStage0Gate → host` in `wen_server.install_wen`.
   - Stage 0 checks every vehicle it receives against `server.wen_is_eligible` and raises otherwise. It never applies eligibility itself.
   - Logged pools confirm that no existing run gave Stage 0 a busy vehicle (Prompt I Step 0a).

2. **Stage 0 uses our estimator whatever the host.**
   - The `_s0` arms hand `WenStage0Gate` a `GpuEstimator`:
     - T̂ = T̂_train(GPU) + T̂_up, with |χ|² = 1, at x̂ = wrap(x + V_k·T̂_train);
     - residence (1,600 − x)/V_k ≥ T̂;
     - Ê = P_tier·T̂_train + p·T̂_up ≤ E_max = 600 J.
   - The host's own estimator (Wen's CPU F1 / F2, or none for random) is never used by Stage 0.
   - Existing arms keep their own estimator in Stage 0. `cpu + Stage 0` still uses the CPU family (ADR-0014).

3. **Fallback.** This is the existing `gpu + Stage 0` rule (`WenStage0Gate.select_clients`): when Stage 0 rejects every eligible vehicle, the unfiltered eligible pool is passed to the host and the event is logged.

4. **Stage 0 in S1 has no exit boundary.**
   - Coverage is the whole loop, so R_i = ∞ and only the energy test remains, which is non-binding at 600 J for every tier.
   - Before this ADR, Stage 0 was refused outside S2. "Ours" (gpu + Stage 0) is now defined in S1 as well.
   - Prompt I Step 0c checks the no-op on a run; it is not assumed.

5. **The GPU plug-in for Wen replaces only F1 / F2, with the GPU family (S2-2a):**
   - F1 = T̂_train(GPU) + T̂_up, with x̂ = x + V_k·T̂_train(GPU);
   - F2 = P_tier·T̂_train + p·T̂_up.

   F3 (fresh), F4, the Pareto set, min–max normalisation, the pick rule and the tie-break are unchanged.

6. **GPU limits use the W5 rule (S2-2b).**
   - T_max and E_max are the 90th percentiles of the new F1 and F2, taken over all 30 vehicles × trace samples at 1 s for t ≤ 30,000 s, with estimates only and the same wrap (`wen_limits.py --family gpu`).
   - Result: **T_max = 13.564085609369194 s**, **E_max = 113.50862433236982 J**.
   - These limits cut only far Xavier positions (20% of Xavier, both limits together). They cut nothing inside [400, 1600].
   - The CPU host keeps 34.334 s and 36.424 J.

7. **Arms.**

   | arm | host | Stage 0 | Wen F1 / F2 |
   |---|---|---|---|
   | `random_s0` | random (ε-greedy, ε = 1) | on, gpu estimator | — |
   | `wen_fresh_s0` | `wen_fresh` | on, gpu estimator | CPU |
   | `wen_fresh_gpu` | `wen_fresh` | off | GPU |
   | `wen_fresh_gpu_s0` | `wen_fresh` | on, gpu estimator | GPU |

   `study2_plugin_off_test = true` turns the arm's plug-in off inside the arm's own code path (Step 3 test 1).

8. **Logs.**
   - `stage0_log.csv` has one row per Stage 0 call: eligible count, passed count, rejections by reason (time / energy / both) and by tier, and fallback.
   - Wen decision rows of the `_s0` arms carry the same counts (`n_eligible`, `n_stage0_rej_*`, `stage0_fallback`).
   - Every manifest records the source-tree sha256, the source commit and a dirty flag.
