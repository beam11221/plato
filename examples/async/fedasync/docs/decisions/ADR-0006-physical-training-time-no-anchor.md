# ADR-0006: Physical training time; the anchor is removed

Status: accepted 15 Oct 2026, implemented 16 Oct 2026 (Prompt B, Step 1).
This supersedes ADR-0004 for the Wen-aligned environment. Folders that already have finished runs keep their anchored setup unchanged.

## Context

ADR-0004 kept a multiplicative anchor that pinned Xavier at 15 s. Under it, `eta_nominal = 1.0` was a placeholder, and the anchor absorbed the real efficiency.

The Wen environment adds a physical upload time of about 5–11 s, so the ratio of training time to upload time now decides which regime the experiment is in. A free anchor would make that regime a tuning knob.

## Decision

```
T_train_tier = 3 × FLOPs_fwd × |S_i| × E / (η × TFLOPS_tier × 1e12)
T_train_true = T_train_tier × exp(eps_i),   eps_i ~ N(0, 0.149166) drawn once per client
```

**FLOPs_fwd is measured**, not assumed (`wen_env/measure_flops.py`).
- Tool: fvcore `FlopCountAnalysis` on PLATO's `resnet_18`, at 32×32, batch 1. fvcore counts multiply-adds.
- Result: **556,651,520 MACs, i.e. 1,113,303,040 FLOPs**. That is +1.2% against 1.1 GFLOP, inside the 20% stop band.
- The ×3 stands for forward + backward + optimizer.

**Other settings.**
- η = 0.2 for every tier. This is a placeholder until the Jetsons are profiled. It lives in `tiers/tiers_wen.csv` as `eta_nominal`.
- E = 1.
- Datasheet throughput is 10 / 78 / 150 TFLOPS FP16.

**Result at |S| = 1,667:** Xavier 2.784 s, Orin 0.357 s, Thor 0.186 s.

## Where the anchor is gone, and how that is enforced

- The fleet is `gpu_tiers_dataset/fleets/n30_x15-o10-t5_phys_p1667_j0p149_s4126`. It was built with `build_fleet.py --no-anchor --w-model-flops 3339909120`, so its manifest records `anchor_tier = null`. `wen_fleet.Fleet.load` refuses any fleet that is anchored.
- The Wen path does not call `gpu_tier_sleep`'s tier predictor at all. `WenPolarisServer` builds its own strategies.
- Configs may hold only `[gpu_tier] t_round_model`, and the entry point refuses any other key.
- `tests/test_wen_no_anchor.py` runs the selection path against a config section that raises if `anchor_*`, `tiers_csv`, `clients_csv` or `w_model_flops` is read.
- The client's static sleep table is still named in the config, because PLATO's client requires one. Every value in it is overwritten per dispatch.

## Consequences

- Seconds in the Wen runs are physical, given η. Changing η changes every regime statement, so it must be revisited with Jetson measurements.
- Results are not poolable with any anchored run.
