# ADR-0018: `wen_fresh` — the Wen-faithful selector with a fresh F3

Status: implemented on 18 Oct 2026 (Prompt H, Step 2) in `wen_env/wen_fresh.py`. Runs are gated on the predictions v3.3 addendum commit.

## Context

In the `wen` arm (ADR-0017, W4), F3 is the last training loss a vehicle reported. A vehicle that has never trained gets the mean of every loss reported so far. Prompt F (`wen_env/evidence/wen_concentration_v1.md`) and Prompt H Step 0 (`wen_env/evidence/wen_fresh_v1.md` § 0) show how this ages:

- **The prior falls with training.** It is the running mean of all reports in history, which in S1 seed 4127 drops from 1.86 at round 8 to 0.13 at round 4,000. A never-trained Thor vehicle therefore averages F3 ≈ 0.34 over the run.
- **A vehicle picked once early keeps its early loss.** For example, client 30 reported 1.0746 at round 477 and kept that F3 for the remaining ≈ 3,500 rounds.
- **Frequently picked vehicles keep reporting low losses.**

So in `wen`, F3 measures *when* a vehicle last trained, not how useful it is to the current model. `wen_fresh` removes that staleness while keeping everything else the same.

## Decision

1. **New arm `wen_fresh`, on the same selector code path.** `WenFreshSelection` subclasses `WenFaithfulSelection` and overrides only `f3`. The following are inherited unchanged:
   - F1, F2, F4;
   - T_max = 34.334345905647005 s, E_max = 36.423846640394224 J;
   - fallback, exact Pareto set, min–max normalisation, pick rule, tie-break;
   - determinism.
2. **F3_k = mean cross-entropy of the current global model on vehicle k's full disjoint partition.**
   - Eval mode, no augmentation.
   - The CIFAR-10 test transform: ToTensor plus the training Normalize of `plato/datasources/torchvision.py`.
   - Computed in float64 sums over float32 per-sample losses, with deterministic cuDNN for the evaluation.
3. **Recomputed for all 30 vehicles once per global-model version.**
   - `WenEnvMixin.global_version` counts aggregations that changed the model. Version 0 is the initial model, and a cycle whose updates were all lost does not change it.
   - The evaluation runs at the first decision after a version change, and the result is cached until the next change. Every decision therefore uses the model current at that decision.
4. **No prior, and no reported training loss.**
5. **The evaluation is not charged** to the online clock or to energy. It runs on the host between PLATO events and never touches the simulated clock. Its host wall time is logged per version as `fresh_eval_s` in `wen_fresh_eval.csv`, together with all 30 F3 values.
6. **Logs.** The per-decision log and the per-candidate side file are the same as for `wen`, plus `global_version`.
7. **Test mode `wen_fresh_stale_f3_test = true`** (Step 3, test 1). The fresh evaluation still runs and is logged, but decisions use the `wen` stale F3. This checks the new code path against an existing `wen` run.
8. **Lock.** No `wen_fresh` run, of any length, is allowed without the committed `docs/predictions_v3_3_addendum.md` (`wen_manifest.py`).

## Difference from Wen's Algorithm 3

Wen's F3 is the training loss each candidate obtains *after* local training. `wen_fresh` uses the loss of the *global model before* training, evaluated on the candidate's data.

This is a different quantity: how badly the current global model fits that vehicle's data, rather than what the vehicle reported after its own update. It is chosen because it needs no training of non-picked candidates and is current at every decision.

## Cost

The evaluation is charged neither to the clock nor to energy, so this is the best case for Wen.

- One evaluation of 50,000 images with ResNet-18 takes 4.4 s on an otherwise idle RTX 4000 Ada (`wen_fresh_v1.md` § 7).
- Up to 4,000 versions per run gives about 4.9 h of extra host time per run before GPU sharing.
- The Prompt H stop rule applies: over 8 h of projected extra wall time per run means report and wait.
