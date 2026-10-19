# Prompt C, Step 4: pilot gate (seed 4126, 4,000 rounds)

| run | rounds | plateau slope (pts / 1,000 rounds) | plateau | 80% crossing round | target | last dispatch (s) | trace | clock diff (s) | clock |
|---|---|---|---|---|---|---|---|---|---|
| S1_random_s4126 | 1770 | +3.943 (mean acc 0.8775) | **FAIL** | 380 | PASS | 6643.9 | PASS | 0.0e+00 | PASS |
| S1_gpu_s4126_hung_diskfull_0740 | 1950 | +0.803 (mean acc 0.8775) | **FAIL** | 390 | PASS | 6344.1 | PASS | 0.0e+00 | PASS |

**Gate: FAIL — stop and report**

## Context (added 16 Oct 2026)

- **The gate does not stop the experiment.** By the user's instruction (16 Oct), Phase A continues and this result is reported, not acted on.
- **Plateau fails because neither run reached a plateau.** Both ended near 90% accuracy, long before 4,000 rounds:
  - S1 random ended through PLATO's `target_accuracy = 0.9` early stop at round 1,770 (deviation A in `promptC_progress.md`).
  - S1 gpu is the run that hung when the disk filled, at round 1,950 (deviations B and C).
- **The slopes show both curves were still rising** over their last 500 rounds: +3.9 points per 1,000 rounds for random and +0.8 for gpu, against a limit of 0.5. A plateau test over rounds 3,500–4,000, as the prompt defines it, cannot be done on these runs.
- **The other three gates pass.** Both runs cross 80% long before round 3,500 (rounds 380 and 390). The last dispatch is far inside the 30,000 s trace. The online and replayed clocks agree exactly.
- **Curves:** `pilot_curves.png`.
