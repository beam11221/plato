# Stage 6 — progress log

Running record against `prompts/STAGE6_seeds.md`. Appended to as each step
completes; not rewritten in place.

---

## Pre-flight — seed infrastructure. Found a gap, fixed it, verified it.

Before generating a single config, checked what "run seed" (the prompt's
"lottery draws, torch, dispatch order") actually controls in this pipeline.
**`sumo_fedasync.py` did not seed torch at all** — every Task 4 / Stage 5 /
Stage 5b run set only `[data] random_seed` (the data sampler). Model init and
CIFAR-10 augmentation (`RandomHorizontalFlip` + `RandomCrop`) were free-
running. `gpu_tiers_dataset/seeding.py` already measured what that alone is
worth, for the plain (non-SUMO) FedAsync arms: two runs of one otherwise-
identical config differed by **6.29% of simulated clock** from torch alone.
For Stage 5/5b that sat safely inside a much larger filter effect; for Stage
6, whose entire point is to measure the SIZE of run-to-run noise, it would
have been exactly the thing being measured, uncontrolled — silently
inflating every "seed spread" number in this stage.

**Fix**: `sumo_close_loop/sumo_fedasync_seeded.py` (new file, `sumo_fedasync.
py` untouched — same reasoning `gpu_tiers_dataset/fedasync_seeded.py` gives
for being a separate file: Task 4/5/5b's results must stay reproducible by
re-running the entry point they actually used). Adds, at module level (before
torch is touched, so it also reaches every spawned client process):

- `seeding.seed_everything()` — seeds `random`, `numpy`, torch CPU+CUDA,
  pins cuDNN, all from `[clients] random_seed`.
- `client_seeded_trainer.ClientSeededTrainer` as the client's trainer —
  re-seeds from `(base_seed, client_id, round)` at the top of every local
  training run, so a client's randomness follows the CLIENT, not whichever
  worker process happens to be free that round (a second, independently
  known bug — `stage1_evidence/determinism.txt` §4 — fixed by the same
  module `gpu_tiers_dataset` already uses).

Both are reused, not reimplemented, from `gpu_tiers_dataset/` — added to
`sys.path` alongside the existing `gpu_tiers/` entry.

### Determinism test (before trusting this for real)

Two 15-round runs of the same config, same seed (4126), different ports,
run concurrently on the same GPU:

- **`train_loss` and `accuracy` were byte-identical at every matching round**
  (e.g. round 10: `accuracy=0.1862`, `valid_loss=2.2242720554351805` in
  both) — proof the actual training computation (model init, augmentation,
  gradients) is now fully reproducible given the same client and round.
- **Dispatch order matched exactly through round 13, then diverged slightly**
  (round 14: one run selected client 18, the other selected 19 then 14).
  Traced to `elapsed_time` differing by fractions of a second between the two
  processes (host `processing_time`, a `perf_counter` delta PLATO adds to
  every report — never seeded, documented as such in `client_seeded_trainer.
  py`'s own docstring and `stage1_evidence/determinism.txt`), which can flip
  which of two near-simultaneous client reports the async event loop
  processes first. This is a **real, pre-existing, already-documented
  limitation of the simulator's host-timing model, not a seeding bug** — it
  does not threaten Stage 6's actual question (whether *different* seeds
  4126/4127/4128 produce stable orderings and a small spread), only clarifies
  that "same seed twice" would not be perfectly bit-identical in dispatch
  order, only in every computation. Not investigated further — Stage 6 asks
  about deliberately different seeds, not repeats of the same one.

### Fleet checksum (the prompt's explicit ask)

`sleep_times.csv` for fleet `n20_x10-o7-t3_xa15_p2500_j0p149_s4126`:

```
80e6a50fb75052ce89dc33a7149057a4  gpu_tiers_dataset/fleets/n20_.../sleep_times.csv
80e6a50fb75052ce89dc33a7149057a4  sumo_close_loop/output/client_sleep_times_n20r400.csv
```

Identical, as expected — every Stage 6 config's `[clients.simulation_
distribution] csv_path` points at the exact same file
(`sumo_close_loop/output/client_sleep_times_n20r400.csv`), read as plain
text and never regenerated. The run seed cannot touch it regardless of which
RNG it feeds; verified rather than assumed.

### The asymmetry this leaves, stated plainly

Two rows this stage reuses — **not** re-run, per the prompt — were
produced by the OLD, unseeded `sumo_fedasync.py`: `baseline/none` (Task 4,
seed 4126) and `baseline + tier_spec + vmax_distance` (Stage 5b, seed 4126).
Every one of Stage 6's 12 new runs goes through the new, fully-seeded entry
point. This means:

- Batch A's table (below) compares 4 newly-seeded rows against 2 not-newly-
  seeded reference rows. Stage 5/5b's own effect sizes (20-40% on total
  time) dwarf the ~6% torch-noise figure that motivated the fix, so this is
  very unlikely to change any qualitative reading of Batch A — but it is a
  real difference in how those two rows were produced, and is not "the same
  as" the four new ones.
- Batch B's per-cell "3 seeds" (4126 reused + 4127/4128 new) inherits the
  same asymmetry for `baseline/none` and `baseline + tier_spec + vmax_
  distance` specifically: the seed-4126 point in those two cells sits on
  old infrastructure, 4127/4128 on new. The clean, apples-to-apples
  comparison in those two cells is 4127-vs-4128; 4126's role there is a
  useful third point, not a perfectly matched one. `gpu + tier_spec + vmax_
  distance` and `baseline + fleet_mean + vmax_distance` have no such issue
  — all three of their seeds (Batch A's 4126 run plus Batch B's 4127/4128)
  are newly-seeded.

This is inherent to the prompt's own "do not re-run existing rows" rule, not
a shortcut taken here — flagged now so Batch B's write-up doesn't need to
re-derive it.

---

## Batch A — launched.

4 runs, seed 4126, `stage0_r_i = vmax_distance` throughout, all through
`sumo_fedasync_seeded.py`. Smoke-tested all four live (45s each) before this
launch: all boot cleanly, seeding fires (`[Seed] ... seed=4126 ... torch=True
cudnn_deterministic=True`), `[Stage0]` lines show the expected per-arm belief
(`cpu`'s xavier/orin both predicting 15.000000s, the same tier-collapse Stage
5 found; `fleet_mean` showing the constant 8.323077 regardless of arm).

Launched 2026-09-20 13:49:59 UTC:

| run | port |
|---|---|
| gpu_tier_spec_s4126 | 18620 |
| cpu_tier_spec_s4126 | 18621 |
| baseline_fleet_mean_s4126 | 18622 |
| observed_fleet_mean_s4126 | 18623 |

**Gate: Batch B will not be generated or launched until Batch A is analysed
and reported, per the prompt.**

---

## Batch A — DONE, gate cleared.

All 4 finished cleanly 2026-09-20 18:48–18:56 UTC (~5h each), round 4000, no
errors. (Report delayed to 2026-09-21 — the runs sat finished for about a day
before this write-up; noted so the timeline is honest.)

### Stop conditions — all clear

| run | fallback | conservation error | min median pool | plateau |
|---|---|---|---|---|
| gpu_tier_spec | 0.12% | +0.52% | 7.0 | 81.66% |
| cpu_tier_spec | 1.10% | +0.68% | 5.0 | 82.23% |
| baseline_fleet_mean | 0.33% | +0.65% | 7.0 | 83.01% |
| observed_fleet_mean | 0.15% | +0.74% | 7.0 | 82.52% |

All well inside every threshold (fallback <10%, conservation <3%, pool
≥ C=4, plateau within 2 pts of Stage 5's 82.1–83.7% band).

### The six-row table

| row | T @ 4000 | vs `baseline/none` | waste | xavier share |
|---|---|---|---|---|
| `baseline/none` (Task 4, not re-run) | 7525.2 s | — | 13.17% | 39.0% |
| `baseline+tier_spec` (Stage 5b, not re-run) | 4529.6 s | -39.81% | 0.20% | 23.5% |
| **A1** `gpu+tier_spec` | 5007.4 s | -33.46% | 0.32% | 27.0% |
| **A2** `cpu+tier_spec` | 6606.5 s | -12.21% | 0.92% | 39.5% |
| **A3** `baseline+fleet_mean` | 6116.2 s | -18.72% | 3.91% | 34.0% |
| **A4** `observed+fleet_mean` | 6578.5 s | -12.58% | 4.17% | 37.4% |

Decomposition (waste channel / mix channel) against `baseline/none`:

| run | waste channel | mix channel |
|---|---|---|
| A1 `gpu+tier_spec` | -12.9% | -23.2% |
| A2 `cpu+tier_spec` | -12.4% | **+0.9%** |
| A3 `baseline+fleet_mean` | -9.6% | -9.5% |
| A4 `observed+fleet_mean` | -9.4% | -2.8% |

### The two gate checks

**A1 — does `gpu` still show slower than `baseline` behind the filter?**
Yes: 5007.4 s vs 4529.6 s, **`gpu` is 10.55% slower**. Stage 5 measured
11.9% slower at `vmax_elapsed`. Same finding, similar margin, at the
method's own R_i. **Gate condition NOT triggered — Batch B's `gpu` cell
proceeds as planned.**

**A3 — has the `T̂` ladder's gap collapsed from ~21 points?** No:
`baseline+fleet_mean` (-18.72%) trails `baseline+tier_spec` (-39.81%) by
**21.09 points** — essentially identical to the ~21-point gap measured at
`vmax_elapsed`. **Gate condition NOT triggered — the hardware-spec signal is
just as valuable at the corrected R_i. Batch B's `fleet_mean` cell proceeds
as planned.**

**Neither gate condition fires. Batch B needs no redesign — proceeding with
it exactly as specified.**

### Two things noticed in passing, not gate-relevant but worth recording

- **A2 (`cpu+tier_spec`)'s mix channel is ~0 (+0.9%)** — at `vmax_elapsed`
  the wrong spec still bought a real mix effect (over-conservative R_i
  papered over cpu's inability to tell xavier from orin apart); at the
  accurate `vmax_distance`, that's gone: A2's xavier share (39.5%) is
  statistically indistinguishable from `baseline/none`'s own 39.0%. The
  wrong spec's ENTIRE remaining benefit at this R_i is the waste channel
  (-12.4%).
- **Both `fleet_mean` runs (A3, A4) show higher waste (3.91%, 4.17%) than
  their Stage 5 `vmax_elapsed` counterparts (2.60%, 1.96%)** — the same
  direction as Stage 5b's `oracle_exit` finding: a tighter, more accurate
  R_i removes some of `vmax_elapsed`'s incidental safety margin against a
  client's own jitter, at the cost of a little more waste. Consistent
  cross-check between two different stages' data, not a new mechanism.

**Proceeding to Batch B.**

---

## Batch B — launched.

8 runs, seeds 4127/4128 × 4 cells, all through `sumo_fedasync_seeded.py`.
Smoke-tested `baseline_none_s4127` live first (the one new combination —
Stage 0 disabled — not yet exercised through the seeded entry point): boots
cleanly, `[Seed] ... seed=4127 ... torch=True`, no `[Stage0]` lines (correct,
filter off). Launched 2026-09-21 16:21 UTC, all 8 sharing one GPU:

| cell | seed 4127 port | seed 4128 port |
|---|---|---|
| `baseline_none` | 18620 | 18621 |
| `baseline_tier_spec_vmax_distance` | 18622 | 18623 |
| `gpu_tier_spec_vmax_distance` | 18624 | 18625 |
| `baseline_fleet_mean_vmax_distance` | 18626 | 18627 |

(Ports 18620-18623 numerically repeat Batch A's — a config-generator quirk,
harmless here since Batch A's processes had already exited before Batch B
started; flagged rather than silently left for whoever reads the port list
next to Task 4/5/5b's blocks.)

8 processes sharing the GPU (vs Batch A's 4) — expect somewhat longer than
Batch A's ~5h each. Will report Steps "across seeds" (mean/std, ordering
stability, noise floor, conservation per seed) once all 8 finish.

### Incident: 4 of 8 hung on CUDA OOM, found and fixed 2026-09-22

4 runs finished cleanly (`baseline_none_s4127`, `baseline_none_s4128`,
`baseline_tier_spec_vmax_distance_s4128`, `baseline_fleet_mean_vmax_
distance_s4128`, all 2026-09-21 21:19–21:50 UTC). The other 4
(`baseline_tier_spec_vmax_distance_s4127`, `gpu_tier_spec_vmax_distance_
s4127`, `gpu_tier_spec_vmax_distance_s4128`, `baseline_fleet_mean_vmax_
distance_s4127`) hit `torch.OutOfMemoryError` during round 1 training at
2026-09-21 16:23 UTC — 8 concurrent runs × up to 4 client processes each
briefly exceeded the 20 GB GPU, one client per hung run crashed mid-
`forward()`, and the server side of each just sat waiting for a report that
would never arrive. No crash, no exit code, no further log lines for
~17.5 hours until this was checked.

Fix: killed the 4 hung server processes and their cleaners, deleted their
(round-1-only, nothing salvageable) run directories and logs, relaunched
just those 4 together (same concurrency as Batch A, which never had this
problem) once the GPU was confirmed clear. Watched the restart for OOM/
traceback in the first minute — none. Will confirm the retried 4 finish
clean before treating any of Batch B's 8 rows as final.

Lesson for later stages: 8-way GPU sharing is not safe headroom for this
model/batch size on this GPU, even though 4-way (Task 4, Stage 5, Stage 6
Batch A) always has been. Launch Batch-B-sized (8-run) batches in two
waves of 4 from now on rather than all at once.

**All 8 Batch B runs finished clean 2026-09-22** (4 originally + 4 retried,
14:46–14:56 UTC for the retry group). No errors in any of the 8.

---

## Steps "across seeds" — analysis (`analyze_stage6.py`, extended for Stage 6).

### Stop conditions — all 8 clear

| run | fallback | conservation error | min median pool | plateau |
|---|---|---|---|---|
| baseline_none_s4127 | 0.00% | +0.47% | — (filter off) | 83.11% |
| baseline_none_s4128 | 0.00% | +0.64% | — (filter off) | 83.86% |
| baseline_tier_spec_vmax_distance_s4127 | 0.12% | +0.67% | 6.0 | 82.87% |
| baseline_tier_spec_vmax_distance_s4128 | 0.12% | +0.05% | 7.0 | 84.06% |
| gpu_tier_spec_vmax_distance_s4127 | 0.22% | +0.08% | 7.0 | 83.26% |
| gpu_tier_spec_vmax_distance_s4128 | 0.07% | -0.53% | 6.0 | 83.19% |
| baseline_fleet_mean_vmax_distance_s4127 | 0.30% | +0.63% | 7.0 | 83.45% |
| baseline_fleet_mean_vmax_distance_s4128 | 0.12% | +0.68% | 7.0 | 83.91% |

All 8 conservation errors are **under 0.7%** — tighter than even the "~1%"
the prompt expected, and far under the 3% stop threshold. Independent
evidence the three seeds differ only in the intended way (dispatch order /
lottery / torch), not in something structural.

### Per-cell, across 3 seeds (4126 reused + 4127/4128 new)

| cell | seed 4126 | seed 4127 | seed 4128 | mean T | sd | spread (sd/mean) |
|---|---|---|---|---|---|---|
| `baseline/none` | 7525.2 s | 7485.8 s | 7593.7 s | 7534.9 s | 54.6 s | **0.72%** |
| `baseline+tier_spec+vmax_distance` | 4529.6 s | 4586.4 s | 4590.7 s | 4568.9 s | 34.1 s | **0.75%** |
| `gpu+tier_spec+vmax_distance` | 5007.4 s | 5260.5 s | 4771.3 s | 5013.0 s | 244.6 s | **4.88%** |
| `baseline+fleet_mean+vmax_distance` | 6116.2 s | 6249.7 s | 6246.6 s | 6204.1 s | 76.2 s | **1.23%** |

### Prediction 1 — seed spread on T@4000

**Holds clearly for 3 of 4 cells** (0.72–1.23%, an order of magnitude under
the ~5% ceiling). **The 4th, `gpu+tier_spec`, sits right at the ~5%
boundary (4.88%)** — not a violation, but not comfortably clear of it
either. This is exactly the cell Batch B's own design flagged in advance as
"the smallest effect of the four and the most seed-sensitive" — the data
agrees: its spread is 4–7× every other cell's, using `baseline/none`'s
0.72% as the noise floor reference the prompt asks for.

**TTA→80% spread, the second half of Prediction 1** ("much larger" than the
T@4000 spread): confirmed sharply —

| cell | TTA→80% spread ((max−min)/mean) |
|---|---|
| `baseline/none` | 25.6% |
| `baseline+tier_spec+vmax_distance` | 0.8% |
| `gpu+tier_spec+vmax_distance` | 27.4% |
| `baseline+fleet_mean+vmax_distance` | 20.7% |

Three of four land inside or near Stage 3's own ±6–21% reference band; all
are 15–35× their own cell's T@4000 spread except `baseline+tier_spec`,
whose TTA spread is oddly tiny (0.8%) despite an unremarkable T@4000 spread
(0.75%) — noted, not explained away; plausibly this cell's accuracy curve
just happens to cross 80% at a point in training where the three seeds'
trajectories are still tightly bunched, but that is a guess, not a checked
mechanism.

**Prediction 1 verdict: holds, with one cell (`gpu+tier_spec`) at the edge
rather than comfortably inside it.**

### Ordering stability — checked in every individual seed, not just the mean

| comparison | claim | 4126 | 4127 | 4128 | verdict |
|---|---|---|---|---|---|
| `baseline+tier_spec` vs `baseline/none` | filter is faster | ✓ (-39.8%) | ✓ (-38.7%) | ✓ (-39.5%) | **holds in all 3** |
| `baseline+tier_spec` vs `baseline+fleet_mean` | spec beats spec-free | ✓ (4529.6<6116.2) | ✓ (4586.4<6249.7) | ✓ (4590.7<6246.6) | **holds in all 3** |
| `gpu+tier_spec` vs `baseline+tier_spec` | selector does NOT help | ✓ gpu slower by 10.55% | ✓ gpu slower by 14.69% | ✓ gpu slower by 3.93% | **holds in all 3** |

**All three orderings hold in every individual seed — Prediction 2 holds
fully, no exceptions**, including the third comparison, whose CELL has the
largest seed spread (4.88%) but whose SIGN never flipped: `gpu` was slower
in all three seeds even though by how much varied 3.7×. That distinction —
noisy magnitude vs. stable direction — is the actual finding here, and is
worth keeping separate from Prediction 1's spread number.

### Prediction 3 — A1 reproduces Stage 5's finding

Margin: Stage 5 measured `gpu` 11.9% slower at `vmax_elapsed`. At
`vmax_distance`: 10.55% / 14.69% / 3.93% across the three seeds (mean
9.7%). **"Roughly the same margin" holds on average but hides real
per-seed spread** — report both, not just the mean, per the stage's own
rule. Xavier share: `gpu` (27.0% / 28.9% / 24.8%) is higher than
`baseline+tier_spec` (23.5% / 23.9% / 23.7%) in **every** seed — the
"dispatches more xavier" half of the prediction holds cleanly.

### Prediction 4 — A3's ~21-point gap holds regardless of seed

| seed | `baseline+fleet_mean` gap to `baseline/none` | `baseline+tier_spec` gap to `baseline/none` | difference |
|---|---|---|---|
| 4126 | -18.72% | -39.81% | 21.09 pts |
| 4127 | -16.51% | -38.73% | 22.22 pts |
| 4128 | -17.74% | -39.55% | 21.81 pts |

**Extremely stable — 21.1 to 22.2 points across all three seeds.**
**Prediction 4 holds, tightly**, in contrast to Prediction 3's noisier
margin. The hardware-spec signal's value over a spec-free `T̂` is not just
present but seed-insensitive.

### Prediction 5 — waste ≤ ~1% in filtered runs

| cell | seed 4126 | seed 4127 | seed 4128 |
|---|---|---|---|
| `baseline+tier_spec+vmax_distance` | 0.20% | 0.25% | 0.27% |
| `gpu+tier_spec+vmax_distance` | 0.32% | 0.35% | 0.25% |
| `baseline+fleet_mean+vmax_distance` | 3.91% | 4.14% | 4.07% |

**Holds for the two `tier_spec` cells** (0.20–0.35%, comfortably ≤1% in
all 6 runs). **Does not hold for `fleet_mean`** (3.9–4.2%, consistent with
Stage 5's own `fleet_mean` runs at 2.0–2.6% and Batch A's at 3.9–4.2% — a
crude, spec-free `T̂` has never cleared ~1% waste at any `R_i` tested so
far). Reading the prediction literally ("every filtered run") it fails for
`fleet_mean`; reading it as being about the `tier_spec` cells specifically
(which is where the stage's central claims live), it holds. Both readings
reported rather than picking the convenient one.

### Prediction scorecard

| # | prediction | result |
|---|---|---|
| 1 | seed spread on T@4000 small (<~5%); TTA spread much larger | **holds for 3/4 cells; `gpu+tier_spec` at the 4.88% edge.** TTA spread confirmed much larger (0.8–27.4%) |
| 2 | all three orderings hold in every individual seed | **holds fully, no exceptions** |
| 3 | A1 reproduces Stage 5 (slower, similar margin, more xavier) | **direction and xavier-share holds; margin holds only on average (3.9–14.7% range vs Stage 5's single 11.9%)** |
| 4 | A3's ~21-point gap stays close to `vmax_elapsed`'s | **holds tightly (21.1–22.2 pts across all 3 seeds)** |
| 5 | waste ≤ ~1% in every filtered run | **holds for `tier_spec` cells; does not hold for `fleet_mean` cells (3.9–4.2%)** |

### Bottom line for Prediction 1's own stated stakes

The prompt says Prediction 1 "decides how every other number in the thesis
may be quoted." Three of four seeded cells show noise an order of magnitude
under the 5% ceiling — a mean ± sd is a defensible way to quote those.
`gpu+tier_spec+vmax_distance` sits at the boundary the prompt itself
treated as the dividing line, and this is also the one cell whose central
claim (Prediction 2's ordering) is a "no effect from the selector" result
rather than a large measured one — exactly where seed noise matters most.
**Recommendation, not a decision made unilaterally: quote `gpu+tier_spec`'s
mean with its sd shown explicitly (not just "±X%" in prose), and lean on
the ordering-stability result (holds in all 3 seeds) as the load-bearing
claim for that cell rather than the magnitude of the gap.**
