# Experiment: does Dirichlet alpha make the selectors separable?

## Why

Across all four cells we have run so far, the four selector arms are
indistinguishable on accuracy: measurement noise is 1.4-2.1 points while the gap
between arms is 0.2-0.6 points. One hypothesis is that the task is too easy for
selection to matter: at alpha=1.0 every client's data is roughly equally useful,
so G_i is nearly flat and the selector has no leverage.

This run tests that hypothesis by changing **alpha only**. If alpha=0.1 makes the
arms separate and alpha=1.0 does not, alpha is the lever and becomes the standard
condition for everything after this.

## What to run

Environment: **surgical** (no mobility, no SUMO).

| setting | value |
|---|---|
| N clients | 20 |
| concurrency C | 4 |
| rounds | 4000 |
| Dirichlet alpha | **1.0** and **0.1** |
| arms | baseline, gpu, cpu, observed |
| seeds | code must accept a seed list; **run seed 1 of 3 on this machine** |
| data-to-tier mapping | random (unchanged — adversarial mapping is a later experiment) |

**24 runs total planned (2 alpha x 4 arms x 3 seeds). This machine runs 8 of them
(seed 1 only).**

Everything else must stay exactly as the existing surgical config: fleet mix
50/35/15, model, optimizer, anchor_sleep, partition_size, eval interval, tier
spec file. Change alpha and rounds and nothing else.

**Re-run alpha=1.0 too — do not reuse the existing results.** The existing runs
are 10,000 rounds on a single seed; comparing those against new 4,000-round runs
would confound alpha with round budget. The fresh alpha=1.0 runs also serve as a
reproduction check.

## Parallelism

One machine, two jobs in parallel: **job A = all alpha=1.0 runs, job B = all
alpha=0.1 runs.** Split by alpha, not by arm, so all four arms of a given alpha
share identical machine conditions.

**Before launching the full set, verify that parallelism does not corrupt the
simulated clock.** PLATO uses a simulated wall-clock (anchor_sleep), so GPU
contention should not affect `sim_time_s` — but verify rather than assume:

1. Run one config (alpha=1.0, gpu, seed 1) for 200 rounds alone. Record
   `sim_time_s` at the final round.
2. Run the identical config again for 200 rounds while a second job is running.
3. Compare. Identical or within 0.1% → safe to parallelise. Report both numbers.

If they differ materially, **stop and report** — do not run the full set.

## What to record

Per run, emit a long-format CSV with the same columns as the existing
`accuracy_curves.csv` (`experiment, arm, seed, round, sim_time_s, accuracy`),
plus a per-run summary row containing:

- plateau accuracy: median of the last 10% of evaluations
- noise: standard deviation of residuals around a 101-eval rolling median
- **time to accuracy at 70%, 75%, 78% and 80%**, in simulated hours
- **rounds to accuracy at the same four thresholds**
- dispatch count, and dispatch share per tier (xavier / orin / thor)
- total simulated time to complete 4,000 rounds

**Record all four thresholds, not just 80%.** Lowering alpha will lower the
plateau. If alpha=0.1 plateaus near 80%, then a first crossing of 80% measures
when noise happened to touch the ceiling, not learning speed — and we would have
to re-run the whole set to recover a usable threshold. Computing all four costs
nothing because they come from the same curve.

## What NOT to do

- Do not interpret, rank the arms, or draw conclusions. Emit data.
- Do not change the dataset, the model, the fleet mix, or the data-to-tier mapping.
- Do not tune anything to make a result appear.
- Do not skip the alpha=1.0 arm because "we already have it".

## Note on what this run can and cannot answer

One seed cannot establish separability — the decision rule needs a pooled standard
deviation across at least 3 seeds. This run is a **scouting run**: it says whether
the direction is worth spending the other two seeds on. Label it that way in any
output.

---

# Tracking table

Status: `done` / `running` / `queued` / `blocked`

## Phase 1 — alpha sweep (surgical, N=20, C=4, 4000 rounds)

| cell | seed 1 | seed 2 | seed 3 | plateau | noise SD | usable threshold |
|---|---|---|---|---|---|---|
| alpha=1.0, random mapping | queued | queued | queued | | | |
| alpha=0.1, random mapping | queued | queued | queued | | | |

## Phase 1 results — time to target (simulated hours, mean +/- SD)

| alpha | arm | TTA @70% | TTA @75% | TTA @78% | TTA @80% | effect / noise vs baseline |
|---|---|---|---|---|---|---|
| 1.0 | baseline | | | | | reference |
| 1.0 | gpu | | | | | |
| 1.0 | cpu | | | | | |
| 1.0 | observed | | | | | |
| 0.1 | baseline | | | | | reference |
| 0.1 | gpu | | | | | |
| 0.1 | cpu | | | | | |
| 0.1 | observed | | | | | |

## Phase 1 — the question, answered

| contrast (effect / noise) | alpha=1.0 | alpha=0.1 | grew? |
|---|---|---|---|
| gpu vs baseline | | | |
| gpu vs cpu | | | |
| gpu vs observed | | | |

## Phase 1 — control check

| alpha | arm | xavier share | orin share | thor share | sim hours for 4000 rounds |
|---|---|---|---|---|---|
| 1.0 | baseline | | | | | |
| 1.0 | gpu | | | | | |
| 1.0 | cpu | | | | | |
| 1.0 | observed | | | | | |
| 0.1 | baseline | | | | | |
| 0.1 | gpu | | | | | |
| 0.1 | cpu | | | | | |
| 0.1 | observed | | | | | |

Confirms the selectors still behave differently at both alphas. If xavier share is
identical across arms, the selector is not working — which is a different problem
from selection not mattering.

## Decision rule (fixed before running)

**Separable** = |mean TTA difference| / pooled SD > 2.0, with the same sign in all
three seeds. Requires 3 seeds; seed 1 alone only indicates direction.

| outcome | meaning | next |
|---|---|---|
| 0.1 separates, 1.0 does not | alpha is the lever | adopt alpha=0.1 as the standard condition |
| neither separates | alpha is not the lever | try C-sweep (C=2, C=1) or adversarial mapping |
| both separate | the old null was a single-seed artefact | seeds were the missing piece, not alpha |
| 1.0 separates, 0.1 does not | unexpected | check config before interpreting |

## Later phases (not this run)

| phase | what | status |
|---|---|---|
| 2 | C-sweep: C=2 and C=1 at N=20 | queued |
| 3 | adversarial data-to-tier mapping (skewest partitions to Xavier) | needs design |
| 4 | carry the winning cell into SUMO, then add Stage 0 | queued |
| 5 | open population | queued |

## Sanity checks before trusting anything above

- alpha=0.1 plateau should be **lower** than alpha=1.0, and its noise **higher**.
  If not, the alpha setting is not taking effect — stop and check the config.
- The chosen threshold must sit at least **5 points below the plateau of every
  cell it is applied to**. Pick it from the plateau column, do not fix it in advance.
- Fresh alpha=1.0 runs should reproduce the shape of the existing surgical results.
  If they do not, something else changed.
