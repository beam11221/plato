# Stage 3 — run the sweep. 23 runs, one seed.

Stage 1 and Stage 2 are accepted. The fleets are locked at sigma = 0.149166, the
`gpu` arm is no longer an oracle, and the clock is reconstructed outside PLATO.

This is a **scouting campaign**: one seed everywhere, to see the shape of the
landscape. Extra seeds come after, on whichever cells turn out to matter.

---

## Step 0 — one optional fix, timeboxed to 2 hours

`determinism_after.txt` § 6 found the remaining source of run-to-run divergence:
PLATO pops the smallest `(finish_time, client_id)` off a heap, and `finish_time`
carries the host's accumulated `processing_time`. When two clients finish within a
few tenths of a second of each other in simulated terms, **the machine decides which
merges first, not the experiment.**

Stage 2 concluded this cannot be fixed without editing `plato/servers/base.py`.
There may be a third way: `fedasync_seeded.py` already manipulates module state
before PLATO is imported, so the finish-time computation could be **patched at
runtime** from the example folder — keeping `processing_time` out of `finish_time`
without editing any file under `plato/`.

That is the same quantity Change C already removes when reconstructing the clock.
Removing it at run time as well makes the merge order agree with the time axis we
actually use. It is not a change to the model being simulated; it stops the
experiment host's own elapsed time from leaking into it.

**Before patching, check whether `processing_time` is consumed anywhere else** —
aggregation weighting, staleness computation, logging that later analysis depends
on. If it is, abandon this step and say why.

**Verification if it goes in:** repeat the 200-round determinism check. Success is
two runs producing an identical dispatch sequence, identical merge order, and an
identical reconstructed clock.

**Timebox: 2 hours.** If it is not working and verified by then, abandon it, record
what was tried, and go to Step 1 with the campaign unchanged. Do not let this block
the sweep.

---

## Step 1 — the runs

Fixed in every cell: alpha = 0.8 · 4,000 rounds · seed 4126 · sigma = 0.149166 ·
fleet mix 50/35/15 · `N x partition_size = 50,000` · `staleness_bound = 6.25 x C`

### Batch 1 — C sweep, N = 20, partition 2,500

| C | staleness_bound | arms | runs |
|---|---|---|---|
| 1 | 7 | baseline, gpu, cpu, observed | 4 |
| 2 | 13 | baseline, gpu, cpu, observed | 4 |
| 4 | 25 | baseline, gpu, cpu, observed, **gpu again as `_rep2`** | 5 |
| 8 | 50 | baseline, gpu, cpu, observed | 4 |

**17 runs.** C = 6 and C = 10 are deliberately skipped to save time; if the effect
looks flat between C = 4 and C = 8 they get added back.

`_rep2` is a byte-identical repeat of the C=4 gpu config under a different output
tag. It measures the run-to-run spread of **time-to-target at 4,000 rounds**, which
has never been measured — every determinism check so far stopped at 200 rounds,
where the model is at its noisiest. Its number decides how every other number in
this campaign may be read.

### Batch 2 — fleet size, C = 4, bound 25

| N | partition_size | arms | runs |
|---|---|---|---|
| 40 | 1,250 | baseline, gpu, cpu, observed | 4 |

**4 runs.** N = 20 comes from Batch 1. N = 30 is excluded — it does not divide the
fleet mix evenly.

### Batch 3 — upper bound

| cell | arm | runs |
|---|---|---|
| N=20, C=4 | `gpu_oracle` | 1 |
| N=20, C=8 | `gpu_oracle` | 1 |

**2 runs.** Run these last.

**Total: 23 runs.**

---

## Parallelism — one rule

**Run the arms of ONE cell together, and never two cells at once.**

`processing_time` enters the merge order, so a cell run on a loaded machine can get
a different dispatch sequence than the same cell run on an idle one. Running a
cell's four arms simultaneously means all four meet the same machine conditions, so
the arm-vs-arm comparison — the only one we care about — stays fair. Running two
different cells at once would make them incomparable to each other.

If Step 0 succeeded, this constraint is gone and cells may be run in any
arrangement. Say which regime the campaign actually ran under.

If two machines are available, split **by cell**, not by arm, and record which
machine ran which cell.

---

## What to record, per run

Emit `accuracy_curves_reconstructed.csv` with both clocks:

```
experiment, arm, seed, round, sim_time_plato, sim_time_reconstructed, accuracy
```

Plus a summary row per run:

- plateau: median accuracy over the last 10% of evaluations
- noise: SD of residuals around a 101-evaluation rolling median
- **time to target at 70 / 75 / 78 / 80%, from `sim_time_reconstructed`**
- rounds to target at the same four thresholds
- mean and max report staleness, measured from the server log
- dispatch count and dispatch share per tier
- seconds per round: total reconstructed time / rounds
- belief-vs-reality gap on E[T] for that fleet

Record all four thresholds. A cell whose plateau lands low makes a high threshold
meaningless, and re-running to recover one costs a day.

**Time-to-target is read from the reconstructed clock only.** `sim_time_plato` is
kept for audit, not for analysis.

---

## Reporting

Emit data and the tables below. Do not rank the arms, do not declare a winner, do
not explain away a result that looks wrong.

**Table A — the gate.** C=4 gpu against `_rep2`: plateau, noise, and time-to-target
at all four thresholds, with the percentage difference. This is the campaign's
noise floor at full length.

**Table B — C sweep.** Per cell and arm: time-to-target at the cell's usable
threshold, as a percentage against that cell's own baseline. A threshold is usable
only if it sits **at least 5 points below the lowest plateau in that cell** — pick
it from the measured plateaus, do not fix it in advance. State which threshold each
cell was read at.

**Table C — staleness.** Mean report staleness per cell against C. The prior
campaign measured almost exactly C−1; whether that still holds on the jittered
fleets is a check on the whole setup.

**Table D — fleet size.** N=20 against N=40 at C=4, same columns as Table B.

**Table E — the upper bound.** `gpu` against `gpu_oracle` at C=4 and C=8.

**Table F — control.** Dispatch share per tier for every run. If the arms show the
same tier mix, the selector is not working — which is a different problem from
selection not mattering, and would invalidate everything else.

---

## Written down before the runs start

| # | prediction |
|---|---|
| 1 | C=1 gives a null across all four arms — staleness is zero there, so there is nothing for a staleness-aware selector to exploit |
| 2 | The selector advantage grows from C=1 to somewhere around C=4, then flattens or falls by C=8: raising C adds staleness (which helps) but shrinks the idle pool to N−C (which hurts) |
| 3 | `cpu` is slower than `baseline` in every cell with C > 1 |
| 4 | N=40 gives the same advantage as N=20 at C=4 — if staleness really is set by C alone, fleet size should not matter |
| 5 | `gpu` loses to `gpu_oracle`. **If they tie, the jitter did not make estimation matter and the oracle problem is not actually fixed** |
| 6 | `gpu` against `observed` stays undecided at one seed. That is an acceptable answer, not a failure |

Report each as held or not held. A prediction that fails is a result; it is not
something to reconcile.

---

## Stop conditions

Stop and report rather than continuing if:

- Table F shows the arms dispatching the same tier mix — the selector is not working
- mean staleness does not track C−1 at all — something changed that we do not
  understand
- `_rep2` differs from `gpu` by more than 10% on time-to-target — one seed cannot
  support any comparison in this campaign, and the remaining runs should be
  reconsidered before being spent
- any cell's plateau leaves no threshold 5 points clear

---

## Do not

- change sigma, the fleet mix, the tier files, the dataset, the model, the optimiser
  or the selector
- read time-to-target from PLATO's clock
- add seeds, cells or arms beyond the 23 listed
- tune anything to make a prediction come true
