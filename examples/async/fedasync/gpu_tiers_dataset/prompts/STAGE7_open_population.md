# Stage 7 — open population. Integration, one gate, then 8 runs.

This is the cell the whole programme has been pointing at. Stages 3–6 established that
the scoring claim is a null in a closed pool and explained why (§3–4 of
`consolidated-findings.md`): `s(τ̂)` is a fixed three-valued constant, `G_i` diverges by
tier until it cancels it, and any reactive estimator catches up once it has ~30–70
observations of a client. Open population is the regime where that catch-up cannot
happen.

A new SUMO trace has been generated and is provided:
`sumo_trace/e1_open_poisson_seed4126_range400/`.

Do not touch anything under `plato/`.

---

## What the trace actually contains — verified, not assumed

| | |
|---|---|
| vehicles | **10,523**, each with `n_passes = 1` and `n_windows = 1` — every vehicle crosses once and never returns |
| arrivals | Poisson, calibrated to 525.67 veh/h per direction (two directions), converged to N = 19.97 on the road |
| geometry | road 1,200 m, 1 lane per direction, RSU at 600 m, coverage radius 400 m → the same 800 m span `[200, 1000]` as the closed-loop trace |
| speed | `v_max = 19.444444 m/s`, same as closed loop; `sigma = 0.5` driver imperfection |
| residence | ~42–47 s per window in the head of the file; the full distribution is a gate deliverable |
| realised tier mix | 0.4911 / 0.3491 / 0.1597 (xavier / orin / thor) |
| `veh_id` | **strings** — `fEB.0`, `fWB.12` — not integers |
| time origin | `warmup_s = 150`, so `t_enter_s` is negative for early windows; `n_windows_open_at_end_dropped = 9` |
| files | `trace.csv` (731,143 rows, per-second position), `coverage_windows.csv` (10,510 windows), `vehicles.csv`, `config.yaml` |

## Three things that are fixed and are not open to re-derivation

**1. The training-time model stays exactly as in Stages 3–6.**

`config.yaml` contains `t_bar_train_s: {xavier: 30, orin: 18, thor: 10}` carrying the
flag `t_bar_train_provisional: true`. **Ignore it.** Its tier ratio is 3 : 1.8 : 1; every
result from Stage 3 onward uses 15 : 1.923077 : 1. Using the config's values would make
tier separation 3× instead of 15× and nothing would be comparable to any earlier stage.

Use `T_tier = {xavier: 15.0, orin: 1.923077, thor: 1.0}` and `sigma = 0.149166`.

**2. Per-vehicle sleep is drawn the same way the closed-loop fleet was.**

There is no fleet file for 10,523 vehicles and there should not be one.
`sleep_veh = T_tier(tier) · exp(ε)` with `ε ~ N(0, sigma²)` drawn **deterministically
from `(base_seed, veh_id)`** — a hash of the string id, so the same vehicle gets the
same sleep in every run and every arm. Emit the realised per-tier mean and CV over all
vehicles and check them against the closed-loop fleet's (14.37 / 1.78 / 1.07 s).

**3. Data partitions follow the vehicle, not the slot.**

Partition index = `hash(veh_id) mod 20`, with `partition_size = 2,500` (Step 1 fixes
`M = 20`, so `M × partition_size = 50,000`). If partitions were tied to the slot, a
recycled slot would hand a brand-new vehicle the previous occupant's data, and `G_i`
would carry information across an identity boundary that is supposed to be sealed —
which matters because §4 showed `G_i` is the dominant term late in a run.

Two vehicles in different slots may therefore hold the same partition at the same time.
That is correct: vehicles draw from a shared data distribution, they do not own
disjoint shards of it.

---

## Step 1 — the slot allocator

PLATO needs a bounded set of integer client ids; the trace has 10,523 string ids. Map
the vehicles currently inside coverage onto `M` slots.

### A slot is held for coverage residency, never for selection

A vehicle takes a slot when it **enters** coverage and releases it when it **leaves**,
whether or not it was ever selected, and whether or not it is training at the time.

This is not an implementation detail — it is what makes the rest of the stage readable.
Occupancy then depends on the trace alone, so the occupancy sequence is **byte-identical
across all eight runs**, and any vehicle turned away for lack of a slot is turned away
in every arm at the same moment. A slot bound to selection instead would make occupancy
depend on selector behaviour, and every arm would face a different population.

**Verify this rather than assume it:** emit a hash of the per-round occupancy sequence
in each run and confirm all eight match.

### `M = 20`, fixed

`M` is the RSU's concurrent-session capacity, not a tuning knob, and it is set to 20:

- `partition_size = 2,500`, identical to every cell from Stage 3 onward, keeping
  `M × partition_size = 50,000` exactly
- makes this stage directly comparable to Task 4's closed cell (Stage 0 off, N=20,
  C=4), which is the natural cross-stage partner
- `κ` is set by in-coverage occupancy (~13–14), not by `M`, so the regime this stage
  exists to reach does not depend on this choice

Mean in-coverage occupancy is expected around 13–14 (20 vehicles on 1,200 m × the
800 m coverage span; equivalently ~0.292 arrivals/s × ~45 s residence), so occupancy
should exceed 20 roughly 4–5% of the time. **Arrivals turned away at that rate are
expected and acceptable**, because the truncation is symmetric across arms and
independent of tier. Log a `slot_exhausted` event for each one.

Two checks make that claim rather than assume it, both gate deliverables:

1. the tier mix of turned-away vehicles must match the admitted mix (49.1 / 34.9 / 16.0)
2. the occupancy-sequence hash must be identical across all eight runs

### `p_recur` — the identity policy, and the crux of this stage

`p_recur` is the probability that a freed slot retains its previous occupant's
server-side state rather than starting clean.

- `p_recur = 0` — **open population.** On recycle, everything the server knows about
  that slot is reset: the reactive arm's remembered staleness returns to its unseen
  default (`0.01`), `G_i` returns to the unexplored prior, selection counts return to
  zero. The new vehicle's tier and sleep are its own.
- `p_recur = 1` — **closed population, same trace.** State persists across recycles, so
  the reactive arm accumulates history exactly as it does in the closed-loop cell.

`p_recur = 1` on this trace is the control, and it is a better one than the Stage 4/5
closed-loop cell: same arrivals, same windows, same everything but the identity policy.

**This reset is the single thing most likely to be wrong and least likely to announce
itself.** If state leaks across a recycle, `p_recur = 0` is quietly closed population
and the stage measures nothing. Required evidence before any run:

- a test that asserts, after a recycle at `p_recur = 0`, that the reactive arm's `τ̂`
  for that slot is the unseen default, `G_i` is at the prior, and `times_selected` is 0
- a test that asserts the opposite at `p_recur = 1`
- a log line per recycle carrying `slot_id, outgoing_veh_id, incoming_veh_id, tier_out,
  tier_in, state_reset` so the run itself is auditable
- tier must follow the incoming vehicle, never the slot

---

## Step 2 — pre-run gate. Report and stop.

From the trace and the allocator alone, no FL training:

1. **Crossing-time distribution** — median, p10, p90 of `residence_s`, and the fraction
   below 41.142857 s. Compare to the closed-loop trace's median 45.0 s (p10 42, p90 52).
2. **Occupancy** — median, p90, p99 and max of concurrent in-coverage vehicles, and
   the fraction of time occupancy is at or above `M = 20`.
3. **Feasible-and-idle pool per round**, median / p10 / p90, and the fraction of rounds
   below `2 × C = 8`. Same gate as Stage 4: **stop and report if the median is below
   2C.** Note that with Stage 0 off (Step 3) the predicate is not applied at run time,
   but the number still says whether selection has any freedom.
4. **κ — expected dispatches per vehicle before it exits.** Estimate from the trace:
   window length ÷ seconds-per-round ÷ eligible-pool size. The closed-loop cells ran at
   70–188; the prior estimate here is ≈ 3, well below the measured k\* of 30–70. Report
   the distribution, not just the mean.
5. **Tier mix among vehicles that appear in coverage**, to confirm 49.1 / 34.9 / 16.0
   survives the window filtering.
6. **Slot exhaustion at `M = 20`**, replayed over the whole trace: the fraction of
   arrivals turned away, **and the tier mix of those arrivals against the admitted
   mix**. The rate is expected around 4–5%. A rate in that range is a property of the
   RSU's capacity and is accepted; a tier mix that departs from 49.1 / 34.9 / 16.0
   means the allocator is keyed to something other than coverage residency, which is a
   defect — stop and report.
7. **Warm-up handling** — how negative `t_enter_s` and the 9 dropped end-of-sim windows
   are treated. Clip to `t ≥ 0` as `remap_fleet.py` does, and say how many windows that
   removes.

Stop and report all seven before running anything.

---

## Step 3 — the 8 runs

`C = 4`, α = 0.8, 4,000 rounds, seed 4126, `upload_policy = coverage_gated`,
**Stage 0 filter OFF**, `V_i`-weighted lottery unchanged.

| | `p_recur` | arms | runs |
|---|---|---|---|
| open | **0** | baseline, gpu, cpu, observed | 4 |
| closed (control) | **1** | baseline, gpu, cpu, observed | 4 |

Stage 0 stays off on purpose. This cell tests the **scoring** claim, and the filter has
already been shown to dominate the selector when it is on (§12). Stage 0 in open
population is a separate cell and is not part of this stage.

---

## Step 4 — what to report

Per run, the same set Stage 6 produced: conservation law, cumulative
`sim_time_reconstructed` at rounds 100 / 250 / 500 / 1,000 / 2,000 / 4,000, waste
overall and per tier, dispatch share per tier per window, plateau, noise, TTA and
rounds-to-target at 70 / 75 / 78 / 80%, mean and max staleness.

Plus, specific to this stage:

- **κ measured**, not estimated: dispatches per vehicle before exit, distribution and
  mean, per arm
- **the reactive arm's information state**: at each dispatch, how many prior
  observations the server held for that slot's current occupant. At `p_recur = 0` this
  should be near zero almost always; at `p_recur = 1` it should climb
- **`G_i` at dispatch, split by whether the value is the unexplored prior or measured**,
  per window. §4 predicts that at `p_recur = 0` most candidates sit at the prior, which
  is what hands the ranking to `s(τ̂)`
- **`s(τ̂)` per tier per arm.** For `observed` specifically, report how often an unseen
  vehicle's default `τ̂ = 0.01` (giving s = 0.995, above thor's 0.822) makes it outrank a
  known fast vehicle
- **recycle log summary**: recycles per run, state resets, slot-exhaustion events, and
  the occupancy-sequence hash
- **`gpu` minus `observed`** on cumulative time at round 4,000 and on early
  seconds-per-round, for both `p_recur` values — the headline comparison

---

## Step 5 — written down before the runs start

| # | prediction |
|---|---|
| 1 | At `p_recur = 0`, **`gpu` beats `observed`** on cumulative time at round 4,000, by a margin larger than Stage 6's 1.2% noise floor. This is the thesis claim and this cell is its test |
| 2 | At `p_recur = 1`, `gpu` and `observed` are **indistinguishable**, reproducing every closed-pool cell so far. If they separate here, the difference is not about identity churn and the interpretation of predictions 1 and 3 changes |
| 3 | At `p_recur = 0`, **`observed` degrades toward `baseline`** — every unseen vehicle receives the same default `τ̂` and the same `G_i` prior, so the arm has nothing to rank on. This is the mechanism behind prediction 1, and it is the more direct measurement of the two |
| 4 | The `s(τ̂)` decay of §4 **does not appear at `p_recur = 0`**: `G_i` cannot diverge by tier because identities do not persist long enough to be starved, so `gpu`'s cumulative advantage should not cross zero within 4,000 rounds as it did at round ≈2,400 in the closed pool |
| 5 | `cpu` remains worst in both regimes — a wrong spec is wrong regardless of churn |
| 6 | Measured κ at `p_recur = 0` lands in the low single digits, far below the k\* of 30–70 measured in Stage 3 |

Report each as held or not held.

**Predictions 1 and 3 are the point.** Prediction 3 is the one to trust more: it is a
direct observation of the estimator's information state rather than an outcome subject
to seed noise. If 3 holds and 1 does not, say so plainly — that would mean the reactive
arm is crippled exactly as predicted and it still does not cost it anything, which is a
finding about the scoring function, not about the environment.

---

## Stop conditions

- the Step 2 gate's median feasible-and-idle pool is below `2 × C`
- slot exhaustion exceeds **15%** of arrivals — below that it is the RSU's capacity
  constraint, above it the admitted population is truncated enough that this is no
  longer the open population the trace describes
- the tier mix of turned-away vehicles departs materially from the admitted mix
- the per-round occupancy-sequence hash differs between any two of the eight runs
- the recycle-reset tests fail, or the recycle log shows state persisting at
  `p_recur = 0`
- realised per-tier sleep means or CV depart materially from the closed-loop fleet's
  (14.37 / 1.78 / 1.07 s at CV 15%)
- conservation-law error exceeds 3% in any run
- plateau falls more than 2 points below the 82.1–83.7% band seen in every prior cell

---

## Do not

- use `config.yaml`'s `t_bar_train_s` — it is provisional and on a different tier ratio
- tie data partitions to the slot instead of the vehicle
- allocate or release a slot on anything other than coverage entry and exit
- change `M` away from 20, or `partition_size` away from 2,500
- carry any server-side state across a recycle at `p_recur = 0`
- enable the Stage 0 filter in this stage
- change sigma, the tier ratio, the anchor, the dataset, the model, the optimiser, the
  scoring function or the selection rule
- add seeds, arms or `p_recur` values beyond the eight runs listed
- launch anything before the Step 2 gate is reported
- tune anything to make a prediction come true
