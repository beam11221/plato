# Open vs closed population, `p_return`, and data partitioning

How the vehicles in the SUMO traces behave, what `p_return` (Stage 11) does to
them, and how CIFAR data partitions are assigned. Every number here comes from
`population_facts.py` (read-only, ~6 s). Run it again if a trace or the
allocator changes. Measured 2026-10-05.

Traces compared (same geometry, N ≈ 20, R = 400 m, seed 4126):

- open: `sumo_trace/e1_open_poisson_seed4126_range400/`
- closed: `../sumo_close_loop/sumo_trace/e1_closed_seed4126_range400/`

The other closed traces (`e1_closed_n20_seed1`, `e1_closed_n100_seed1`) use
R = 300 m, so they are **not** directly comparable to the open trace.

---

## 1. Geometry: R = 400 m is a radius, so 800 m of road is covered

- The road is a straight line 1,200 m long, one lane each way (EB, WB). The RSU
  is at x = 600 m and `signal_mode: disk`.
- In every row, `dist_to_rsu_m` equals |x − 600|. In-coverage rows span
  x = 200 … 1000 m exactly, so **800 m is covered and 400 m is not**
  (200 m at each end, on both directions).
- This is consistent with the timing: mean residence is 46.2 s at about
  17.3 m/s, which is roughly 800 m. In the closed loop, a vehicle is out of
  coverage for about 24.5 s per lap: about 400 m of driving plus re-insertion.
- R = 300 m traces cover x = 300 … 900 m (600 m).

## 2. Vehicle behaviour

Both regimes follow the same rule: **a vehicle is born at the start of its
direction, drives the full road, and leaves only at the far end.** The spawn
position is not random, and no vehicle disappears mid-road.

| | Open (Poisson) | Closed (N = 20) |
|---|---|---|
| Spawn | Always 5.1 m from the start of its direction (`departPos="base"`). **Only the spawn time is random**: two independent Poisson flows (`fEB`, `fWB`), 525.7 veh/h each | Initially spaced evenly along the road (x = 109, 218, …). After each lap, at the start of the road |
| Motion | One direction, never reverses, no gaps in the 1 Hz trace | Same within each lap |
| Leaves at | Far end only (last position ≥ 1,180 m; one vehicle cut off by the end of the trace) | Far end only (position ≥ 1,180 m before every disappearance) |
| After the end | Gone for good: **one coverage window per vehicle**, 63–102 s on the road | Re-inserted after 1–2 s with the **same id**. 5,075 of 10,219 re-insertions flip direction (U-turn-like), the rest restart the same direction (ring-like). The ~50:50 split suggests a random choice, but the generator script is not in this repo to confirm it |

`e1_closed_n100_seed1` is congested: re-insertion is sometimes delayed by
30–95 s (no room at the road start), and 3,336 rows are near standstill.

## 3. Trace numbers (t ≥ 0)

| | Open | Closed (N = 20, R = 400) |
|---|---|---|
| Unique vehicles | 10,523 | 20 |
| Vehicles on road | 20.25 ± 4.44 (6–38) | 19.70 ± 0.51 (16–20) |
| Vehicles in coverage | 13.45 ± 3.64 (2–29) | 13.08 ± 2.85 (2–20) |
| Coverage windows | 10,509 | 10,223 |
| Residence per window | 46.2 ± 4.0 s | 46.2 ± 4.2 s |
| Windows per vehicle | 1 | ~511 |

Occupancy, residence and arrival rate are almost identical. The difference
that matters is **identity**: in the open trace a new vehicle arrives every
time, while in the closed trace the same 20 vehicles return about every 70 s.

---

## 4. What `p_return` is

`p_return` (Stage 11, `occupant_identity.ReturningPopulation`) is **not** a
new SUMO run, and it is not `p_recur`. `p_recur` only takes 0 or 1, and every
Stage 11 config sets `p_recur = 0`. Every `p_return` value and every run seed
uses the same physical trace (`physics_seed = 4126`). `p_return` only relabels
**who** occupies a slot.

**Why the relabelling is needed.** In the open trace, no physical vehicle ever
returns. A "returning vehicle" therefore cannot be a trace vehicle. Instead it
is one of 20 synthetic **commuter** identities, `C1 … C20`, which are the
Stage 3–10 closed-loop fleet. Each time a commuter returns, it borrows the
coverage window of some trace vehicle.

On **every slot allocation**, which happens when a trace vehicle enters
coverage and gets one of the M = 20 slots (not when it spawns in SUMO):

```
draw u ~ U(0, 1)
if u < p_return and some commuter is not currently in coverage:
    occupant = commuter Ck, chosen uniformly among the free ones
               (tier, sleep and partition are Ck's; the trace vehicle's are ignored)
else:
    occupant = the trace vehicle itself (a transient)
               (tier from the trace, partition = hash(veh_id) mod 20)
in both cases: enter/exit times and position come from the trace vehicle
```

- If all 20 slots are full, the arriving vehicle gets no slot and no draw is
  made. This happens 216 times over the whole trace (~2%).
- "All commuters already in coverage" never happened (`exhaustions = 0`).
- Both random streams are seeded from the physics seed. The identity sequence
  is therefore the same for every arm and run seed, and the u-sequence is the
  same for every `p_return` (common random numbers).

Example: commuter C7 at p = 0.25, t ≤ 4,600 s. C7 made 18 visits, each one on
a different trace vehicle:

| t (s) | Window borrowed from | hash(veh_id) mod 20 | Partition C7 uses |
|---|---|---|---|
| 0 | fEB.6 | 18 | 6 |
| 69 | fEB.28 | 1 | 6 |
| 327 | fEB.78 | 17 | 6 |
| 524 | fWB.85 | 2 | 6 |
| 650 | fEB.128 | 9 | 6 |

## 5. Counts under `p_return`

**Vehicles on the road and in coverage do not change with `p_return`.** For
every p they are the open-trace values: 20.25 on the road, 13.45 in coverage,
and 10,523 physical vehicles (10,271 admitted to a slot from t ≥ 0).

The thing that changes is the number of **unique identities**, written
commuters + transients:

| p_return | to 4,000 s | to 4,600 s | to 7,200 s | Whole trace (36,000 s) |
|---|---|---|---|---|
| 0 | 1,135 (0 + 1,135) | 1,306 | 2,004 | 10,271 |
| 0.25 | 861 (20 + 841) | 987 | 1,496 | 7,713 |
| 0.5 | 583 (20 + 563) | 664 | 996 | 5,145 |
| 0.75 | 311 (20 + 291) | 355 | 501 | 2,608 |
| 1 | 20 (20 + 0) | 20 | 20 | 20 |

A run of 4,000 rounds ends somewhere between ~4,000 and ~7,200 s, depending
on the arm (see `T@4000` in `../gpu_tiers_dataset/stage11_evidence/STAGE11_analysis.md`).
Count at that run's T. For example, gpu+F at p = 0.25 (seed 4126) ends at
4,062 s with 846 transient identities, against 841 at 4,000 s here.

Rough rule: transients ≈ (1 − p) × allocations, and there are **always
exactly 20 commuters, whatever p is**.

## 6. "p_return = 0.25 means 25% commuters": true only for some measures

All values at t ≤ 4,600 s:

| Measure | p = 0.25 | p = 0.5 | p = 0.75 |
|---|---|---|---|
| Share of coverage **visits** made by commuters | 26.0% | 50.7% | 74.3% |
| Share of vehicles **in coverage at an instant** (time-averaged) | 25.9% (3.4 of 13.1) | 50.7% (6.6 of 13.1) | 74.4% (9.7 of 13.1) |
| Share of **unique vehicles (headcount)** | 2.0% (20 / 987) | 3.0% (20 / 664) | 5.6% (20 / 355) |

Visit share and instantaneous share both equal p, because commuters and
transients borrow windows from the same trace and so have the same residence.
Headcount share is far smaller: 20 vehicles come back many times, while
hundreds come once.

**Recommended wording:** "p_return is the fraction of coverage visits made by
returning vehicles; equivalently, about p_return of the vehicles in coverage
at any instant are returning (a fixed pool of 20)." Avoid "p_return of the
population", which reads as headcount.

---

## 7. Data partitioning

### The partitions

- CIFAR-10 train (50,000 images) is used, with `partition_size = 2500` and
  M = 20, giving **20 partitions** (20 × 2,500 = 50,000). This is the same as
  Stages 3–6.
- `sampler = "noniid"` is PLATO's Dirichlet sampler (`plato/samplers/dirichlet.py`)
  with `concentration = 0.8`.
- Partition j uses `sampler_id = j + 1`:
  - Its label proportions are drawn from Dirichlet(0.8) with seed
    `random_seed × sampler_id`.
  - It then draws 2,500 images by weight, **without replacement within the
    partition**.
- **Partitions are not disjoint.** Each one samples the full 50,000
  independently, so different partitions can contain the same image. The
  closed loop works the same way.
- The same sampler_id always gives **exactly the same images**. Only the
  training seed differs between occupants, because it is keyed on identity,
  so minibatch order differs.

### Rule: the partition follows the identity, never the slot or the trace vehicle

| Identity | Partition |
|---|---|
| Transient (identity = `veh_id`) | `blake2b("partition:" + veh_id) mod 20` (`stage7_population.partition_index`) |
| Commuter `Ck` | **k − 1** (C1 → 0 … C20 → 19) |

- Both groups share **all 20 partitions**. There is no split such as 5 for
  commuters and 15 for transients.
- At p = 0.5, t ≤ 4,600 s, every partition is used by exactly one commuter
  (23–39 visits) and by 22–44 transients (one visit each).
- **Not keyed on the slot:** a recycled slot must not hand a new vehicle the
  previous occupant's data (Stage 7).
- **blake2b, not `hash()`:** Python's `hash()` of a string changes between
  processes, so the server and the clients would disagree on a vehicle's
  partition.

### Why commuters do not use `hash(veh_id)`

A commuter has no `veh_id` of its own. Each visit borrows a different trace
vehicle, as section 4 shows: C7's 18 visits came from vehicles whose hashes
take 11 different values. Hashing the borrowed `veh_id` would give C7 new data
on almost every return, which defeats the point of a returning vehicle.

Hashing `"Ck"` would also be stable. `k − 1` is used instead for two reasons:

1. **Closed-loop parity.** Client k in the closed loop trains on partition
   k − 1, so p = 1 reproduces the closed cell. The anchor check is within
   +2–3%.
2. **No commuter–commuter clash.** The 20 commuters cover 20 distinct
   partitions.

### Consequence: occupants in coverage at the same time often share a partition

A partition is reused **over time** by design, as in the closed loop. It is
also often held by **two occupants in coverage at once**: commuter Ck with a
transient whose hash is k − 1, or two transients with the same hash. Two
commuters never share a partition.

All values at t ≤ 4,600 s, over 1,306 allocations:

| p_return | Arrivals whose partition is already held in coverage | Time with ≥ 1 shared partition | Occupants / distinct partitions | Clashing pairs (transient–transient, commuter–transient) |
|---|---|---|---|---|
| 0 | 47.9% | 94.6% | 13.1 / 9.7 | 626, 0 |
| 0.25 | 45.3% | 92.7% | 13.1 / 9.9 | 340, 252 |
| 0.5 | 38.9% | 90.9% | 13.1 / 10.4 | 152, 356 |
| 0.75 | 26.9% | 81.6% | 13.1 / 11.2 | 41, 310 |
| 1 | 0% | 0% | 13.1 / 13.1 | none |

- This is the birthday problem: about 13 occupants drawing from 20 partitions.
- The open population (p = 0) therefore differs from the closed loop (p = 1)
  in **data diversity in coverage** as well as in server-side history: about
  9.7 versus 13.1 distinct partitions. Keep this in mind when attributing a gap.
- These figures count **occupants in coverage**, not concurrent dispatches.
  With C = 4, clashes among updates that are actually in flight are rarer.
  Counting them needs the per-run dispatch logs.
- Removing clashes is an experiment-design decision, and nothing has been
  decided. Options:
  - Partition-aware allocation, which conflicts with "partition follows the
    vehicle".
  - K > 20 partitions, which changes data per client and closed-loop parity.
  - Dedicated commuter and transient partition sets (e.g. 5 / 15), which
    breaks p = 1 parity and ties data diversity to p.

  Any of these would be a new stage, not a change to Stage 11.

---

## 8. Caveats found along the way

- Two closed configs carry a stale `n_vehicles_target: 100`:
  - `e1_closed_seed4126_range400/config.yaml` has `n_vehicles_closed_loop: 20`
    and 20 vehicles in the data.
  - `e1_closed_n20_seed1` has the same mismatch.
- `t_bar_train_s` (30 / 18 / 10) in every trace config is provisional and is
  **not** used. Experiments use `stage7_population.T_TIER`
  (15 / 1.923077 / 1).
- The script that generated the traces is not in this workspace. Section 2's
  closed-loop re-insertion behaviour is inferred from `trace.csv`.

## Reproduce

```bash
cd examples/async/fedasync/sumo_open_loop
python3 population_facts.py                 # horizon 4,600 s
python3 population_facts.py --horizon 7200  # clash/share tables at another T
```
