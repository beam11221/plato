# ADR-0001 — Wen-aligned closed-population mobility environment

- **Status**: Accepted
- **Date**: 2026-10-15
- **Scope**: SUMO mobility trace consumed by the asynchronous-FL experiments on RunPod
- **Decided by**: the thesis author, on 2026-10-15, after the first implementation was run
  to completion and reported 4/8 gate rows FAIL. The motion model and the G2 bound were
  chosen by the author from the options recorded below; the closed-loop mechanism (the
  position wrap) was an implementation call, since it is a SUMO limitation rather than a
  modelling question.
- **Supersedes**: the earlier E1 open/closed environments (`sumo/e1_sim.py`, `sumo/e1_closed.py`,
  `sumo/e1_open.py`), which used a different road, a different fleet size and an
  in-SUMO RSU

## Context

The thesis studies GPU-aware client selection for asynchronous federated learning in
vehicular networks and compares against Wen et al., *Information Sciences* 664 (2024).
To make the comparison meaningful, the mobility environment has to follow Wen §5.3.4
rather than the earlier home-grown E1 setting. The mobility trace is produced and
validated on the MacBook; all FL code runs later on a separate RunPod server, which
consumes the trace as a frozen bundle.

## Decisions

### Road geometry: one-way, 2,000 m, 2 lanes, lane speed 34 m/s

Wen §5.3.4 specifies a "2000×20" highway, i.e. 2,000 m of road, and gives no lane count.
We use **2 lanes in the same direction**: a single lane would make overtaking impossible
and would collapse the fleet to the speed of its slowest member, whereas 2 lanes keep the
geometry minimal while still allowing lateral movement. 34 m/s is the upper bound of Wen's
highway speed range and is set as the lane speed limit. The network is generated with
`netconvert` from plain `.nod.xml` / `.edg.xml` so the inputs stay reviewable.

### Speed model: one constant draw per vehicle, v ~ U[17, 34] m/s

Wen's Table 1/2 assigns one speed `V_k` per vehicle, and Fig. 2 computes the distance
travelled in a round as `T_com · V_k`. Each vehicle therefore draws a desired speed once,
tied to its ID, held constant for the whole trace, with `sigma = 0` and `speedDev = 0`
(no driver noise). The draw uses `numpy.random.default_rng` (PCG64) with seed
**20261015**, which matches the convention already used in `sumo/e1_closed.py`, and the
drawn speeds are written to `fleet_speeds.csv` before SUMO starts.

### Motion model: kinematic constant speed, not Krauss car-following

**This is the substantive deviation from the original specification, and it was approved
explicitly on 2026-10-15.**

The first implementation used SUMO's default Krauss car-following plus LC2013, as
specified. It fails the validation gate badly, and not because of an implementation
defect: on a 2 km two-lane closed loop, 30 vehicles with speeds spread over 17–34 m/s
bunch into a **standing platoon**. The three or four slowest vehicles (≈17 m/s) act as
moving blockers; once two of them sit side by side across both lanes, nothing can pass and
the whole fleet accumulates behind them. Measured on the full 30,000 s run
(kept in `sumo/wen_closed_v1_krauss_run/`):

| Gate | Krauss result | Required |
|---|---|---|
| G1 realized vs desired speed | 4/30 within ±2%, worst −34.4%, fleet mean −21.3% | ≥ 28/30 within ±2% |
| G3 re-entry delay p95 | 18.6 s | ≤ 1.0 s |
| G4 mean lap time | 4/30 within ±3%, worst +52.6% | 30/30 within ±3% |
| G6 in-coverage count | mean 17.6 but sd 10.4, p5 = 1 | mean ≈ 18 |

A lane-count sweep (1,800 s each) shows the congestion is structural rather than a matter
of too few lanes: 2 lanes → 5/30 vehicles within ±2%, 3 lanes → 19/30, 4 lanes → 23/30,
6 lanes → 24/30. Even six lanes misses the 28/30 bar, because overtaking itself costs
time whenever vehicles must yield at all.

The gate thresholds (G1 ±2%, G4 ±3%) encode Wen's own model, in which vehicles do **not**
interact: position is `x_k(0) + V_k · t`. We therefore reproduce that model directly —
each vehicle is pinned to its constant `V_k` through TraCI `setSpeedMode(0)` +
`setSpeed(V_k)`, so it never yields to another vehicle — and keep 2 lanes, LC2013 and the
Krauss parameters in the inputs as declared geometry. `collision.action` is set to `none`,
since collisions are meaningless once vehicles stop yielding.

Consequence to be stated in the thesis: the trace is a **kinematic** mobility model
faithful to Wen's assumptions, not a microscopic traffic simulation. Vehicles can pass
through one another longitudinally. Car-following realism is deliberately traded away for
comparability with the baseline. The Krauss run is retained so the congestion finding can
be reported as a limitation of Wen's constant-speed assumption at this density.

### Closed population with persistent IDs, implemented as a position wrap

The thesis scope is a closed population: the same 30 clients (`veh00`…`veh29`) must be
selectable for the entire training run, so a vehicle leaving at x = 2,000 m must come back
at x = 0 m **with the same ID**, same vType and same desired speed. New IDs would turn
every lap into a new client and destroy the staleness semantics of asynchronous FL.

The specification called for this as remove-and-reinsert on arrival. That cannot meet the
G3 bar of p95 ≤ 1.0 s: SUMO's insertion gap check refuses to insert at `departPos = 0`
with `departSpeed = "desired"` whenever the leader near x = 0 is slower — a 54 m gap is
not enough to insert at 32 m/s — and `insertionChecks="none"` on the vType does not
override that check in SUMO 1.21. Measured re-entry delay was p95 3.8 s / max 8.1 s even
in kinematic mode.

The loop is therefore implemented as a **position wrap**: on the step where the vehicle
would pass the end of the edge, TraCI `moveTo()` places it at `x + V_k·dt − 2000` on the
same lane. The vehicle never leaves the network, so identity, vType and speed persist
trivially and re-entry is instantaneous. SUMO retires a vehicle once its front is within
`POSITION_EPS` (0.1 m) of the lane end, so the wrap fires against `2000 − 0.1 m` and the
new position is clamped at 0; a lap is therefore 2,000 m to within 0.1 m and costs one
simulation step, making measured lap time `2000/V_k − 0.1 s` (about −0.15%, inside the
G4 ±3% band). The spec-literal remove-and-reinsert path is kept behind
`--loop reinsert` and is the default for `--model krauss`.

### RSU and coverage applied downstream

The RSU (x = 1,000 m, 10 m lateral offset) is **not** modelled in SUMO. The trace carries
only `t, veh_id, x_m, lane, speed_mps, lap`; the server derives distance, rate and
coverage from `x_m`. This keeps one trace valid for both coverage scenarios —
**S1** 100% coverage and **S2** coverage only for x ∈ [400, 1600] m — and lets the
coverage width change on the server without regenerating mobility. S2 statistics are
computed here only to feed the validation gate.

### Trace duration: 30,000 s after a 300 s discarded warm-up

Budget: ≈4,000 rounds × ≈11 s per task ÷ C = 3 ≈ 15,000 s for the fastest arm; 30,000 s
leaves headroom for the slower arms. The trace is **not** looped — if the server needs
more simulated time, the trace is regenerated with a longer duration, because looping
would reintroduce periodicity into client availability. Step length is 0.1 s and the trace
is sampled every 1.0 s, with trace time re-based so t = 0 is the end of warm-up.

### G2 upper bound relaxed from 33 to 32.5 m/s

Approved on 2026-10-15. Under the locked seed, 30 plain draws from U[17, 34] span
17.077–32.590 m/s. Reaching 33 m/s would require searching for a seed, which the
specification forbids, and 30 uniform draws rarely land within 1 m/s of the upper bound
(expected maximum is ≈33.45 m/s, so falling short is unremarkable). The gate now asks for
min ≤ 18 and max ≥ 32.5; the seed and the plain uniform draw are unchanged.

## Options considered and rejected

Both decisions were taken from an explicit menu after the Krauss implementation had been
run for the full 30,000 s and every gate number reported. Nothing was tuned to make a gate
pass; the failures were reported first and the specification was amended afterwards.

### For G1 / G3 / G4 (the platoon)

| Option | Measured effect | Outcome |
|---|---|---|
| **Kinematic constant speed (chosen)** | G1/G3/G4 pass by construction; vehicles reproduce `x_k(0) + V_k·t` exactly | **Accepted** |
| Keep Krauss, widen to 4 lanes | fleet-mean speed error −1.3%, re-entry p95 0.1 s, but G1 still only ~23/30 and G4 would need widening to about ±6% | Rejected: keeps realism but still fails the Wen-derived bar, so it buys neither comparability nor a clean gate |
| Keep 2 lanes and Krauss, relax G1/G3/G4 to the measured values | would need G1 fleet-mean −25%, G3 p95 ≤ 20 s, G4 +55% | Rejected: the resulting trace has ~28% of sampled instants with fewer than 6 feasible clients, which would confound the client-selection results with congestion |
| Stop and escalate without changing code | — | Rejected: the conflict is internal to the specification and was resolvable directly |

### For G2 (speed span)

| Option | Effect | Outcome |
|---|---|---|
| **Relax G2 to max ≥ 32.5 m/s (chosen)** | seed, draw and distribution all unchanged; drawn fleet spans 17.077–32.590 m/s | **Accepted** |
| Stratified (Latin-hypercube) draw, same seed | would span ~17.1–33.9 m/s and satisfy the original bound by construction, still reproducible | Rejected: changes the sampling scheme, and the plain i.i.d. draw is what Wen's model assumes |
| Search for a seed that satisfies max ≥ 33 | would satisfy the original bound | Rejected: this is the seed tuning the specification forbids |

## Validation outcome

The environment described above passes all eight gate rows on the full 30,000 s trace
(900,000 rows, 11,995 wraps). Full numbers are in `gate_report.md` inside the bundle.

| # | Check | Result |
|---|---|---|
| G1 | realized vs desired speed, ±2% | 30/30, error range −0.00%..+0.00% |
| G2 | speed span (min ≤ 18, max ≥ 32.5) | min 17.077, max 32.590 m/s |
| G3 | identity persistence, re-entry p95 ≤ 1.0 s | all 30 IDs present at 100.0% of sampled instants; delay mean/p95/max 0.000 s (n = 11,876) |
| G4 | mean lap time ±3% | 30/30, per-vehicle error −0.16%..−0.09%, 11,846 laps |
| G5 | S2 dwell ≈ 35–71 s | n = 11,862 passes; min 36.8 / median 42.5 / max 70.3 s; 100% in range |
| G6 | S2 in-coverage count | mean 18.03, sd 2.67; p5 14 / p50 18 / p95 22 |
| G7 | S2 feasibility, median ≥ 2·C = 6 | median 13.0; P(count < 6) = 0.0032 |
| G8 | lane use and overtaking | lane 0 37.9% / lane 1 62.1%; 1.62 lane changes per vehicle per lap |

Under the kinematic model the G5 histogram is a set of discrete spikes at `1200/V_k`, one
per vehicle, rather than a smooth distribution — the expected signature of constant speeds
and a useful check that the model is doing what it claims.

Bundle: `sumo/bundle_wen_closed_v1.zip`,
sha256 `18532ec2348ae53279b0cbad4d916712c23772c0a62ed7b70bcc7cd8cb5d8a24`
(a sha256 per file is in `manifest.yaml`).

## Consequences

- One frozen bundle, `bundle_wen_closed_v1.zip`, with a sha256 per file, is the only
  mobility input the RunPod side consumes; the FL code never re-runs SUMO.
- Realized speeds equal desired speeds exactly, so the server can treat `V_k` as known
  and constant, as Wen does.
- S2 feasibility is comfortable: the median number of in-coverage vehicles with at least
  13 s of remaining coverage is well above the 2 × C = 6 bar, so client selection is not
  starved by geometry.
- The kinematic choice must be disclosed in the thesis as a modelling limitation, and the
  Krauss evidence in `sumo/wen_closed_v1_krauss_run/` supports that discussion.
