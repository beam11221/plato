# SUMO closed-loop FedAsync

The same four client-selection arms as the surgical experiment, but the vehicles
now move. An RSU can only reach a vehicle while it is inside its range, so the
selector no longer has the whole fleet available at every moment, and a vehicle
that drives out of range before its training finishes has nowhere to upload to.

**This step adds mobility and nothing else.** The score is unchanged --
`V_i = G_i / sqrt(1 + tau_hat_i)`, with only client speed and dataset quality in
it. Location *prediction* is the next step; the number it has to improve is the
waste rate this step measures.

## The four arms

Identical to the surgical experiment (`../surgical/BRIEF_surgical.md`,
`../docs/decisions/ADR-0002`). One key decides which arm runs:

| arm | `[server] staleness_predictor` | believes |
|---|---|---|
| `baseline` | *absent* | nothing -- random among vehicles in range |
| `gpu` | `"gpu_tops"` | the GPU spec (correct) |
| `cpu` | `"cpu"` | the CPU clock (wrong on purpose) |
| `observed` | `"observed"` | the client's own last staleness (Polaris's estimator) |

`gpu` and `cpu` run the same code; they differ only in `[gpu_tier] tiers_csv`.

## The scenario

A 1200 m closed loop with one RSU at x = 600 m and a 300 m radius, so the
coverage zone is the middle 600 m. Traces are 10 hours at 1 s resolution.

| | n20 | n100 |
|---|---|---|
| vehicles | 20 | 100 |
| in range at any moment | 9.8 on average (49 %) | 34.5 (34.5 %) |
| coverage window | 34.7 s mean | 38.2 s mean |
| lap period | 70.7 s | 110.6 s |
| moments with nobody in range | 3 of 36 150 s | 67 of 36 150 s |

Residence is the same for every tier -- hardware affects how fast a vehicle
trains, not how fast it drives -- so mobility is orthogonal to the speed axis the
surgical experiment measured.

## `anchor_sleep = 15`, changed from the surgical 100

The anchor scales every tier by the same factor, so `tau_hat = T_i / T_round` is
unchanged by it: at any anchor the four arms select identically, and every
round-indexed result from the surgical experiment still applies. What it does
change is how the fleet lines up against SUMO's clock, which does not scale.

| tier | T at anchor 15 | T at anchor 100 | waste at 15 (n20) |
|---|---|---|---|
| xavier | 15.000 s | 100 s | 43.5 % |
| orin | 1.923 s | 12.8 s | 5.6 % |
| thor | 1.000 s | 6.7 s | 2.9 % |

At anchor 100 a xavier vehicle could never finish inside a 34.7 s window and the
tier would vanish from the run. At 15 it wastes about 43 % of its dispatches --
enough to measure and to improve on, with every tier still contributing.

The lower bound on the anchor is the host's own `processing_time`, which is real
and does not scale: it is an additive constant, so a smaller anchor compresses
the effective tier gap. At 15 s the gap is 14.6-15.0x against a nominal 15x.

**The anchor is not a free knob here, and it is not a noise-floor hack** (ADR-0004).
Coverage windows are in trace seconds and the simulated clock maps onto them 1:1,
so the anchor fixes the ratio `T_train : coverage window` and therefore decides
the waste rate -- the headline result. What makes 1:1 the right mapping is that
the *un*-anchored number is the unphysical one: `tiers.csv` sets
`eta_nominal = 1.0`, i.e. 100 % of peak FP16, so the raw 275 ms for xavier is a
theoretical bound no hardware reaches. The lab GPU -- much faster than a Jetson
Xavier -- measured 4.06-4.89 s for the same workload, logged every round as
`[SleepSimulation] measured_training_time=`.

So the anchor is a calibration of *effective utilisation*: 15 s implies **1.83 %
of peak**, and "a Jetson Xavier takes 15 s per epoch over 2500 CIFAR-10 images"
is the claim these runs make. `preflight.py` restates it on every run. When real
per-tier efficiencies exist, set `eta_nominal` in `tiers.csv` and drop the anchor
-- same arithmetic, self-documenting -- but re-derive the tier gap, `tau_hat` and
the waste rates, because all three move. ADR-0004 has the migration notes.

`tau_hat` is therefore identical to the surgical experiment's: **7.2089 / 0.9242
/ 0.4806**, and `T_round = 2.0808 s`. `preflight.py` prints these, and the server
log reproduces them -- if they disagree, something is wrong with the inputs.

## What a round is

The two layers are kept separate.

* **Client layer.** A vehicle is dispatched, trains for `T_i` seconds and
  reports. That happens whether or not the RSU can still hear it. A report that
  arrives from outside range is **waste**: the compute is spent, the slot is
  freed, and the vehicle is available again.
* **Federated layer.** A **round** is one aggregation. It exists only when at
  least one update actually reached the RSU. A dispatch cycle whose only
  candidate was lost leaves `current_round` where it was -- no weight update, no
  evaluation, no row in the results CSV.

So `rounds = 10000` means ten thousand real aggregations, and the run needs
about `10000 / (1 - waste)` dispatch cycles to get there. Both counters are
logged: `[SumoRsu] waste round=... cycles=...`.

**The consequence to keep in view.** Staleness is measured in rounds
(`current_round - starting_round`), and rounds now arrive more slowly than
reports do -- by a factor of `1 - waste`. The two spec arms predict
`tau_hat = T_i / T_round` from a speed model that knows nothing about mobility,
so they should over-predict by roughly `1 / (1 - waste)`; the `observed` arm
measures tau directly and does not inherit that bias. That is a real asymmetry
between the arms which did not exist in the surgical experiment, and it must be
reported as a property of the round definition, not read as estimator quality.
Measure it in the pilot before drawing any conclusion from MAE.

## Files

| | |
|---|---|
| `remap_fleet.py` | maps SUMO vehicles onto client IDs, tier-preserving |
| `sumo_mobility.py` | `CoverageOracle` -- is client i in range at time t? |
| `sumo_gates.py` | the two places coverage changes FedAsync's behaviour |
| `sumo_server.py` | the waiting clock, the upload verdict, the waste counter |
| `sumo_fedasync.py` | entry point for all four arms |
| `make_sumo_configs.py` | renders the four configs; asserts they differ in 4 keys |
| `preflight.py` | checks the inputs agree and prints the expected numbers |
| `parse_waste_log.py` | server log -> per-report CSV + waste tally |
| `build_inputs.sh` | runs everything above in order |
| `run_sumo.sh` | runs arms in parallel, one checkpoint cleaner each |
| `make_handoff_bundle.sh` | packs the untracked files for a second machine |
| `HANDOFF.md` | what the second machine needs in order to run its two arms |

## Running

```bash
cd examples/async/fedasync
./sumo_close_loop/build_inputs.sh n20
python3 sumo_close_loop/preflight.py --cell n20

PYTHONPATH=gpu_tiers:.:sumo_close_loop \
    python3 sumo_close_loop/sumo_fedasync.py \
    -c sumo_close_loop/configs/gpu_n20.toml
```

Or, with the checkpoint cleaner and both arms of this machine at once:

```bash
PYTHON=python3.11 ./sumo_close_loop/run_sumo.sh cpu gpu --cell n20
```

Two arms in parallel used 4.4 GB of GPU memory and ran at ~13 rounds/minute on
the reference box, so 10 000 rounds takes about 13 hours. **The disk allocation
is 200 GB and an uncleaned arm would write ~1.1 TB**, so `run_sumo.sh` starts
`gpu_tiers/clean_checkpoints.sh` per arm; that holds each at ~3.9 GB.

Tests: `python3 -m pytest sumo_close_loop/tests -q`

## What to grep for

```
[SumoRsu] report ...        one line per client report, with the upload verdict
[SumoRsu] waste ...         cumulative waste per tier
[SumoRsu] ... in RSU range  how many vehicles the selector could choose from
[SumoRsu] ... idling        the clock waiting for a vehicle to arrive
[PolarisVi] Staleness prediction accuracy   predicted vs actual tau, per report
[GpuTierSleep] round=...    per-dispatch tier labels, for dispatch-share analysis
```

## Landmines

Everything in `../gpu_tiers/HANDOFF.md` section 4 still applies. These are new.

### The pipeline must be refilled to its full depth, not by the requested count

PLATO refills with exactly as many clients as it just merged
(`servers/base.py:591`). Any round where coverage blocks a refill would lose that
slot for the rest of the run. `RsuSelectionGate._open_slots` asks for the deficit
instead: `clients_per_round - training_clients - reported_clients`.

Both subtractions matter. Under `simulate_wall_time` a report arrives long before
its *simulated* finish time, so counting only `training_clients` calls a vehicle
that is still training idle and dispatches it again. The first version of this
code did exactly that and ran the pipeline at 4 dispatches per merge instead of
1 -- concurrency is the denominator of every staleness in the experiment, so that
would have silently changed what the run measures. Pinned by
`tests/test_sumo_gates.py`.

### A lost report must not update the selector

`RsuSelectionGate.on_reports_received` filters out reports that never reached the
RSU. This matters most for the `observed` arm, whose entire estimator is the last
staleness a client reported: letting a lost report through would give the server
knowledge it never actually received.

### The clock has to be able to wait

With `simulate_wall_time = true` the server only advances when a client report
arrives. If no vehicle is in range and none is training, no report will ever
arrive. `SumoRsuMixin._advance_clock_to_coverage` steps the clock to the next
entry time instead. Rare (3 seconds out of 36 150 on n20) but a hang costs a run.

### The n100 cell uses ~12 GB per arm, and ~8 GB of that is not the cleaner's fault

`comm_simulation = true` makes the server hand each dispatched client its model
through a file, `checkpoints/resnet_18_<client_id>.pkl`, with a matching one for
the reply. There is **one pair per client ID**, so they scale with fleet size, not
with rounds:

| cell | payload files | checkpoints kept | total per arm |
|---|---|---|---|
| n20 | 20 ids -> ~1.7 GB | ~2-4 GB | ~4 GB |
| n100 | 100 ids -> **8.3 GB** | ~2-4 GB | **~12 GB** |

`clean_checkpoints.sh` deliberately leaves them alone: it only recognises
round-numbered files, and a payload for a client currently in flight is live data.
Deleting one under a running client would break that round.

This is bounded, not a leak -- the ceiling is `fleet x 2 x 42.7 MB`. Four n100 arms
across both machines settle around **50 GB of the 200 GB allocation**. Do not
"fix" the cleaner to delete them; the saving is not needed and the risk is real.

### A wasted cycle must still refill the pipeline

`_process_reports` returns early when nothing was delivered, but `_select_clients`
still runs and still dispatches a replacement. Skipping the refill as well would
drain concurrency exactly as in the section above.

### The final tally cannot live in `wrap_up()`

PLATO's `wrap_up()` calls `_close()`, which ends in `os._exit(0)`
(`servers/base.py:1487`), so nothing after the `super()` call runs. The summary
is logged from `server_will_close()`.

### `comm_simulation` must stay true

The upload verdict recomputes the finish time from the report, and needs
`comm_time` to be a known zero. `sumo_server.configure()` raises otherwise.

### Do not read SUMO's own `tier` column

`tiers.csv` is the single source of truth for hardware. `remap_fleet.py` drops
SUMO's tier column from the coverage table on purpose, and `config.yaml`'s
`t_bar_train_s` (marked provisional in the trace itself) is never read.

### The three input files are one fleet seen from three sides

`clients_<cell>.csv`, `coverage_<cell>.csv` and
`output/client_sleep_times_<cell>.csv` must be regenerated together. Crossing
files from different cells is silent -- vehicles would sleep by one composition
while the predictor believed another. `build_inputs.sh` regenerates all three;
`preflight.py` checks they agree.
