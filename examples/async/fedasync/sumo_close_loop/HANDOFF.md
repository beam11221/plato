# HANDOFF — running the SUMO closed-loop experiment on a second machine

You are on the **second** machine. Machine A is already running two arms; your
job is the other two. Read this file to the end before running anything: several
settings look adjustable and are not.

**Your two arms: `baseline` and `observed`.**
Machine A runs `cpu` and `gpu`. Do not run those here — they are already running
there, and a duplicate is wasted GPU time, not a replication.

---

## 1. What the experiment is, in one page

Federated learning across a fleet of simulated vehicles on a 1200 m closed loop
with one roadside unit (RSU) at the centre. The RSU can only talk to a vehicle
while that vehicle is inside its 300 m radius, and it can only train a few
vehicles at a time, so on every round it has to **choose** which vehicles to ask.

To choose well it needs to guess how late each vehicle's answer will come — its
*staleness*. Four ways of guessing are being compared:

| arm | how it guesses staleness |
|---|---|
| `baseline` | it does not — random pick among vehicles in range |
| `gpu` | from the vehicle's GPU spec sheet (a correct spec) |
| `cpu` | from the vehicle's CPU clock (a spec that is **wrong on purpose**) |
| `observed` | from how late that vehicle was **last time** (Polaris's estimator) |

Everything except that one guess is held identical across the four arms — same
fleet, same seed, same data split, same optimiser. That is what makes the
comparison mean anything.

**What is new in this experiment** (compared with the earlier `../surgical/` one,
which had the same four arms but stationary clients):

1. Vehicles move, so the pool of selectable clients changes every second.
2. A vehicle that finishes training **outside** RSU range cannot upload. Its work
   is thrown away and counted as **waste-computing**. That waste number is the
   headline result — the next step of the project adds location prediction, and
   the thing it has to reduce is this number.

The selection formula is deliberately **unchanged** — `V_i = G_i / sqrt(1 + τ̂_i)`,
with only client speed and dataset quality in it. Mobility restricts *who may be
asked*, not *who looks best*. Do not add a mobility term; that is the next step.

---

## 2. Setup

### 2.1 Files — you are on the SAME disk as machine A

Both machines mount the same volume, so **everything is already here**. There is
nothing to clone, copy or unpack.

```bash
cd <the shared mount>/plato/examples/async/fedasync
ls sumo_close_loop/configs/     # baseline_n20.toml … observed_n20.toml
```

That convenience comes with one rule, and it is the most important line in this
file:

> ### DO NOT RUN `build_inputs.sh`.
>
> Machine A already built `input/`, `output/` and `configs/`, and you are looking
> at *those* files, not copies of them. Rebuilding would rewrite the inputs of a
> run that is currently executing on the other machine. Even though the generator
> is deterministic and would produce identical bytes, do not take the risk while
> a run is live.

Verify instead — it is read-only and it is the check that matters anyway:

```bash
python3.11 sumo_close_loop/preflight.py --cell n20
```

CIFAR-10 is already downloaded at `examples/async/fedasync/data/`, and because
the disk is shared both machines train on literally the same bytes. Nothing to do.

(`make_handoff_bundle.sh` exists for a future machine that is *not* on this mount.
You do not need it.)

### 2.2 Python

**The bundled `.venv` is broken. Do not try to use it.** On machine A the working
interpreter is the system `python3.11` with torch 2.4.1+cu124. Check yours:

```bash
python3.11 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If plato's imports fail, these are the packages machine A had to add on top of a
torch install:

```bash
python3.11 -m pip install munch tenseal safetensors datasets transformers \
    zstandard timm torch_optimizer scipy boto3 python-socketio aiohttp pytest
```

Export the interpreter so the runner picks it up:

```bash
export PYTHON=python3.11
```

### 2.3 Hardware needed

| | |
|---|---|
| GPU | one CUDA GPU. Two arms in parallel used **4.4 GB** on machine A. |
| RAM | ~8 GB per arm (1 server + 4 client processes at ~1.6 GB each). |
| Disk | **the binding constraint — see § 4.** |

---

## 3. Running

```bash
cd <plato>/examples/async/fedasync

# 1. build every input for the 20-vehicle fleet (a few seconds)
./sumo_close_loop/build_inputs.sh n20

# 2. check the inputs agree and print the numbers the run must reproduce
python3.11 sumo_close_loop/preflight.py --cell n20

# 3. a 200-round test build FIRST — about 20 minutes. Machine A already ran
#    this on all four arms and it passed, so a failure here is an ENVIRONMENT
#    problem on this box (Python, torch, GPU), not a code problem. The config
#    names carry the arm, so they cannot collide with machine A's.
python3.11 sumo_close_loop/make_sumo_configs.py --cell n20 --rounds 200 --tag _t200
PYTHON=python3.11 ./sumo_close_loop/run_sumo.sh baseline observed --cell n20 --tag _t200

# 4. the real thing
PYTHON=python3.11 ./sumo_close_loop/run_sumo.sh baseline observed --cell n20
```

`run_sumo.sh` runs the named arms **in parallel**, starts a checkpoint cleaner
per arm, and writes a `.done` marker so a re-run skips what already finished.
Run it under `screen` or `tmux` — it takes most of a day.

**Expected duration**, measured on machine A with two arms sharing one GPU:
200 rounds took 15-16 minutes, i.e. ~13 rounds/minute, so **10 000 rounds is
about 13 hours**, both arms finishing together. Each running arm held ~3.9 GB of
checkpoints with the cleaner at its defaults.

### After the 200-round test, before launching the real run

Confirm all five, from `sumo_close_loop/logs/`:

```bash
cd sumo_close_loop/logs
grep -h '\[SumoRsu\] Server:'          baseline_n20_t200.log observed_n20_t200.log
grep -h 'Staleness predictor:'         observed_n20_t200.log
grep -h '\[PolarisVi\] T_round ='      observed_n20_t200.log | head -1
grep -h '\[SumoRsu\] waste'            *_t200.log | tail -2
grep -c Traceback                      *_t200.log
```

| check | expected |
|---|---|
| server class | `SumoBaselineServer` for baseline, `SumoPolarisServer` for observed |
| predictor | `observed-τ (rounds, default)` — and **nothing** for baseline |
| `T_round` | **2.080769 s** exactly, for baseline and observed |
| waste line | present, `round=200 cycles=<about 250>` |
| xavier waste | 35-42 % (machine A measured 40.9 % baseline, 35.7 % observed) |
| tracebacks | **0** |

If `T_round` is not 2.080769, the inputs do not match machine A's and the two
halves of the experiment are not comparable. (The `cpu` arm on machine A logs
**3.620192 s** instead, and that is correct: its `T_round` is computed from the
CPU spec it believes, in which xavier and orin are indistinguishable. You are not
running that arm, so 2.080769 is the only value you should see.) Stop and compare
`sumo_close_loop/input/` and `output/` file by file.

---

## 4. Disk — the thing that will bite you

**The 200 GB allocation is shared between both machines, not 200 GB each.** All
four arms write into the same `runs/sumo_close_loop/`, so your usage and machine
A's add up. A full 10 000-round arm writes about **85 MB per dispatch cycle** and
would leave roughly **1.1 TB** behind if nothing deleted it. The cleaner is
therefore not optional — and if yours dies, you take machine A down with you.

`run_sumo.sh` starts `gpu_tiers/clean_checkpoints.sh` per arm and kills it when
the arm ends. With the defaults (`--keep-rounds 50 --clean-interval 3`) each
running arm holds ~10-15 GB.

**Watch it.** On 2026-08-13 an earlier sweep lost a run because the cleaner died
and the volume filled while the trainer hung with no error:

```bash
watch -n 300 'du -sh <plato>/examples/async/fedasync/runs/sumo_close_loop/*'
tail -f sumo_close_loop/logs/*.cleaner.log
```

If a cleaner dies mid-run you can restart just it, without touching the run:

```bash
gpu_tiers/clean_checkpoints.sh <run_dir>/checkpoints 50 3 12000 &
```

Do **not** clear old runs to make space without asking. `runs/` is shared and
holds the finished surgical experiment's results and logs, which are still needed
for analysis. Machine A already deleted the checkpoints from those (48 GB → 804 MB);
what is left is analysis data.

---

## 5. What a round is — read this before reading any result

The two layers are kept separate, and this is a deliberate design decision, not
an accident of the implementation.

* **Client layer.** A vehicle is dispatched, trains for its `T_i` seconds and
  reports. That happens whether or not the RSU can still hear it. A report from
  outside range is **waste**: the compute is spent, the slot frees, the vehicle
  becomes selectable again.
* **Federated layer.** A **round** is one aggregation. It exists only if at least
  one update actually reached the RSU. A dispatch cycle whose only candidate was
  lost leaves the round counter where it was — no weight update, no evaluation,
  no row in the results CSV.

So `rounds = 10000` means ten thousand real aggregations, needing about
`10000 / (1 - waste)` dispatch cycles. Both counters are logged:

```
[SumoRsu] waste round=200 cycles=253 total=... (…%) orin=… thor=… xavier=…
```

**The consequence to carry into the write-up.** Staleness is counted in rounds,
and rounds now arrive more slowly than reports do, by a factor of `1 - waste`.
The `gpu` and `cpu` arms predict `τ̂ = T_i / T_round` from a speed model that
knows nothing about mobility, so they should **over-predict** by roughly
`1 / (1 - waste)`. The `observed` arm measures τ directly and does not inherit
that bias. That asymmetry is a property of the round definition, **not** evidence
that one estimator is better. Report it as such.

---

## 6. Settings — none of these are yours to change

Inherited unchanged from `../surgical/` (`../docs/decisions/ADR-0003`), so the
two experiments differ only where they are meant to.

| key | value | why it is that |
|---|---|---|
| dataset / model | CIFAR-10 / ResNet-18 | |
| `total_clients` | 20 (10 xavier / 7 orin / 3 thor) | contiguous ID blocks |
| `partition_size` | 2500, Dirichlet α = 1.0 | not a disjoint partition |
| `per_round` / `max_concurrency` | 4 | must stay equal to each other |
| seed | 4126 | one seed, shared by every arm |
| `rounds` | 10000 | aggregations, not cycles |
| `target_accuracy` | 0.9 | unreachable in practice = no early stop |
| `staleness_bound` | 25 | inert here; xavier lands near 7.2 |
| `anchor_sleep` | **15** | changed from surgical's 100 — see below |
| `comm_simulation` | true | means "charge nothing for communication" |
| `avg_training_time` | 0.0 | must be present and zero |
| ports | 18100 / 18200 / 18300 / 18400 | baseline / gpu / cpu / observed |

**Treat the anchor as a physical parameter, not a tuning knob** (ADR-0004). With
mobility it fixes `T_train : coverage window` and so decides the waste rate. The
sleep times imply an effective GPU efficiency of **1.83 % of peak** — the tier
CSVs carry `eta_nominal = 1.0`, a placeholder, and the anchor absorbs the rest.
`preflight.py` prints this every run. If you are asked to change it, read
ADR-0004 first: the tier gap, `τ̂` and the waste rates all move with it, and
results from before and after are not poolable.

Derived values the run must reproduce: `E[T] = 8.3231 s`,
`T_round = 2.0808 s`, `τ̂ = 7.2089 / 0.9242 / 0.4806` for xavier / orin / thor.
`preflight.py` prints all of them.

The simulated clock consumes 1.70-2.26 s per round depending on the arm, so
10 000 rounds uses **47-63 % of the 35 991 s trace**. It fits with room to spare;
`preflight.py` section 6 re-checks this against your own inputs.

### Why `anchor_sleep` is 15 and not 100

The anchor scales every tier by the same factor, so `τ̂ = T_i / T_round` is
unchanged by it — at any anchor the four arms select identically. What it changes
is how the fleet lines up against SUMO's clock, which does *not* scale. Coverage
windows last ~34.7 s, so at the old anchor a xavier vehicle's 100 s of training
could never finish in range and the tier would vanish from the run. At 15 s:

| tier | T | predicted waste | measured on machine A (200 rounds) |
|---|---|---|---|
| xavier | 15.000 s | 43.5 % | 35.7-42.1 % |
| orin | 1.923 s | 5.6 % | 2.2-8.6 % |
| thor | 1.000 s | 2.9 % | 0.0-6.0 % |

---

## 7. Landmines

Everything in `../gpu_tiers/HANDOFF.md` § 4 still applies. These are the ones
specific to this experiment.

**The n100 cell uses ~12 GB per arm, and ~8 GB of that is not the cleaner's fault.**

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

**The filesystem is shared, so some files have two writers.** These are the ones
that matter:

| path | shared? | why it is safe / what to avoid |
|---|---|---|
| `input/`, `output/`, `configs/` | yes | read-only for you — see § 2.1, do not rebuild |
| `runs/sumo_close_loop/<arm>_n20/` | no collision | arm names differ; yours are `baseline_n20`, `observed_n20` |
| `logs/<arm>_n20.log`, `.done` | no collision | same reason |
| `data/` (CIFAR-10) | yes | read-only after the first download; already present |
| the 200 GB quota | **yes** | your usage and machine A's add up — see § 4 |
| `__pycache__/` | yes | harmless if both machines run the same Python; if yours differs, delete it rather than debugging a stale `.pyc` |

Ports do not collide because you are on a different machine, but keep the
assigned ones anyway (`baseline` 18100, `observed` 18400) so a log tells you which
arm it came from.

**The three input files are one fleet seen from three sides.**
`input/clients_n20.csv`, `input/coverage_n20.csv` and
`output/client_sleep_times_n20.csv` must be regenerated together;
`build_inputs.sh` does that. Crossing files from different builds is silent —
vehicles would sleep by one composition while the predictor believed another.

**Do not read SUMO's own `tier` column.** `surgical/input/tiers_*.csv` is the only
source of truth for hardware. `remap_fleet.py` drops SUMO's tier column on
purpose, and the trace's `t_bar_train_s` (marked provisional in the trace itself)
is never read.

**The trace's warm-up is excluded.** `trace.csv` starts at t = -150 s, where SUMO
is still spreading vehicles out and the mean speed has not settled.
`remap_fleet.py --start-time 0` drops coverage before t = 0 and clips a window
straddling it. Training starts at t = 0.

**The simulated clock, not the real one, is what maps onto the trace.** Three
clocks get confused here: real GPU time (~17 h), PLATO's simulated clock
(`wall_time`), and trace time (0 … 35 991 s). The mapping is
`trace_t = wall_time - initial_wall_time`. Preflight § 6 checks the run fits
inside the trace; if it ever runs past the end, the oracle wraps and logs a
warning — sound for a closed loop in steady state, but it should be a decision.

**Do not delete another machine's run directory or `.done` marker.** They live
next to yours under the same `runs/` and `logs/`, and `run_sumo.sh` treats a
`.done` marker as "this arm is finished". Removing machine A's would make a later
re-run redo six hours of finished work.

**A run interrupted mid-way leaves no `.done` marker**, so re-running the same
command resumes from PLATO's own checkpoint. That is intended. Deleting the run
directory starts it over.

**Do not "fix" the exploration bonus.** A vehicle whose uploads keep failing is
never marked as explored, so it keeps its 10× `G_i` proxy and keeps being picked.
That is honest — the server genuinely learned nothing from it — but it is a
feedback loop that did not exist in the surgical experiment. Watch for it in the
logs; do not change it.

---

## 8. When it is done

**Nothing to send.** The disk is shared, so your results are already where the
analysis will look for them:

```
sumo_close_loop/logs/baseline_n20.log     per-report upload verdicts, staleness records
sumo_close_loop/logs/observed_n20.log
runs/sumo_close_loop/baseline_n20/results/*.csv   accuracy vs round and elapsed time
runs/sumo_close_loop/observed_n20/results/*.csv
```

Both halves are needed — the analysis joins the logs to the CSVs.

**Leave the checkpoints alone when you finish.** They are ~4 GB per arm and the
cleaner stops with the run; machine A will clear them once the results are
confirmed. Deleting them yourself is fine only if you have checked that
`results/` and the `.log` files are intact first.

Do report back with a one-line sanity check, because a run that finished is not
the same as a run that worked:

```bash
python3.11 sumo_close_loop/parse_waste_log.py sumo_close_loop/logs/observed_n20.log
```

It prints dispatched / wasted per tier. Machine A measured **35.7 % xavier waste
on the observed arm** over 200 rounds. If yours is nowhere near 35-42 %, say so
rather than letting it pass quietly.

---

## 9. If you get stuck

* `python3.11 -m pytest sumo_close_loop/tests -q` — 90 tests, all should pass.
  They need no GPU and take ten seconds. Run them first when anything is odd.
* `sumo_close_loop/README.md` — the design, and why each piece is the way it is.
* `../surgical/BRIEF_surgical.md` — the four arms and where they came from.
* `../gpu_tiers/HANDOFF.md` § 4 — the landmine list this one builds on.
