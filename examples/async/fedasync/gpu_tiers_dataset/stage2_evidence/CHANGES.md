# Stage 2 — every edit, and why

Three changes. Paths are relative to `examples/async/fedasync/`.

**Nothing under `plato/` was touched.** All three live in `gpu_tiers_dataset/`,
verified with `find -newermt` at the end of this file.

---

## Change A — jitter lowered to a nominal CV of 15 %

### The edit

`gpu_tiers_dataset/make_configs.py:143` — `JITTER_SIGMA = 0.149166`, was `0.18`.

    sigma = sqrt(ln(1 + 0.15^2)) = 0.149166
    nominal CV = sqrt(exp(sigma^2) - 1) = 15.0000 %

`gpu_tiers_dataset/make_configs.py:131` — `PARTITION_SIZE` drops its `30` entry.
`gpu_tiers_dataset/make_configs.py:122` — `N_SLOT = {20: 0, 40: 2}`; slot 1 is left
unused rather than reassigned, so an old N = 30 config cannot collide with a new run.
`gpu_tiers_dataset/make_configs.py:119` — `TAG_SLOT` gains `_det3`, `_det4`.

### Why, and what it bought

At `sigma = 0.18` the orin and thor sleep ranges cleared each other by only **5 %**
in the seed-4126 draw. orin/thor is precisely the pair the `cpu` arm already cannot
separate, so an overlap there would shrink the `gpu` advantage under measurement
rather than reveal it.

Six fleets built — N ∈ {20, 40} × seeds {4126, 4127, 4128}. Worst orin/thor margin:

| | 4126 | 4127 | 4128 |
|---|---|---|---|
| **N = 20** | 1.166x | 1.539x | 1.382x |
| **N = 40** | 1.363x | 1.317x | **1.121x** |

All six clear the 1.10 floor; the worst clears it by 0.021. xavier/orin ranges from
4.202x to 5.337x. Full statistics — realised CV per tier under both conventions,
tier mean ratios, and the belief-vs-reality gap on `E[T]` — in
`jitter_check_cv15.txt`.

**Sigma is locked.** It was chosen on the separation margin, a property of the
fleet computable before any GPU time, and the comment at the constant says so.

---

## Change B — randomness belongs to the client, not the worker

### The edits

| file:line | what |
|---|---|
| `gpu_tiers_dataset/client_seeded_trainer.py` | new, 130 lines |
| `client_seeded_trainer.py:70` `derive_seed()` | `(base_seed, client_id, round)` → seed, via blake2b |
| `client_seeded_trainer.py:113` `ClientSeededTrainer.train_model()` | re-seeds, then delegates |
| `gpu_tiers_dataset/fedasync_seeded.py:140` | `import client_seeded_trainer` |
| `gpu_tiers_dataset/fedasync_seeded.py:214` | `trainer=client_seeded_trainer.ClientSeededTrainer` |

### Why

`stage1_evidence/determinism.txt` § 4: PLATO hands each round's client to whichever
of the `max_concurrency` workers is free (`plato/servers/base.py:662-682`), and
which one that is depends on the order the client processes connected over
socket.io — a startup race nothing seeds.

    worker    run det1        run det2
    2         client 14       client 15
    4         client 15       client 14

Stage 1 seeded torch once per **process**, so a client served by worker 2 in one
run and worker 4 in the other drew a different augmentation sequence, computed
different gradients, moved `G_i`, and the selector then ranked `V_i` differently.
`train_loss` diverged at round 3, the dispatch sequence at round 18.

This is not a change to the model being simulated. A worker process is an
implementation detail of the simulator; a client is not. Before this, the
parallelism strategy leaked into the experiment — the same federation at
`max_concurrency = 2` would have drawn different augmentations for the same clients
in the same rounds.

### Three details that are load-bearing

**`train_model`, not the `on_train_run_start` callback.** The DataLoader — and
therefore every augmentation draw — is created inside `train_model` *after* that
callback fires (`plato/trainers/composable.py:412-470`). Seeding in the callback
would sit upstream of nothing.

**blake2b, not `hash()`.** Python randomises `hash()` of a `str` per process via
`PYTHONHASHSEED`, which would reintroduce exactly the cross-process variation this
change removes. `tests/test_client_seeded_trainer.py::test_is_stable_across_processes`
runs the derivation under three different `PYTHONHASHSEED` values and asserts one
answer.

**A hash, not `base + client*k + round`.** Adjacent Mersenne Twister states are
correlated in their first draws, and a round here consumes only a few hundred.

**`seeding.py` still runs and is still needed** — it seeds each process at import
time, covering model construction and the numpy stream the selector samples `V_i`
from. This is a second, finer re-seed scoped to one client's one round.

### Evidence

`tests/test_client_seeded_trainer.py`, 22 tests. The one that matters:
`TestWorkerIndependence::test_two_different_workers_draw_identically` launches two
subprocesses with different `PLATO_CLIENT_ID`s and different prior RNG consumption,
seeds both for `(client 14, round 5)`, and asserts identical draws.

Confirmed live in the runs:

```
[ClientSeededTrainer] client_id=15 round=1 -> seed 1079283309
[ClientSeededTrainer] client_id=14 round=1 -> seed 1079772118
[ClientSeededTrainer] client_id=4  round=2 -> seed 2104229298
```

### What it did to two full runs — and what it did not

`determinism_after.txt`, two 200-round runs of one config, sequential:

| | before Change B | after Change B |
|---|---|---|
| `train_loss` identical to round | **3** | **38** |
| dispatch sequence identical to dispatch | **18** | **42** |
| worker assignment identical | no | no (and it no longer matters) |

**The runs are still not identical, and Change B is not what is stopping them.**
The worker assignment still shuffles — Change B does not stop the socket.io
connection race, it makes the race *harmless* — and the divergence, when it comes,
no longer starts there.

It starts at a **near-tie in the merge order**. At merge 38 the two runs merged
clients 10 and 20 in opposite orders; the three clients involved (#7, #10, #20)
finish within **0.592 s** of each other in reconstructed time, and one run merged
them in ascending order while the other did not. PLATO pops the smallest
`(finish_time, client_id)` off a heap and `finish_time` carries the accumulated
`processing_time` — ~0.065 s per report, accumulating *per slot*, so two slots
differ by several tenths of a second by round 39. A 0.592 s gap is inside that.

So **which client the server merges first is decided by the machine, not by the
experiment.** This is the residual Stage 1 predicted but could not see, because
the worker-assignment race dominated it; Change B removed the larger cause and
exposed this one.

**It cannot be fixed from the example folder.** The reordering happens inside
PLATO's heap on a value PLATO computes; the only fix is to keep `processing_time`
out of `finish_time`, which is an edit to `plato/servers/base.py` and out of scope.
Change C gives a time axis immune to it, not a merge order immune to it.

---

## Change C — the clock, reconstructed outside PLATO

### The edits

| file | what |
|---|---|
| `gpu_tiers_dataset/clock_reconstruction.py` | new, ~330 lines. The replay and the CSV. |
| `gpu_tiers_dataset/make_stage2_clock_report.py` | new. C1/C2/C3 with numbers. |
| `gpu_tiers_dataset/determinism_check.py:+` | `reconstructed_section()` and `--reconstructed` |

### Why

`stage1_evidence/determinism.txt` § 3: `processing_time` is a `perf_counter()`
delta measured on the host (`plato/clients/strategies/defaults.py:252`) and PLATO
adds it to every report's finish time (`plato/servers/base.py:1025-1030`), which
becomes the simulated clock. So the clock carries whatever else the machine was
doing, two runs on a loaded box are not comparable, and parallel runs contaminate
each other.

### The model

```
slots = [0.0] * C
for client in dispatch_order:           # FILE ORDER of the log's Selecting lines
    i        = argmin(slots)
    t_start  = slots[i]
    t_finish = t_start + sleep[client] + comm_time
    slots[i] = t_finish
```

Round R's clock is the finish time of the client PLATO merged at round R. Merges
are matched to dispatches by walking the log's merge sequence and taking each
client's earliest unconsumed dispatch — **not** by assuming the merge order is the
ascending finish order. It should be, and `verify_merge_order()` checks it; assuming
it would hide a staleness-bound flush, which merges several clients at once.

`comm_time` is read from `[clients] comm_simulation` (`comm_time_seconds()`), not
hard-coded. `true` → 0. `false` → **refused**: PLATO measures comm_time on the host
there, so there is no configured value to rebuild from. Any comm latency baked into
the sleep times by `build_fleet.py --comm-bandwidth-mbps` is already inside
`sleep_times.csv` and is deliberately not added twice.

### Output

`results/accuracy_curves_reconstructed.csv`:

```
experiment, arm, seed, round, sim_time_plato, sim_time_reconstructed, accuracy
```

Both clocks, as asked.

### Results

`clock_reconstruction.md` carries C1, C2 and C3 in full. The headline:

- **C1**: the gap is positive at every round (+3.46 s, +1.24 % at round 200) and
  correlates with the per-slot report count at **r = 0.945**, with a per-report
  slope of 0.059–0.075 s in every slot — which is the measured host overhead. It is
  **not globally monotone**, and it cannot be: `processing_time` accumulates *per
  slot*, and consecutive rounds merge from different slots. Within a slot it rises
  on 185 of 196 steps.
- **C2**: before Change B, reconstructing the clock does **not** reduce the
  run-to-run spread (0.8014 % → 0.8118 %). By the brief's own table that puts the
  cause in the **dispatch order**, not the host overhead — so Change B is
  load-bearing. After Change B the dispatch order still diverges (at dispatch 43
  rather than 18), so the reconstructed spread does not collapse either
  (2.1577 % → 2.1435 %). The spread moving from 0.80 % to 2.16 % is not a
  regression: once two runs diverge it is a sample of a noisy quantity, and the
  two pairs also ran on different fleets. The measurable gain is *where* they
  diverge.
- **C3**: the order is recoverable, and it is the **file order of the log lines**.
  The round number cannot carry it: round 1 contains C dispatches.

---

## Two things caught while doing this

**A port overflow.** Registering `_det3` and `_det4` pushed the tagged block past
the 16-bit ceiling (port 66001), because each tag owned 2000 ports and the offset
under a tag maxes at 913. The block is now 1000 wide.
`tests/test_make_configs.py::test_no_port_exceeds_the_16_bit_maximum` caught it.

**A silent cross-fleet reconstruction.** Regenerating `configs/` at the new sigma
meant `det1` — which ran on the sigma-0.18 fleet — was being reconstructed against
the sigma-0.1492 fleet's sleep times, producing entirely plausible numbers that
meant nothing. Two changes close it:

- `fedasync_seeded.py:save_config_used()` copies each run's TOML into its own
  output directory at startup, and `clock_reconstruction.config_for()` prefers
  that copy over the live `configs/` file.
- `clock_reconstruction.verify_fleet_matches_run()` cross-checks the sleep table
  against the `[GpuTierSleep] ... sleep_time=` lines each run wrote into its own
  client logs — the run's own record of the fleet it slept on, independent of any
  config file. All 20 clients must agree or the reconstruction refuses.

The guard was written after the mistake and then confirmed against it: it reports
"20 of 20 clients disagree" on the mismatched pair.

---

## Nothing under `plato/` was touched

```
$ find . -newermt "<stage 2 start>" -type f \( -name "*.py" -o -name "*.csv" \
      -o -name "*.toml" -o -name "*.md" \) -not -path "./gpu_tiers_dataset/*"
(no output)
```

The two Stage-1 edits outside the folder (`gpu_tiers/gpu_tier_sleep.py`,
`fedasync_polaris_vi_server.py`) were not revisited.
