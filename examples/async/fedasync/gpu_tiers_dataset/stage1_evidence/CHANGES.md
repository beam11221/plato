# Stage 1 — every edit, and why

Three changes were asked for. They are implemented, plus the apparatus needed to
run and check them. This file lists **every** edit; nothing else in the repository
was changed.

Paths are relative to `examples/async/fedasync/`.

---

## Summary: what was touched outside this folder

Exactly **two files**, both default-off — a config that does not carry the new key
behaves as it always did, byte for byte.

| file | edit | inert unless |
|---|---|---|
| `gpu_tiers/gpu_tier_sleep.py` | `oracle_sleeps()`, `_client_seconds()`, three call sites rerouted | `[gpu_tier] oracle_sleep_csv` is set |
| `fedasync_polaris_vi_server.py` | `gpu_oracle` added to the predictor set; `check_oracle_wiring()` | `staleness_predictor = "gpu_oracle"` |

`../surgical/`, `../separability/`, `../separability_n100/`, `../sumo_close_loop/`
and `gpu_tiers/` were **not** edited. None of their configs carries either new key,
so their runs are unaffected. Two tests assert this rather than assume it:

- `tests/test_gpu_oracle.py::TestDefaultOffIsByteIdentical::test_t_round_unchanged_without_the_key`
  — `T_round` at C=4 is still `2.080769 s`, the value `../separability/README.md`
  records the server must log.
- `tests/test_build_fleet.py::TestNoJitterReproducesOldGenerator` — with
  `--jitter-sigma 0` this folder's generator reproduces
  `../separability/output/client_sleep_times_xa15_n20.csv` and
  `../separability_n100/output/client_sleep_times_xa15_n100.csv` **byte for byte**.

---

## Change 1 — Seed torch

### Which RNGs were previously unseeded

| RNG | before | seeded by |
|---|---|---|
| python `random` | seeded, server process only, after startup | `plato/servers/base.py:374`, `[server] random_seed` |
| `numpy.random` | seeded, every process | `plato/config.py:456` inside `simulate_client_speed()`, `[clients] random_seed` |
| sampler `torch.Generator` | seeded | `plato/samplers/dirichlet.py:81`, `[data] random_seed` |
| **torch global CPU RNG** | **not seeded** | — |
| **torch CUDA RNG** | **not seeded** | — |
| **cuDNN algorithm choice** | **not pinned** | — |

The torch global RNG initialises ResNet-18's weights and drives the CIFAR-10 train
transform (`RandomHorizontalFlip` + `RandomCrop(32, 4)`,
`plato/datasources/torchvision.py:141-148`). Both ran free.

This is the measured cause of the 6.29 % run-to-run spread in
`../separability/PARALLELISM_CHECK.md`: different weights → different `G_i` →
different `V_i = G_i / sqrt(1 + tau_hat_i)` ranking → a different dispatch
sequence → a different simulated clock.

### Where the seeding now happens

**`gpu_tiers_dataset/seeding.py`** (new, 230 lines) — `seed_everything()` seeds
python `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`,
and sets `cudnn.deterministic = True` / `cudnn.benchmark = False`.

- `seeding.py:107` `config_path_from_argv()` — finds `-c/--config` the way
  `plato/config.py` does.
- `seeding.py:128` `read_seed()` — reads `[clients] random_seed` from the TOML
  **directly**, not via `Config()`. `Config()` has side effects at first call
  (argparse, base-path creation, a logging FileHandler, `simulate_client_speed()`)
  and provoking them early is a risk with no upside. A missing key is **fatal**,
  never defaulted.
- `seeding.py:152` `process_offset()` — `PLATO_CLIENT_ID`, which the server sets
  immediately before `proc.start()` and deletes immediately after
  (`plato/servers/base.py:493-516`). Server sees nothing → offset 0; client
  process *k* → offset *k*.
- **No second seed knob.** Everything is `[clients] random_seed + offset`.
  `tests/test_seeding.py::TestReadSeed::test_no_second_seed_knob_exists` greps the
  source to keep it that way.

**`gpu_tiers_dataset/fedasync_seeded.py:134`** — `SEED_RECORD = seeding.seed_everything()`
at **module level**, before any PLATO import.

Module level, because PLATO spawns client processes with the `spawn` start method
(`plato/servers/base.py:479`) and a spawned child re-imports the parent's
`__main__` as `__mp_main__` before unpickling its target. Module-level code
therefore runs in the server **and** in every client process, with no change to
PLATO's core. Verified directly before relying on it:

```
[module-level] pid=1024582 __name__=__main__      argv=['probe.py','-c','x.toml']
[module-level] pid=1024584 __name__=__mp_main__   argv=['/abs/probe.py','-c','x.toml']

[module-level] pid=1024588 PLATO_CLIENT_ID=None    <- the server
[module-level] pid=1024590 PLATO_CLIENT_ID='1'     <- client process 1
[module-level] pid=1024591 PLATO_CLIENT_ID='2'
```

Why a per-process offset rather than one seed everywhere: the same seed in all C
workers would be deterministic too, but all of them would then draw the identical
augmentation sequence — a correlation the runs do not currently have, and this
change is not meant to introduce one.

`torch.use_deterministic_algorithms(True)` is **not** on by default: it needs
`CUBLAS_WORKSPACE_CONFIG` and raises on ops with no deterministic kernel, which
would turn a nondeterminism question into a crash. `PLATO_DETERMINISTIC_ALGORITHMS=1`
enables it.

### What it achieved, measured — and what it did NOT

Two 200-round runs of one config, sequential and alone on the box:

| | unseeded (`../separability/PARALLELISM_CHECK.md`) | seeded (here) |
|---|---|---|
| `sim_time_s` spread at round 200 | **6.29 %** | **0.80 %** |

**The runs are still not identical.** `determinism.txt` says so in full; the
short version, because it matters before 34 runs are spent on top of it:

- `train_loss` diverges at **round 3**, the dispatch sequence at **round 18**.
- The cause is **not** the torch seeding, and not `processing_time`. It is that
  PLATO hands each round's client to whichever of the `max_concurrency` worker
  processes is free (`plato/servers/base.py:662-682`), and which worker that is
  depends on the order the four processes happened to connect over socket.io — a
  startup race nothing seeds. In run A worker 2 served client 14 and worker 4
  served client 15; in run B they were swapped.
- The torch RNG lives in the **worker process**, not in the logical client, so a
  client served by a different worker draws a different augmentation sequence,
  computes different gradients, moves `G_i`, moves the `V_i` ranking, and changes
  who is selected.

This is reported, not worked around, as the brief requires. The candidate fix —
re-seed torch at the start of each local training run from
`(base_seed, client_id, round)` so a client's augmentation belongs to the client
rather than to the worker — is **not applied**: it changes what the runs do, and
that is a decision to take deliberately rather than inside a stage meant to prove
three changes are right.

Separately and independently, `sim_time_s` can never be bit-identical: PLATO adds
the host's `processing_time` (`time.perf_counter()`) to every report's finish
time, and that finish time becomes the clock.

### New entry point rather than editing the old ones

`../fedasync_gpu_tier.py` and `../fedasync_gpu_tier_polaris_vi.py` are shared with
four folders holding finished results kept as the oracle branch. Seeding changes
what a run does, so putting it there would mean those results could no longer be
reproduced by re-running them, with nothing in a log to say why.
`fedasync_seeded.py` imports their pieces instead of copying them, and only configs
generated here point at it.

---

## Change 2 — Per-client jitter in the sleep generator

### The generator

**`gpu_tiers_dataset/build_fleet.py`** (new, ~560 lines). It replaces the two-step
`make_fleet.py` + `gpu_tiers/generate_sleep_times.py` sequence that
`../separability/build_inputs.sh` performs, and adds the jitter.

`gpu_tiers/generate_sleep_times.py` was **not** edited. It is now legacy: the
folders that call it keep producing exactly what they produced.

The formula, unchanged up to the anchor, with one new term:

```
raw(tier) = W_MODEL_FLOPS * partition_size * epochs / (tflops * 1e12 * eta)
scale     = anchor_sleep / raw(anchor_tier)
T_tier    = raw(tier) * scale
sleep_i   = T_tier(tier_i) * exp(eps_i) + comm_latency        <- new
eps_i     ~ Normal(0, sigma)
```

- `build_fleet.py:237` `draw_epsilons()` — drawn **once per client**, in ascending
  `client_id` order, from a dedicated `random.Random(seed)` instance so nothing
  else in the process can move the stream. `sigma == 0` short-circuits to all-zero
  **without touching the RNG**, which is what makes a no-jitter build byte-identical
  to the old generator for any seed.
- `build_fleet.py:263` `sleep_times()` — `comm_latency` is added **after** the
  jitter and is identical for every client; it is already in simulator units, so it
  is neither scaled nor jittered.
- The existing `noise_sigma` column in the tier CSVs is untouched and still 0. That
  mechanism was the wrong shape anyway: additive `(1 + N(0,sigma))`, redrawn on
  **every call**, i.e. transient load rather than a persistent per-unit offset.

Lognormal rather than additive Gaussian: sleep cannot go negative, and the
multiplicative form is what "efficiency is off by x %" means. The consequence is
recorded rather than corrected — `T_tier` becomes the tier's **median**, not its
mean. The brief specifies `sleep_i = T_tier * exp(eps_i)`; adding an
`exp(-sigma^2/2)` correction would quietly move the belief/reality gap that `gpu`
exists to measure.

### The sigma used, and the CV measured

**sigma = 0.18.** Chosen on the *nominal* CV of the distribution,
`sqrt(exp(sigma^2) - 1) = 18.15 %` — the middle of the 15-20 % target. That is a
property of sigma alone: it does not move with the draw and so cannot be tuned
after seeing a result. It was fixed before any fleet was built.

Realised CV, N = 20, seed 4126:

| tier | n | CV (population) | CV (sample) |
|---|---|---|---|
| xavier | 10 | 16.55 % | 17.45 % |
| orin | 7 | **14.90 %** | 16.10 % |
| thor | 3 | 17.25 % | **21.13 %** |

Both conventions are given because the tiers are small and the two differ by
`sqrt(n/(n-1))` — 22 % at n = 3.

**Two of the six numbers are outside 15-20 %, and no sigma puts them all inside.**
orin's population CV reaches 15 % only near sigma = 0.17 and thor's sample CV
passes 20 % at about the same point, so the window is roughly 0.001 wide. Chasing
it would be tuning sigma to make a statistic land somewhere. These are sampling
errors at small n, not a property of the jitter: the same sigma gives thor
13.68 / 14.99 % at N = 40, where it has 6 clients — the lowest of the three
instead of the highest. Full table in `jitter_check.txt` § 2.

### The separation requirement

Neither pair overlaps, at any of the three fleet sizes built:

| fleet | xavier / orin | orin / thor |
|---|---|---|
| N = 20 | separated, margin 4.935x | separated, **margin 1.051x** |
| N = 30 | separated, margin 4.157x | separated, margin 1.167x |
| N = 40 | separated, margin 4.242x | separated, margin 1.269x |

`margin = slower.min / faster.max`, so > 1 means the ranges do not touch. Reported
as a number rather than a boolean because 1.05 and 1.27 are both "separated" and
only one of them would survive a different draw.

**The binding constraint on sigma is orin/thor, not the CV target.** Those tiers
are only 1.92x apart before jitter (1.9231 s against 1.0 s) against 7.8x for
xavier/orin. At sigma = 0.20 they overlap in the N = 20 draw; at 0.18 they clear by
5.1 %.

Tier mean ratio, xavier:thor — **15.00x before jitter, 13.09x after**. Still an
order of magnitude.

### One consequence that must not be missed

`E[T]` moves **8.3231 s → 7.9180 s** (−4.87 %) for the N = 20 fleet. `T_round =
E[T]/C` follows it, so the `gpu` arm's belief is now wrong in its denominator as
well as its per-client numerator. `jitter_check.txt` § 6 carries `T_round` and
`tau_hat_xavier` for every C in the planned sweep.

---

## Change 3 — Split `gpu` into two arms

| arm | T-hat it holds | how |
|---|---|---|
| `gpu` | the tier mean from the datasheet tier file | unchanged code path |
| `gpu_oracle` | the per-client true sleep, jitter included | new |

`gpu` needed no edit: the generator now differs from it, which is the whole change.

### `gpu_tiers/gpu_tier_sleep.py`

- **`:205`** — `_oracle_sleeps` module global. `False` is the "resolved, and there
  is none" sentinel, kept distinct from `None` = "not resolved yet", so the absent
  case is not re-read from Config on every prediction.
- **`:416` `oracle_sleeps()`** — resolves `[gpu_tier] oracle_sleep_csv` once per
  process. Absent → `None` and every path below behaves as it always has. Present
  but unreadable → **raises**, following the `t_round_model()` precedent: a silent
  fallback would turn the upper-bound arm into an undetectable duplicate of `gpu`.
- **`:487` `_client_seconds()`** — the single place the oracle branch is taken.
- **`:506`, `:773`, `:842`, `:936`** — the three predictor-side entry points
  rerouted through it: `gpu_tops_predict_seconds`, `compute_pool_T_round_seconds`,
  `compute_pool_mean_T_seconds`.

All three, not just the first. The selector
(`fedasync_polaris_vi_selection.py:221-240`) computes busy-client residuals from
`gpu_tops_predict_seconds` and its denominator from the pool functions; an oracle
applied to one and not the others would be a half-oracle — true `t_client` against
a tier-mean `T_round` — whose behaviour no table could honestly describe. Routing
them through one function is also what let **`fedasync_polaris_vi_selection.py` stay
untouched**, as the brief requires.

The oracle value is the simulator's own sleep time: already in simulator units and
already carrying any comm latency, so it does **not** go through `predictor_scale()`
and C is **not** added to it again.

### `fedasync_polaris_vi_server.py`

- **`:56`** — `gpu_oracle` added to `STALENESS_PREDICTORS`.
- **`:60`** — `SECONDS_PREDICTORS = {"gpu_tops", "cpu", "gpu_oracle"}`, replacing
  the inline `in ("gpu_tops", "cpu")` test at the wiring site.
- **`:106` `check_oracle_wiring()`** — new. The key and the arm name are two halves
  of one switch written in two different TOML sections, and **either half alone
  produces a run that is silently a different arm**: `gpu_oracle` without the key
  becomes a second copy of `gpu`; `gpu_tops` with the key regains the oracle this
  stage exists to remove. Both are fatal. Neither is visible in an accuracy curve.

---

## Apparatus (new, all inside `gpu_tiers_dataset/`)

| file | role |
|---|---|
| `tiers/tiers_estimated_15x.csv` | copied from `../surgical/input/`, sha256 `323548f8f67e…` — identical |
| `tiers/tiers_cpu.csv` | copied from `../surgical/input/`, sha256 `045c265f4d8c…` — identical. It was duplicated byte-for-byte in `gpu_tiers/input/` and `surgical/input/`, with live configs pointing at **both** |
| `make_fleet.py` | moved from `../surgical/`, plus a remainder rule so N = 30 is buildable |
| `build_fleet.py` | the generator; one command → one immutable fleet directory |
| `fleets/n{20,30,40}_…/` | `clients.csv`, `sleep_times.csv`, `sleep_table.csv`, `manifest.json` |
| `seeding.py`, `fedasync_seeded.py` | Change 1 |
| `make_configs.py` | five arms, fleet directories, `staleness_bound` derived from C |
| `preflight.py` | see below |
| `make_stage1_evidence.py` | writes three of the five evidence files |
| `tests/` | **223 tests**, no GPU needed for most, ~35 s |

### Two preflight checks are the *opposite* of `../separability/preflight.py`

| that file | this file |
|---|---|
| FAIL if a tier has more than one sleep time (`preflight.py:114-117`) | FAIL if a tier has only one — the jitter did not apply |
| FAIL unless xavier sleeps exactly `anchor_sleep` (`:124`) | FAIL unless `T_tier[xavier]` is `anchor_sleep`; no client sleeps exactly 15 s any more |

Running the old checks against a jittered fleet would block every launch. Not
running the new ones would let an un-jittered fleet through — and an un-jittered
fleet silently puts the oracle back inside the `gpu` arm.

### N = 30 does not divide evenly — recorded, not hidden

50/35/15 over 30 clients is 15 / 10.5 / 4.5. `../surgical/make_fleet.py` refused
such a fleet, for a good reason it stated: a silently rounded fleet would differ
from the one named in every filename downstream. This version builds it — remainder
to the fastest tier, giving **15 / 10 / 5** — and removes the objection instead of
ignoring it, by naming the fleet after its **realised counts** (`n30_x15-o10-t5`).

The cost is real and is in the manifest: thor holds **16.7 %** of the fleet at
N = 30 against 15.0 % at N = 20 and N = 40, and thor is the fastest tier. A sweep
along N moves two things at once in that cell.

Also: 30 × 1667 = **50,010**, not 50,000. Kept as the brief specifies.

### `staleness_bound` is now derived from C

The planned sweep runs to C = 10, and a flat 25 would sit close to the routine
staleness of the xavier tier there — the aggregator would start censoring exactly
the tier under measurement (`gpu_tiers/HANDOFF.md` § 4.9). The rule is
`../separability_n100/make_configs.py`'s: 6.25 bound-units per slot, which
reproduces the historical 25 at C = 4 exactly, so no finished cell is disturbed.

C: 1 / 2 / 4 / 6 / 8 / 10 → bound 7 / 13 / 25 / 38 / 50 / 63.

---

## Two bugs the tests caught before a run could

1. **Infinite recursion in `gpu_tier_sleep.py`.** The edit that rerouted
   `gpu_tops_predict_seconds` through `_client_seconds` also matched the identical
   line *inside* `_client_seconds`, making it call itself.
   `tests/test_gpu_oracle.py` hit `RecursionError` immediately. In a sweep this
   would have surfaced as a crash somewhere into a 7-hour run.

2. **Every client process died at launch.** `fedasync_seeded.py` removed `--arm`
   from `sys.argv` before importing PLATO (necessary: `plato/servers/base.py:382`
   evaluates `Config()` at class-definition time, so argparse runs during the
   import). But spawn snapshots `sys.argv` at `proc.start()`, which happens much
   later — so the children re-imported the module, found no `--arm`, and exited.
   The server stayed up, correctly seeded, waiting forever for clients that never
   connected: **no traceback, no failure, just a hang.** Found by a 3-round smoke
   run, fixed by restoring argv after the imports
   (`fedasync_seeded.py:151`), and pinned by
   `tests/test_fedasync_seeded.py::TestArgvSurvivesToChildren`, which reproduces
   the broken version and asserts it fails.

---

## Nothing else changed

No edit to: the tier files' contents, the fleet mix, the dataset, the model, the
optimiser, `fedasync_polaris_vi_selection.py`, `fedasync_server.py`,
`fedasync_aggregation.py`, any file under `plato/`, or any file in
`../surgical/`, `../separability/`, `../separability_n100/`, `../sumo_close_loop/`
or `gpu_tiers/` other than the single `gpu_tier_sleep.py` edit listed above.
