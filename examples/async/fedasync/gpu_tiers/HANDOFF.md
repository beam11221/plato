# Gap-sweep handoff

Everything needed to set this experiment up on a fresh machine and run it, plus
the findings that are not visible from the code alone. Written for an agent or
collaborator starting with no context.

**Read `## Landmines` before changing any config.** Several of the settings look
arbitrary and are not.

---

## 1. What the experiment is

100 simulated vehicles split across three GPU tiers (xavier 50 / orin 35 /
thor 15). Each client's training time is faked by a per-client sleep time that
stands in for its tier's speed. Sweeping the tier speed **gap G** asks: as
hardware heterogeneity widens, does the selection algorithm shift its priority
from data quality (`G_i`) toward training speed (`τ̂`)?

Three arms, five gaps each = **15 runs**:

| arm | entry point | what the server believes | ports |
|-----|-------------|--------------------------|-------|
| `gpu` | `fedasync_gpu_tier_polaris_vi.py` | GPU TFLOPS — belief matches reality | 8100+G |
| `cpu` | `fedasync_gpu_tier_polaris_vi.py` | CPU GHz — belief is *wrong* on purpose | 8200+G |
| `baseline` | `fedasync_gpu_tier.py` | nothing — random selection, no predictor | 8300+G |

Gaps: `01 10 20 30 40`. The `gpu` vs `cpu` pair is the core comparison: at the
same gap they use the **same sleep-time file**, so actual training times are
identical and any difference comes purely from what the selector believed.

Two paths that never meet in code, except at one line:

```
BELIEF (drives selection)              REALITY (drives timing)
[gpu_tier] tiers_csv                   [clients.simulation_distribution] csv_path
      |                                        |
  tier_for(cid) -> TFLOPS                Config.client_sleep_times
      |                                        |
  gpu_tops_predict_seconds()             reported as training_time
      |                                        |
  tau_hat -> V_i = G_i/sqrt(1+tau_hat)   server's simulated wall clock
```

The one place they meet is `residual = max(0, t_full - (wall_now - start_time))`
in `fedasync_polaris_vi_selection.py` — which is why anchoring matters (§ Landmines).

---

## 2. Setup on a new machine

### 2.1 Files that must come across

The whole `plato/` checkout, plus:

- `examples/async/fedasync/data/cifar-10-batches-py/` — copy it. Without it every
  run re-downloads 170 MB.
- `examples/async/fedasync/gpu_tiers/input/` and `output/` — the tier specs and
  per-client sleep CSVs. These are inputs, not outputs of a run.

### 2.2 Python environment

**The bundled `.venv` has a broken torch** — `torch/lib/` is missing entirely and
there is no dist-info, so `import torch` dies with
`libtorch_global_deps.so: cannot open shared object file`. Either repair it
(`uv sync --reinstall-package torch`) or use the system interpreter, which is what
`run_sweep.sh` defaults to (`PYTHON=python3`).

For a system interpreter with a working torch 2.4.x:

```bash
pip install munch python-socketio aiohttp scikit-learn pandas scipy tqdm \
    matplotlib requests datasets "transformers<4.46" timm torch_optimizer \
    zstandard boto3 tenseal
```

`transformers` must be pinned: newer releases import `torch.distributed.tensor.DTensor`,
which torch 2.4 does not have.

### 2.3 Generate the configs

```bash
cd examples/async/fedasync/gpu_tiers
python3 make_configs.py                      # same path layout as here
python3 make_configs.py --root /new/path/to/examples/async/fedasync   # different layout
```

Every path inside a rendered config is absolute, so `--root` is how a machine with
a different layout gets usable configs. `make_configs.py` refuses to write unless
all 15 parse as TOML, every referenced input exists, ports are unique, and the
three arms differ only in the four keys they are allowed to differ in.

`configs/` is **generated output — never hand-edit it.** It is deleted and
rewritten on every run of the script. Edit `templates/` instead.

### 2.4 Run

```bash
cd examples/async/fedasync/gpu_tiers
./run_sweep.sh gpu --dry-run          # look before leaping
screen -S sweep -dm ./run_sweep.sh gpu cpu baseline
```

`run_sweep.sh` runs the named arms in order, all five gaps each, and starts a
checkpoint cleaner per run that dies with that run. It skips runs that already
have a `.done` marker in `logs/`, so it is safe to re-run after an interruption.

---

## 3. Sizing — RAM is the constraint, not the GPU

Measured on one live arm (`max_concurrency = 10`, ResNet-18, partition 500, batch 64):

| resource | per arm | note |
|---|---|---|
| **RAM (RSS)** | **17.1 GB** | server 2.6 GB + 10 clients x 1.72 GB |
| RAM incl. page cache | ~23.5 GB | cache is reclaimable but counts in cgroup v1 |
| GPU memory | 3.5 GB | 11 processes x ~314 MB |
| GPU utilisation | **31 %** avg (peaks 74 %, idles at 0 %) | |
| CPU | ~6 of 48 cores | |

Scaling:

```
RAM  ~ n_arms * (2.6 + max_concurrency * 1.72) GB + 4 GB
GPU  ~ n_arms * (max_concurrency + 1) * 0.32 GB
```

To run all **three arms in parallel**: **64 GB RAM minimum** (96 GB comfortable),
**16 GB GPU minimum** (24 GB comfortable), 24+ CPU cores, 200 GB disk.

Do not size RAM to the exact requirement. Two arms needing 34 GB on a 29.5 GB
cgroup — only 15 % short — was enough for the OOM killer to take out 7 runs in a
row. PLATO respawns its client subprocesses every round, so there are moments
where two batches overlap.

Because GPU utilisation is only 31 %, three parallel arms push it to ~93 % and
each arm slows by roughly 20-40 %. Still far better than serialising: ~18-21 h for
all 15 runs versus ~45 h one arm at a time.

### Reading memory correctly

`free -g` reports the **host**, which is meaningless inside a container. The limit
that actually applies:

```bash
cat /sys/fs/cgroup/memory/memory.limit_in_bytes   # v1
cat /sys/fs/cgroup/memory.max                     # v2
cat /sys/fs/cgroup/memory/memory.failcnt          # times the ceiling was hit
grep '^oom_kill ' /sys/fs/cgroup/memory/memory.oom_control
```

`run_sweep.sh` reads these itself and refuses a run that will not fit, rather than
letting the OOM killer decide 70 s in. Override with `--skip-mem-check` if you
disagree with its estimate.

**`exit=137` in the runner log means SIGKILL, which here is almost always the
cgroup OOM killer.** Check `failcnt` and `oom_kill`; if they climbed during the
run, that was it.

---

## 4. Landmines

Each of these was a real bug or a real trap. Do not "clean up" any of them
without reading the reason.

### 4.1 `[gpu_tier] anchor_tier` / `anchor_sleep` must match the sleep CSV

The sleep CSVs are generated anchored (xavier = 20 s) but `gpu_tops_predict_seconds()`
computes the raw formula (xavier = 0.055 s) — a **363x** scale difference.

Ratios cancel in `τ̂ = T_pred / T_round`, so this was invisible there. But
`select_clients` also does `residual = max(0, t_full - (wall_now - start_time))`,
mixing predicted seconds with elapsed simulated wall time. Un-anchored:

| arm | raw prediction | reality | effect on `residual` |
|---|---|---|---|
| `gpu` | 0.055 s | 20 s | collapses to 0 — Phase 1 never fires |
| `cpu` | 275 s | 20 s | saturates — Phase 1 always fires, on garbage |

The two arms then produce τ̂ on different magnitudes (gpu {0,1} vs cpu {2,9}),
i.e. different effective strengths of the `1/sqrt(1+τ̂)` discount — a confound
sitting directly on the effect the sweep exists to measure.

`predictor_scale()` in `gpu_tier_sleep.py` applies the anchor to **all three**
predictor-side call sites. Anchoring only some of them would desynchronise
`T_pred` from `T_round`. Omitting both keys gives scale 1.0 (old behaviour).

Verify with `python3 test_predictor_anchor.py` — 16 checks, including that Phase 1
fires only when anchored.

### 4.2 `polaris_*` keys must sit ABOVE `[server.staleness_weighting_function]`

In TOML any bare key after a sub-table header belongs to that sub-table. The
original `configs_backup/` configs put `polaris_alpha` and friends below it, so
they landed in `server.staleness_weighting_function` and
`getattr(server_cfg, "polaris_alpha", 10.0)` silently used its defaults. The
defaults happened to equal the written values, so nothing looked wrong — until
someone tried to change one.

### 4.3 `sleep_simulation = true` requires `avg_training_time`

`plato/clients/strategies/defaults.py:356` reads `Config().clients.avg_training_time`
**with no guard**. Omit it and the first client report raises AttributeError.

With `sleep_simulation = true` the client never actually sleeps
(`plato/trainers/composable.py:392` returns early) but still *reports*
`training_time = (avg_training_time + sleep) * epochs`. The simulated clock sees
the full 20 / 4.47 / 1.0 s spread while the run stops paying for it in real time —
measured 3.75x less wall clock, simulated clock within 4 %, identical staleness
predictions (`smoke_test_sleep_simulation.py`).

`avg_training_time = 1.5` is **calibrated**, not arbitrary: it is the real
ResNet-18 compute per client per round on the machine this was tuned on, and it
keeps the simulated clock matching the older real-sleep runs. Re-measure with the
smoke test if the model, `partition_size`, or GPU changes. `0.0` is a defensible
alternative — it makes the tier sleep the sole determinant of training time.

### 4.4 The anchor itself is load-bearing

PLATO charges each client an additive floor of `c ≈ 2.0 s` (payload processing +
42.69 MB each way), so the heterogeneity actually experienced is
`(c + s_xavier) / (c + s_thor)`, not `s_xavier / s_thor`. Without the 20 s anchor
the whole sweep would span an effective gap of **1.000 to 1.027** — 2.6 % across a
40x nominal range, i.e. no independent variable at all.

| G | anchored effective gap |
|---|---|
| 1 | 1.00 |
| 10 | 5.50 |
| 20 | 7.33 |
| 30 | 8.25 |
| 40 | 8.80 |

### 4.5 The CPU arm uses `--metric freq` (1.30x), deliberately

`tiers_cpu.csv` holds `cpu_freq_ghz / 1000`: xavier 0.002 / orin 0.002 / thor
0.0026. Xavier and Orin both clock at 2.00 GHz, so **the predictor cannot tell
them apart** — its estimates collapse to two values (20.0 s and 15.38 s) while
reality has three tiers. That mis-ranking is part of what this arm measures, on
top of the spread error. `make_cpu_tiers.py --metric cores_x_freq` would give
1.95x with three distinct tiers; the sweep does not use it.

Mismatch = effective_gap / 1.30, crossing 1 between G=1 and G=5, reaching 6.8x at
G=40 — so the arm covers both over- and under-estimation.

### 4.6 The three arms may differ in exactly four keys

`base_path`, `port`, `staleness_predictor`, `[gpu_tier]`. Everything else —
`partition_size`, `sampler`, `concentration`, `random_seed`, all optimizer
settings, `rounds`, `target_accuracy`, `staleness_bound`, `mixing_hyperparameter`,
`sleep_simulation`, `avg_training_time`, … — must be identical or the arms are not
comparable. `make_configs.py` enforces this across 27 shared keys and refuses to
write if any drift in.

This bites in practice: changing `concentration` in two templates and forgetting
the third means the baseline trains on a different data distribution.

### 4.7 `baseline` has no predictor at all

`fedasync_gpu_tier.py` uses `fedasync_server.Server`, whose selection is random-
skipping-busy. It never reads `staleness_predictor` — that key is read in exactly
one place in the whole repo, `fedasync_polaris_vi_server.py:287`. Adding it to the
baseline config would be a dead key implying an effect that does not exist.

`[gpu_tier] tiers_csv` is still required there, but only feeds the `tier_id` label
in log lines and the `speed_greedy` filter. No TFLOPS value reaches the selector,
so swapping it to `tiers_cpu.csv` would produce a byte-identical run — there is no
meaningful "CPU baseline" unless you change what the clients actually sleep.

### 4.8 `elapsed_time` is not a pure simulated clock — it carries the host

`servers/base.py:1026` sets

```
finish_time = training_time + processing_time + comm_time + start_time
              └ simulated ┘   └── measured on whatever machine ran it ──┘
```

and `base.py:1159` advances the simulated clock to that value. `training_time` is
synthetic under `sleep_simulation = true`, but `processing_time` and `comm_time`
are `time.time()` deltas for serialising and shipping 42.69 MB. **The arms did not
share a host** (§ "Arms are split across two pods"), so part of any arm-to-arm
difference in `elapsed_time` is a difference in machines, not selectors.

`recompute_wall_time.py` rebuilds the clock from the schedule in the logs and
measures the floor each host actually charged:

| arm | host floor per report (median) |
|---|---|
| `gpu` | 0.356 – 0.378 s |
| `cpu` | 0.375 – 0.387 s |
| `baseline` | 0.352 s |

Measured on 2026-08-10 across all 12 finished runs. The reconstruction reproduces
PLATO's own clock to a **median 0.01–0.07 % error** when charged each host's own
floor, which is what says the model is right.

So the contamination is real but small: the floor is 2–15 % of a report's cost
(tier sleeps are 2–21.5 s), and the *between-host* part is ~0.02 s. Re-running the
whole comparison on a uniform floor moves every cpu÷gpu time ratio by **at most
0.004**. It does not explain any result in the sweep — but it is the right axis to
report, and the two arms would not be comparable at all if the payload or the pods
were less alike. `plot_time_to_accuracy.py --time-source clean` uses it.

Two things this exposed, both worth knowing before writing analysis code:

- **`round - staleness` is not the dispatch round.** Matching each merge to the
  client's earliest unconsumed dispatch (FIFO, from the `[GpuTierSleep] round=`
  lines) is exact; staleness arithmetic disagrees by hundreds of rounds on
  re-selected clients and puts the implied floor at a median of **-2.5 s**. The
  two agree at the median (the field is fine for aggregate staleness) and diverge
  in the tail — the ~5 % of reports from stragglers left outstanding for a median
  of 98 rounds while reporting `staleness=1`.
- **`start_time` is the clock at the *start* of the dispatch round**, i.e.
  `clock[s-1]` (`base.py:534` sets `round_start_wall_time = wall_time` before
  dispatching). Using `clock[s]` charges that round's own aggregation to the
  client and the reconstruction diverges.

To remove the contamination at the source rather than after the fact, set
`comm_simulation = true` and rerun — the selector reads `wall_now`, so the
recorded choices were themselves made against the contaminated clock and no
post-hoc pass can undo that part.

### 4.9 The client pipeline used to drain — everything before 2026-08-10 is void

**Fixed on 2026-08-10 in `plato/servers/base.py`. Every run made before that date
is invalid for any time-based claim and should not be compared against runs made
after it.** The pre-fix `gpu_xa20_gap40` run is kept at
`../runs/gap_sweep_prebugfix/` as the reference for what the bug looked like.

`_process_clients` merged clients along two paths but recorded only one of them:

| path | merges into `updates` | records in `current_processed_clients` |
|---|---|---|
| Step 2, normal | yes | yes (`base.py:1156`) |
| Step 3, staleness flush | yes | **no** ← the bug |

`_select_clients` refills with
`choose_clients(selectable_clients, len(self.current_processed_clients))`
(`base.py:589`), so every client a flush merged was never replaced. With
`minimum_clients_aggregated = 1`, Step 2 contributes exactly one per round, which
is why the logs show one selection per round no matter how many clients merged —
`gpu_xa20_gap40` round 13 merges six and round 14 selects one.

Each flush shrinks the working set permanently. Measured in-flight concurrency,
against `max_concurrency = 10`:

| | G=1 | G=10 | G=20 | G=30 | G=40 |
|---|---|---|---|---|---|
| `gpu` | 8 | 4 | 2 | **1** | **1** |
| `cpu` | 7 | 4 | 4 | 2 | **3** |

At G=40 the `gpu` arm trained **one client at a time** for its last ~2 400 rounds.
Since simulated seconds per round is roughly a job's length over how many jobs
overlap, the wall-clock axis was measuring pipeline depth, not selection quality —
which is the whole of the "cpu reaches target accuracy sooner at wide gaps"
result. The round-based results are affected too: with one client in flight
instead of ten, the pool the selector chooses from is a different pool.

How it was found: the tier mix said `gpu` picks *faster* clients as the gap
widens, yet its clock advanced *more* per round. That contradiction only resolves
through concurrency, and reconstructing the schedule (`recompute_wall_time.py`)
showed in-flight stepping 9 → 4 → 3 → 1 at rounds 13, 64 and 262 — each drop
exactly (clients merged − 1).

The fix records the flushed clients, matching what Step 2 already did. Pinned by
`../tests/test_flush_refill.py`, which asserts every client merged into `updates`
appears in `current_processed_clients`; it fails on the old code with
`merged [1,2,3,4,5,6] but recorded [1]`.

The knock-on, and why `staleness_bound` moved 11 → 25
-----------------------------------------------------

With the pipeline actually full, a job's sojourn is ~10 rounds, so the old bound
of 11 sat right on the boundary and flushes became routine. Measured on the
verification runs (gpu, G=40, 800 rounds each, same seed):

| | flushes | C/M | xavier staleness | xavier over bound | sim. time @ r790 |
|---|---|---|---|---|---|
| bound 11 | 105 (13.1 %) | 6.0 | 11.4 | **68 %** | 1348 s |
| bound 25 | 17 (2.1 %) | 8.6 | 20.4 | 11 % | 774 s |

The bound was censoring the slowest tier and nothing else — orin and thor never
exceeded it in either run. That is not neutral between the arms (see the note in
`templates/*.tmpl`): the arm that steers harder toward fast clients lowers the
mean selected duration and inflates the staleness of every straggler it still
picks, so the bound would have penalised exactly the behaviour the sweep exists
to measure.

The model used to pick 25,

    staleness_i  ~  (C / M) * (T_i / T_mean_selected)

predicts within **4 %** once the bound stops truncating the distribution
(bound 25: xavier 20.4 measured vs 21.2 predicted; orin and thor within 5 %). At
bound 11 the same model is off by ~30 %, and that gap *is* the censoring. Note
the feedback loop when re-deriving it: raising the bound removes flushes, which
drops M toward 1, which raises C/M and hence the staleness the bound must clear —
6.0 → 8.6 here.

Accuracy per round is unchanged (0.5241 vs 0.5278 at round 790, inside run-to-run
noise) while simulated time nearly halves, so the staleness weighting absorbs the
extra staleness. Re-derive the bound if `max_concurrency`, the tier spread, or
`minimum_clients_aggregated` change.

### 4.10 Smaller traps

- **`staleness_predictor` typos used to fall back silently — fixed 2026-08-28.**
  It was read via `getattr(server_cfg, "staleness_predictor", "observed")` and
  only `"gpu_tops"` and `"cpu"` were recognised, so `"GPU_TOPS"` or `"gpu_top"`
  silently became `observed`. Now `fedasync_polaris_vi_server.
  resolve_staleness_predictor()` raises on an unrecognised value (an absent key
  still means `observed`, so old configs are unaffected), and
  `make_configs.py` refuses to write a config whose predictor does not match the
  arm in its own filename (`ARM_PREDICTOR`). Pinned by
  `test_predictor_strictness.py`. The reason it had to change: once `observed`
  became a real arm of a sweep (`surgical/`), a typo no longer produced a
  visibly wrong run but an undetectable **duplicate of another arm**. Confirming
  from the log is still the cheap habit:
  `grep '\[PolarisVi\] Staleness predictor:' <log>`.
- **`gpu_tops` and `cpu` take the identical code path.** The only difference is
  `[gpu_tier] tiers_csv`. The two names exist so the log records which arm ran.
- **`max_sleep_time` defaults to 60 s and clips silently** (`plato/config.py:462`).
  The templates set 120 so a 20 s anchor is safe; anchors above 60 would otherwise
  be truncated with no warning.
- **`data_path` is always joined onto `base_path`** (`plato/config.py:360-367`).
  Without an absolute path each run downloads its own copy of CIFAR-10.
- **`PYTHONPATH` must include `gpu_tiers/`.** `gpu_tier_sleep.py` lives there but
  `fedasync_gpu_tier.py`, `fedasync_gpu_tier_polaris_vi.py` and
  `fedasync_polaris_vi_server.py` all do a bare `import gpu_tier_sleep`.
  `run_sweep.sh` sets it; direct invocations must too.
- **Selection is stochastic V_i-weighted sampling, not top-K.** The class
  docstring in `fedasync_polaris_vi_selection.py` still says "Deterministic top-K";
  the code at lines 281-290 samples proportional to V_i. Top-K is only the
  fallback when the weights sum to zero.

### 4.11 `T_round` uses the wrong model, and the error tracks the sweep's x-axis

**Found 2026-08-11, mid-sweep. Not fixed in the runs currently in flight — the fix
exists behind `[gpu_tier] t_round_model` and defaults to the old behaviour on
purpose.** Flipping it invalidates every run made under the old one.

`T_round` is the denominator that turns a training time into a staleness in
rounds, `τ̂_i = T_i / T_round`. `compute_pool_T_round_seconds` computes it as
**E[min of P i.i.d. draws]** from the pool — the time to the first completion
when all P clients start at the same instant. That is a *cold start*. In steady
state the in-flight jobs have staggered start times, so the next merge arrives
much sooner than "min of P fresh draws" implies.

The right model is a **closed queue**: C slots are kept full, each refilled with a
freshly drawn client, so each slot completes one job per `E[T]` and the C slots
merge at `C/E[T]`:

```
T_round = E[T] / C          # E[T] is the ARITHMETIC mean
```

The harmonic mean would be right only if each slot were pinned to one fixed
client for the whole run; PLATO re-draws on every refill, so it is not — and
measured, harmonic is off by −72 % at G=30 where arithmetic is off by −7 %.

Against `d(elapsed_time)/d(round)` in the four finished post-fix gpu runs
(`gpu_tiers/../test_t_round.py::test_t_round_queue_matches_observed_rate` pins
these numbers):

| G | measured s/round | `"min"` | `"queue"` | observed staleness | predicted under `"min"` |
|---|---|---|---|---|---|
| 1  | 2.183 | **+816.1 %** | +1.8 % | 9.00 | 1.00 |
| 10 | 1.336 | +114.5 % | −3.3 % | 8.99 | 4.06 |
| 20 | 1.205 | +40.9 % | −5.0 % | 8.98 | 6.06 |
| 30 | 1.147 | +10.8 % | −7.4 % | 8.93 | 7.57 |

At G=1 every client takes the same 20 s, so "min of 10" **is** 20 s: the
denominator becomes a whole client duration and predicted staleness collapses to
1.0 against an observed 9.0. As the spread widens the fast tier starts to
dominate the minimum and the two models converge *by accident*.

**Why this is not a constant that cancels.** `V_i = G_i / sqrt(1 + τ̂_i)`, so an
under-estimated τ̂ weakens the speed term — and under `"min"` that weakening is
strongest at the narrow-gap end and vanishes at the wide-gap end. The effective
strength of the mechanism the sweep exists to measure therefore varies with the
sweep's own independent variable. Same shape of problem as the un-anchored
predictor in § 4.1, and it is the leading candidate for why prediction § 4b.5
came out backwards: the post-fix `gpu` arm steers *less* than the pre-fix one
(xavier 39.7 % at G=30 vs 35.8 % pre-fix) and far less than the ~29 % predicted.

Ranking within a run is untouched — every client is divided by the same
`T_round`, so `"min"` changes the strength of the discount, not who is preferred.
That is why the sweep still shows the right qualitative trend.

```toml
[gpu_tier]
t_round_model = "min"     # historical, default, wrong
t_round_model = "queue"   # correct
```

Absent key → `"min"`, so configs written before it existed are byte-identical
(verified: the kernel reproduces the `T_round = 1.270266` logged by
`gpu_xa20_gap30`). A typo raises rather than falling back — unlike
`staleness_predictor` (§ 4.10). `make_configs.py` checks it matches across the
predictor arms via `SHARED_GPU_TIER_KEYS`, because `[gpu_tier]` is otherwise
exempt from `SHARED_KEYS` and nothing else would catch the drift.

### 4.11b Decision: finish the sweep under the buggy `T_round`, fix it after

Taken 2026-08-11, with the sweep 4/15 runs in. **Do not flip `t_round_model` to
`"queue"` until all 15 runs are done.** The reasoning, and the evidence that has
to hold for it to stay valid — re-run `python3 analyze_steering.py` to check it
against the arms as they land.

The bias is in the *magnitude* of τ̂, not in the *ranking*: every client in a run
is divided by the same `T_round`, so it changes how hard the selector steers, not
who it prefers. Measured dispatch lift (share dispatched ÷ share of pool):

| G | xavier | orin | thor | |
|---|---|---|---|---|
| 1  | 1.01 | 1.03 | 0.91 | ← control: all tiers sleep 20 s, no speed signal |
| 10 | 0.88 | 1.11 | 1.14 | |
| 20 | 0.83 | 1.14 | 1.24 | |
| 30 | 0.81 | 1.15 | 1.29 | |

G=1 is a genuine control and it behaves: lift ≈ 1.00 for all three tiers, mean
`V_i` within 2 % across tiers (0.3128 / 0.3190 / 0.3189). From G=10 on, the
ordering `thor > orin > xavier` holds in both lift and mean `V_i` at every gap,
and strengthens monotonically. The mechanism works.

**The trend is not an artefact of the bug.** `G_i` does not depend on `T_round`,
so the whole speed preference between two tiers is
`sqrt(1+τ̂_slow)/sqrt(1+τ̂_fast)`. Under both models:

| G | pref `"min"` | pref `"queue"` | ratio |
|---|---|---|---|
| 1  | 1.00 | 1.00 | 1.00 |
| 10 | 2.17 | 2.56 | 1.18 |
| 20 | 2.84 | 3.12 | 1.10 |
| 30 | 3.31 | 3.42 | 1.03 |
| 40 | 3.70 | 3.61 | 0.98 |
| **cpu, any G** | **1.07** | **1.13** | 1.05 |

Same endpoints either way. `"queue"` steers harder in the middle, so the runs
made under `"min"` **understate** the method's own steering for G ≤ 30 — a
conservative bound. It crosses over at G=40 and overstates by 2 %, so "always a
lower bound" is not quite right; say "lower bound for G ≤ 30".

The `cpu` row is why the comparison still works: its belief is one file shared by
all five gaps, so its speed preference is **constant in G by construction** (1.07).
The sweep therefore has a rising treatment (`gpu`, 1.00 → 3.70) against two flat
controls (`cpu` at 1.07, `baseline` at 1.00 by definition), and the `T_round` bias
shifts the treatment curve without touching either control's flatness or the
treatment's endpoints.

What this decision does **not** license:

- Any claim about the *absolute* accuracy of the staleness estimate. The +816 %
  at G=1 is a property of this implementation, not of the idea.
- Any claim about the *curvature* of the advantage-vs-gap curve — where it
  saturates, whether it is super- or sub-linear. Endpoints are safe; the middle
  is damped by up to 18 %.
- Comparing a post-fix run against one made under `"queue"` later. Same rule as
  § 4.9: the model is part of the experiment's identity.

Expect saturation at the top end. `pref` goes 1.00 → 2.17 → 2.84 → 3.31 → 3.70
and the effective gap saturates the same way (1.0 / 5.5 / 7.3 / 8.25 / 8.8,
§ 4.4). A flat G=30→40 segment is the anchor design, not a null result.

The rerun under `"queue"` is then a clean improvement iteration with a
pre-registered prediction: steering rises ~18 % at G=10, ~10 % at G=20, ~3 % at
G=30, and G=1 and the `baseline` arm do not move at all.

### 4.12 Accuracy stopped reaching 0.60 after the § 4.9 fix — that is the fix working

The post-fix runs plateau at 0.5716–0.5996 and never trip `target_accuracy = 0.6`,
where pre-fix runs crossed it routinely. Nothing new is broken; the comparison
was never like-for-like:

| | mean staleness | acc @ r1000 | best |
|---|---|---|---|
| pre-fix `gpu_xa20_gap30` | ~1.2 (drained pipeline) | 0.5725 | 0.6020 (early-stopped) |
| post-fix `gpu_xa20_gap30` | 8.93 | 0.5249 | 0.5960 |

With the pipeline drained to 1–3 clients the run was nearly synchronous, every
update was fresh, and convergence per round was correspondingly good. Restoring
the configured depth of 9–10 raises mean staleness to ~9 by construction
(§ 4.9: mean staleness = C/M), and the polynomial staleness weighting at `a = 0.5`
then damps each update by `1/sqrt(1+9) = 0.32` instead of `1/sqrt(1+1.2) = 0.67`.
Slower per-round convergence is the price of the ~9× throughput, and it is paid
in exactly the place theory says.

**The real problem is that 0.60 sits on the plateau.** Every full-length run in
both sweeps tops out at 0.59–0.60, so whether a run "reaches the target" is
decided by noise: post-fix, three runs land at 0.596–0.600 and `gap20` at 0.572.
`target_accuracy` was lowered 0.7 → 0.6 on 2026-08-09 for the opposite reason —
0.7 was unreachable — and 0.6 is only marginally better placed.

Consequences:

- **Report time-to-accuracy at 0.55, not 0.60.** All four post-fix runs cross 0.55
  cleanly (`plot_time_to_accuracy.py --targets 45 50 55`); 0.60 is censored for
  some and not others, which makes the headline metric a coin flip.
- **`target_accuracy = 0.6` no longer early-stops anything.** Every post-fix run
  pays the full 10 000 rounds (~6 h), so the 15-run sweep costs ~90 h, not the
  ~45 h the 0.6 change was meant to buy back. Setting it to 0.55 would roughly
  halve the remaining sweep.
- `gap20` at 0.5716 is a genuine outlier against 0.596/0.600/0.596. One seed per
  cell (`random_seed = 8888` shared), so there is nothing to separate a real
  effect from run-to-run variance. Worth a second seed before it goes in a paper.

---

### 4.13 First-crossing times carry ±10 % of evaluation noise, and 0.60 is past the knee

Measured 2026-08-12 on all nine finished runs. **A gpu-vs-cpu time ratio below
about 1.15× is not resolvable from evaluation noise alone**, before any
trajectory or seed variance is counted.

A first-crossing time is only meaningful where the accuracy curve is *rising* at
the threshold. Its uncertainty is
`σ_rounds ≈ σ_accuracy / (dAcc/dRound)`, so as the curve flattens the same noise
buys progressively more ambiguity:

| target | slope (acc / 1k rounds) | ambiguity, median | ratio needed to be meaningful |
|---|---|---|---|
| 45 % | 0.19 – 0.51 | 7 % of the measured time | 1.11× |
| 50 % | 0.065 – 0.17 | 10 % | 1.14× |
| 55 % | 0.033 – 0.069 | 10 % (worst 18 %) | 1.14× |
| 58 % | 0.005 – 0.051 | — slope collapses | — |
| 60 % | 0.007 – 0.009, **one negative** | — | — |

Against the observed 55 % ratios — G=1 **1.13×**, G=10 1.23×, G=20 0.71×,
G=30 **1.11×** — the two marked ones sit inside the noise. G=1 is the control and
*should*: its 13 % is fully explained by the ~10 % eval ambiguity, which is the
cleanest confirmation available that the control behaves.

**Measure σ correctly.** A straight-line fit over ±400 rounds reports
σ ≈ 0.02–0.06 because it counts curvature as noise, and it does so worst exactly
where the slope is steepest. Two curvature-free estimators agree on the real
figure — residual around a rolling median, and `std(diff(acc))/sqrt(2)` — both
giving **σ ≈ 0.005–0.011**. That is 1–2× the irreducible binomial noise of a
10 000-image test set at 60 % accuracy (`sqrt(0.6·0.4/10000) = 0.0049`), which is
the sanity check that the estimate is right.

#### `target_accuracy = 0.6` fires on a coin flip — and truncates the run

The early stop reads a single noisy evaluation, so it terminates runs at a random
moment rather than a comparable one:

- **`cpu_xa20_gap20` crossed 0.60 at round 3770 with a local slope of −0.0299 per
  1k rounds** — its trend was *declining*; it crossed on an upward fluctuation
  and stopped. `gpu_xa20_gap20` never crossed and paid the full 10 000 rounds.
  The two runs' lengths differ by 2.7× for no reason connected to either selector.
- Runs loiter in `[0.585, 0.60)` for enormous stretches without crossing:
  `gpu_xa20_gap10` **54.6 %** of its run, `gpu_xa20_gap01` 44.6 %,
  `cpu_xa20_gap10` 31.2 %, `cpu_xa20_gap01` 29.2 %.

So "best accuracy" is not comparable across runs either: a run that early-stopped
reports the accuracy that tripped the stop, a run that did not reports its
plateau. Do not tabulate them side by side.

#### What to do instead

1. **Report 45 % and 50 %**, where the slope is 5–10× steeper and ambiguity falls
   to 7–10 %. 55 % is usable; 58 % and 60 % are not.
2. **Prefer a statistic that does not depend on one evaluation.** Accuracy at a
   *fixed simulated-time budget* is the cleanest — it inverts the axes and the
   crossing problem disappears. A first-crossing on a rolling-median-smoothed
   curve is the cheap alternative.
3. **Turn the early stop off** (`target_accuracy` above any reachable value) so
   every run has the same length and the curves stay comparable. It buys nothing
   here anyway — § 4.12 — because 0.60 sits on the plateau.
4. Combined with the n=1 problem (§ 4b, and the multi-seed note), the honest
   statement today is: **the mechanism results are solid, the end-to-end timing
   results are not yet separable from noise.**

### 4.14 The same `T_round` bug lands on the *mix* axis, harder

**Measured 2026-08-15, while designing the fleet-mix sweep (§ 4d).** § 4.11 shows
`t_round_model = "min"` under-estimating `T_round` by an amount that decays with
the gap. Hold the gap and move the fleet *composition* instead and the same error
reappears — larger, and not even monotone:

| fleet (xav/orin/thor) | E[T] s | `"queue"` | `"min"` | error of `"min"` | pref `"min"` | pref `"queue"` |
|---|---|---|---|---|---|---|
| 85/10/5  | 17.40 | 1.740 | 5.672 | **+226 %** | 2.01 | 3.01 |
| 70/20/10 | 14.80 | 1.480 | 2.169 | +47 % | 2.80 | 3.16 |
| 50/35/15 | 11.38 | 1.138 | 1.270 | +12 % | 3.31 | 3.42 |
| 34/33/33 |  8.22 | 0.822 | 0.721 | −12 % | 3.86 | 3.74 |
| 15/35/50 |  4.61 | 0.461 | 0.670 | +45 % | 3.93 | 4.26 |

All at G=30, P=10, `pref` = the speed term's preference for thor over xavier
(`sqrt(1+τ̂_xavier)/sqrt(1+τ̂_thor)`, § 4.11b).

`E[min of 10]` is a property of the *pool's shape*, not just its mean. With 85 %
stragglers the minimum of ten draws is usually still a straggler, so `"min"`
returns almost a whole client duration; at 15/35/50 half the pool is thor, so the
minimum collapses onto thor and the estimate overshoots the queue model from the
other direction. Both effects vanish near 34/33/33 and reappear at either end.

Consequence for a composition sweep: under `"min"` the selector's steering
strength would vary by up to 1.49x **across the sweep's own x-axis**, with a
minimum in the middle — a U-shape that no property of the fleet explains. Any
"the advantage peaks at an intermediate mix" result would then be unreadable,
because the estimator's error peaks at the ends. This is why § 4d runs on
`"queue"` and re-runs the 50/35/15 cell rather than reusing `runs/gap_sweep`.

The gap sweep is *not* retroactively affected: the 50/35/15 row is the one where
the two models nearly agree (+12 %), which is what § 4.11b already said.

## 4b. Predictions for the post-fix sweep — written 2026-08-10, before it ran

Recorded before the rerun on purpose. The point is that tomorrow is a *check*,
not an exercise in explaining whatever came out. If a line here is wrong, say so
and keep the wrong prediction visible.

Derived from the two 800-round verification runs (gpu, G=40) and the identity
`seconds per round ~ T_mean_selected / C`.

**Preconditions — if these fail the run is void, not interesting**

1. In-flight concurrency ~9-10 in all 15 runs. Any run that drains is discarded.
2. Flushes ~2 % of rounds; staleness p90 under 25.
3. Mean staleness ~ `C / M` (~8-9) and **the same for every arm** — it has to be:
   the mean of `T_i / T_mean` is 1 by construction, whatever the selector picks.
   Measured 8.59 against `C/M` = 8.6 on the bound-25 run. If the arms differ here,
   the model in § 4.9 is wrong and everything below is suspect.

**Mechanism**

4. Seconds per round at G=40: `gpu` ~0.98 (measured 774 s / 791 rounds), `cpu`
   ~1.4 (`T_mean` ~12.7-13.1 s / 9). So gpu rounds ~1.4x shorter.
5. Tier mix: `gpu` steers harder than it did pre-fix — xavier ~29 % (was 37.6 %),
   thor ~29 % (was 22.5 %). `cpu` stays at the pool's own ~50 %: it cannot see the
   difference, so nothing about the scheduler should move it.

**Results**

6. **Rounds** to target: `gpu` still <= `cpu`. This is the one at risk — staleness
   is now ~8.6 mean instead of ~1.2, and gpu's distribution is the wider of the
   two, so the staleness weighting bites it harder. If gpu's round efficiency has
   eroded, that is a real finding about the method, not a bug to hunt.
7. **Time** to target: `gpu` wins at G >= 20 by roughly 1.5-1.8x (1.4x from
   shorter rounds, ~1.2x from fewer of them). This is the headline the pre-fix
   sweep got backwards.
8. `baseline` last of the three, and the `cpu`-vs-`baseline` gap answers a
   question the sweep has not yet answered: is a wrong predictor better than none?

---

## 4c. The multi-seed sweep (built 2026-08-12, not yet run)

```bash
python3 make_configs.py --seeds 8888 1234 5678 --rounds 5000 \
    --runs-dir gap_sweep_ms --outdir configs_ms
CONFIG_DIR=$PWD/configs_ms ./run_sweep.sh gpu cpu baseline --seeds "8888 1234 5678"
```

45 configs, three seeds x three arms x five gaps. **The single-seed path is
byte-identical to before** — `make_configs.py` with no `--seeds` renders exactly
the files the 8888 sweep is running against, verified by hash — so this could be
built while that sweep was in flight.

### Why 5000 rounds, not 10000

The metric is time-to-55%, which is a *prefix* property: the answer is fixed once
the run passes 55%. The slowest cell in the 8888 sweep crossed at **round 3160**,
so 5000 covers every one with 58% margin and costs half. That is the whole reason
three seeds is affordable:

| | round-units | vs today |
|---|---|---|
| 1 seed x 15 runs x 10000 | 150 000 | — |
| 3 seeds x 15 runs x 5000 | 225 000 | 1.5x |

And the 8888 runs already on disk **count as seed 1** for this metric, since a
10 000-round run contains the 5 000-round one as a prefix. Only two more seeds
are actually needed: 150 000 round-units, ~20 % *less* than finishing the current
sweep as-is.

### Do not use `target_accuracy` as the early stop here

It reads a raw, unsmoothed evaluation, so it fires on a noise spike and run
lengths become random — `cpu_xa20_gap20` stopped at round 3770 while its trend
was *declining* (§ 4.13). A fixed `rounds` gives every run the same length and
keeps the curves comparable. The templates still carry `target_accuracy = 0.6`;
it is unreachable at 5000 rounds, so it is inert rather than harmful, but set it
above 1.0 if you want the guarantee.

### Ports

Seed index 0 keeps the historical `port_base + G` (8101-8340). Later seeds are
allocated from 9000 up, clear of that range — the first attempt allocated them
from 8000 and put baseline's third seed on top of cpu's first, which
`make_configs.py`'s uniqueness check caught. Five seeds per arm before the blocks
collide; `port_for()` raises past that.

### If a second machine runs some of these — split by SEED, never by arm

`/workspace` is a shared mount, so a second pod can pull from the same
`configs_ms/`. But § 4.8: `elapsed_time` carries the host's real serialisation and
transfer cost, and the hosts measured 0.352-0.387 s per report — a real
difference. Give one machine **whole seeds** (all three arms, all five gaps).
Then every gpu-vs-cpu comparison is within one host, and the host effect lands in
seed-to-seed variance, which is exactly what averaging over seeds removes. Split
by arm instead and the host difference sits directly on the comparison the sweep
exists to make.

### This sweep stays on `t_round_model = "min"`

Seed 8888's runs were made under it (§ 4.11b), and reusing them as seed 1 requires
the model to match. Flipping to `"queue"` is a separate sweep, after this one.

## 4d. The fleet-mix sweep (built and launched 2026-08-15)

The second axis. The gap sweep moves the tier *speeds* and holds the fleet at
50/35/15; this one pins the speeds at G=30 (xavier 20 s, orin 3.651 s, thor
0.667 s, effective gap 8.25x) and moves the *proportions*:

| fleet | E[T] s | capacity vs 70/20/10 | what it is |
|---|---|---|---|
| 70/20/10 | 14.80 | 1.00x | straggler-heavy |
| 50/35/15 | 11.38 | 1.30x | the historical fleet, re-run under `"queue"` |
| 34/33/33 |  8.22 | 1.80x | balanced |
| 15/35/50 |  4.61 | 3.21x | fast-heavy, the mirror of the historical one |

Three arms x four fleets = **12 runs**, 5 000 rounds each, seed 8888, everything
else identical to the gap sweep.

```bash
cd examples/async/fedasync/gpu_tiers
./make_mix_sweep.sh                      # mixes -> sleep CSVs -> configs_mix/
CONFIG_DIR=$PWD/configs_mix ./run_sweep.sh gpu cpu baseline \
    --gaps 30 --mixes "m50_35_15 m70_20_10 m34_33_33 m15_35_50"
python3 mix_sweep.py --plots             # tables + figures/mix/

# the host-cost-free time axis (§ 4.8): rebuild the clock, then read it
python3 recompute_wall_time.py --runs-dir ../runs/mix_sweep \
    --config-dir configs_mix --out-dir figures/mix/clean_time
python3 mix_sweep.py --plots --time-source clean
```

Both runners write into the same `logs/`, so `check_run.py --all` and
`analyze_steering.py` see the mix runs too; point the first at this sweep with
`--config-dir configs_mix --runs-dir ../runs/mix_sweep`.

### What changes, and what deliberately does not

`make_mixes.py` writes one `input/clients_m<x>_<o>_<t>.csv` per fleet in
**contiguous client-ID blocks**, the layout `input/clients.csv` already had.
Two consequences are load-bearing:

- **Client i keeps its data in every cell.** PLATO derives the shard from
  `random_seed` and the client ID, so the mix changes which *speed* is attached
  to a fixed dataset. Re-drawing tier membership per mix would confound the
  composition with which shards happened to land on the fast tier.
- **The fleets are nested.** Every thor client at 15 % is still thor at 50 %.

The tier spec (`tiers_geo_gap30.csv`) is untouched and the anchor stays
xavier = 20 s — `anchor_scale()` reads the tier spec only, never the fleet — so
"fix the GPU speeds, change the ratio" is literally true: all four sleep CSVs
carry the identical 20 / 3.651 / 0.667 s, just to different client IDs.

### Two settings this sweep does not share with the gap sweep

1. **`t_round_model = "queue"`** (§ 4.14). Under `"min"` the estimator's error is
   +226 %…−12 % *along this sweep's x-axis*, so the steering strength would move
   with the fleet for reasons that have nothing to do with the fleet. The price
   is that the 50/35/15 cell had to be re-run: `runs/gap_sweep` is a `"min"`
   sweep and mixing the two would be exactly the § 4.9 mistake.
2. **`target_accuracy = 1.01`**, i.e. no early stop (§ 4.13). At 5 000 rounds
   0.60 is reachable — `cpu_xa20_gap20` crossed it at round 3770 — so leaving it
   at 0.6 would truncate some cells and not others. Every cell now runs exactly
   5 000 rounds and time-to-55 % is read off the smoothed curve afterwards.

Both are rendered options (`make_configs.py --t-round-model`,
`--target-accuracy`), defaulting to the historical values, so regenerating
`configs/` stays byte-identical to what the gap sweep ran.

### Ports and names

`<arm>_xa20_gap30_m<x>_<o>_<t>`, ports from 11000/12000/13000 (§ `port_for`), and
`runs/mix_sweep/` — all three disjoint from the gap sweep's, so both can be in
flight at once. Mixes and multiple seeds together are refused rather than
silently packed into the same blocks.

### Two lanes on one box — and the disk quota that made it look impossible

Run as two `run_sweep.sh` lanes split by fleet (m50_35_15 + m70_20_10 /
m34_33_33 + m15_35_50), both live for the whole sweep, six runs each. Measured
over the same round window (50 → 150), same configs:

| | per run | aggregate |
|---|---|---|
| two lanes | 24.5 and 24.1 rounds/min | **48.6** |
| one at a time | 26.9 rounds/min | 26.9 |

**1.81x throughput for a 10 % slowdown per run.** Do not conclude this from the
first 200 rounds of a run — those are cheap and make serial look faster than it
is; take a window well past startup.

RAM is not the constraint and never was: measured 16.2 GB RSS per run (cgroup
`total_rss`, 12 processes) against a 46.6 GB limit, so two lanes sit at ~33 GB
with ~13 GB spare, `failcnt = 0`, `oom_kill = 0` throughout.

**What actually broke the first attempt was the disk quota.** Both runs froze at
round ~175 with every process in `ep_poll` at 0 % CPU — the § 4.9-shaped symptom
that looks like a deadlock. It was not. Thousands of lines up, inside a client's
stdout:

```
--- Logging error ---
  File "/usr/lib/python3.11/logging/__init__.py", line 1094, in flush
    self.stream.flush()
OSError: [Errno 122] Disk quota exceeded
```

`/workspace` is a MooseFS share with a **quota `df` cannot see** — it reported
752 TB free while writes were failing. Two runs write checkpoints at twice the
rate, the quota went, and everything blocked on a write that never completed.
Same family as the 2026-08-13 incident in § 5 and, almost certainly, the
2026-08-14 capture in `diag_hang_20260814/` (that one was diagnosed as
"FS healthy" on the strength of `df`).

So the rule is not "never run two". It is:

- **Probe the quota by writing.** `run_sweep.sh` now does, before every run
  (`--probe-mb`, default 4096, 0 disables). `df` is not evidence.
- **Checkpoints of finished runs are the thing that fills it.** They are
  intermediates — no analysis script reads them, only `results/*.csv` and
  `logs/*.log` — and 22 finished runs were holding 92 GB. Pruning them with
  `clean_checkpoints.sh <run>/checkpoints 5 --once 20000` took `/workspace` from
  161 GB to 58 GB.
- **Keep fewer while running.** This sweep uses `--keep-rounds 20
  --clean-interval 3`, ~3 GB per live run instead of ~8 GB.

### All twelve cells run in the same mode, on purpose

The first two cells had already finished serially when the switch was made, and
they were **deleted and re-run** rather than kept. Contention is part of a cell's
measurement (§ 4.8): `elapsed_time` carries the host's real serialisation cost,
and a cell that had the box to itself is not reading the same clock as one that
shared it. Re-running two cells costs ~3.4 h against a sweep that is ~10 h
shorter this way.

`recompute_wall_time.py` can *also* normalise this after the fact — it replays
the recorded schedule charging every report a uniform floor, and
`mix_sweep.py --time-source clean` reads the result. Measured on this sweep the
floor is **0.395 s per report, ~3 % of the simulated clock**, reconstructed to
0.20 % of PLATO's own. But it only fixes the measurement: the selector read
`wall_now` when it computed `residual`, so *which clients were chosen* was
already decided against the contended clock, and no post-hoc pass undoes that.
Uniform contention plus the clean clock is belt and braces; either alone leaves
one of the two problems.

### What the analysis has to separate

`mix_sweep.py` reports three things, and the second is the point:

- **capacity** — E[T] falls 3.2x from 70/20/10 to 15/35/50, so most of the change
  in raw hours-to-55 % is available to a selector that does nothing. The
  `baseline` arm measures it.
- **steering** — every arm as a ratio to *that fleet's own* baseline. This is the
  only number that is about selection.
- **dispatch lift** — share dispatched over share of pool, per tier, so a fleet
  that is 50 % thor does not read as "steering toward thor".

### Pre-registered predictions, written before the runs finished

1. `baseline` lift stays ~1.00 on every fleet. Anything else is a bug.
2. `gpu` keeps thor > orin > xavier on all four fleets; the *lift* shrinks as the
   fast tiers grow, because lift is normalised by pool share and there is less
   left to prefer.
3. `gpu ÷ baseline` hours-to-55 % is **not** monotone in the fast share: at
   70/20/10 there are only 10 thor clients and `max_concurrency = 10` cannot keep
   picking them, at 15/35/50 preferring thor is nearly free anyway, so the
   advantage should be largest in the middle. This is the one at risk, and the
   reason the sweep exists.
4. `cpu ÷ baseline` stays near 1.0 everywhere: its belief cannot separate xavier
   from orin (§ 4.5), and that does not change with the mix.
5. Rounds-to-55 % moves much less than hours-to-55 %: mean staleness is C/M
   (§ 4.9), which the composition barely touches.

---

## 5. Disk

A checkpoint is ~34 MB and rounds arrive at ~56/min, so an unmanaged run would
leave ~440 GB behind. `clean_checkpoints.sh` keeps the last `KEEP_ROUNDS` and is
started automatically by `run_sweep.sh` — **it does not need its own screen.**

Peak usage is set by the cleaner period, not by `--keep-rounds`:

```
peak ~ (KEEP_ROUNDS + 56 * INTERVAL_MINUTES) * 34 MB
```

The default 5 minutes keeps one run near 8 GB. The 20-minute period inherited from
`clean_old_checkpoints.sh` was tuned for runs that really slept 20 s per round; at
the current rate it would let ~20 GB pile up between passes.

`clean_checkpoints.sh` refuses any directory whose basename is not `checkpoints`,
and never touches `current_round.pkl` (no round number in its name, and PLATO
resumes from it).

The older `clean_old_checkpoints.sh` / `_ver2` / `_ver3` in the parent directory
are superseded — three copies of one file differing only in a hard-coded path.

---

## 6. Operational notes

```bash
screen -r sweep                       # attach; Ctrl-A then D to detach
python3 check_run.py --all            # § 4b preconditions for every run so far
python3 plot_time_to_accuracy.py      # comparison so far — safe to run mid-sweep
ls logs/*.done | wc -l                # runs finished
tail -f logs/gpu_xa20_gap01.log       # current run
tail logs/*.cleaner.log               # cleaner activity
du -sh ../runs/gap_sweep/*            # disk per run
grep -c 'PolarisVi] Round' logs/<run>.log   # progress
```

- Stopping: `screen -X -S sweep quit`. The runner's trap kills the trainer, its
  client subprocesses, and the cleaner. A run that did not finish leaves no
  `.done` marker, so re-running picks it up from the start.
- The runner launches the trainer with `exec` inside the subshell on purpose:
  without it `$!` is the subshell's PID, and killing that leaves the trainer
  reparented to init, still running and still writing checkpoints. This actually
  happened during testing.
- `kill_tree` takes the client subprocesses down before the trainer — they are its
  children and hold GPU memory.

### Current run parameters

`rounds = 10000`, `target_accuracy = 0.6` (acts as early stop), `per_round = 10`,
`max_concurrency = 10`, `partition_size = 500`, `epochs = 1`, `batch_size = 64`,
`lr = 0.01`, `sampler = "noniid"`, `concentration = 0.9`, `random_seed = 8888`,
`mixing_hyperparameter = 0.5`, `staleness_bound = 25` (raised from 11 on
2026-08-10, § 4.9), anchor 20 s.

The measured rate is **~28 rounds/min**, not the 56 quoted earlier, so a full
10 000-round run is ~6 h. Observed on `gpu_xa20_gap01`: 9.7 % @ round 11,
36-40 % @ round ~270-700.

**`target_accuracy` was 0.7 until 2026-08-09.** Accuracy saturates near 0.60 by
round ~4000-6000 and is flat afterwards, so 0.7 never fired and every run paid
the full 10 000 rounds:

| run | best accuracy | when |
|---|---|---|
| `gpu_xa20_gap01` | 0.5943 | r8150 (never reaches 0.6 — will still run to `rounds`) |
| `cpu_xa20_gap01` | 0.6032 | r5760 |
| `gpu_xa20_gap10` | 0.6003 | r3880 |

The two `gap01` runs finished under 0.7 and were **not** redone. `gpu_xa20_gap01`
is unaffected — it never crossed 0.6, so 0.6 would have produced the identical
trajectory. `cpu_xa20_gap01` ran ~4200 rounds past where 0.6 would have stopped
it; truncate at the crossing when comparing against runs made under 0.6.

### Arms are split across two pods

`/workspace` is a shared network mount, so a second pod can run its own arm from
the same `configs/`. As of 2026-08-09 the split is:

| pod | arms | launched with |
|---|---|---|
| this one | `gpu`, `baseline` | `screen -S sweep -dm ./run_sweep.sh gpu baseline` |
| the other | `cpu` | its own `run_sweep.sh cpu` |

Consequences worth knowing:

- `screen -r sweep` here shows **only** this pod's arms. The other pod's run is
  invisible to `ps`, `ss` and `nvidia-smi` here even though its log under
  `logs/` is visibly growing — that is not a stuck or orphaned process.
- The memory guard reads *this* cgroup only. It cannot see the other pod, which
  is fine while they are separate containers, but do not use the two-pod split
  as evidence that two arms fit in one container. They do not (§3).
- `configs/` is shared. Regenerating it here changes what the other pod launches
  **next**; runs already in flight there keep the config they started with.

---

## 7. Verification

```bash
python3 make_configs.py --check              # configs parse, inputs exist, arms comparable
python3 test_predictor_anchor.py             # 16 checks on predictor anchoring
python3 smoke_test_sleep_simulation.py       # A/B sleep_simulation, ~4 min, needs a GPU
python3 summarize_sweep.py --anchor-sleep 20 --predictor-tiers input/tiers_cpu.csv
```

Two pre-existing failures in `test_t_round.py` are **not** regressions:
`test_t_round_positive_and_sane` asserts the pool *median* while the function
computes `E[min of P]` (the pool mean at P=1) — a stale expectation from when the
function's semantics changed; and `test_selector_with_gpu_tops_predictor` fails on
the broken venv torch. `test_staleness_predictor.py` passes.

---

## 8. File map

| file | role |
|---|---|
| `run_sweep.sh` | run one or more arms, all gaps, with per-run cleaner |
| `clean_checkpoints.sh` | prune old checkpoints for one run directory |
| `make_configs.py` | render `templates/` -> `configs/`, with validation |
| `templates/*.tmpl` | the three blueprints — **edit these, not `configs/`** |
| `configs/*.toml` | generated, 15 files |
| `gpu_tier_sleep.py` | tier formula, `T_round`, predictor, `predictor_scale()` |
| `generate_sleep_times.py` | build the per-client sleep CSVs (anchored) |
| `make_tier_gaps.py` / `make_cpu_tiers.py` | build the tier spec CSVs |
| `make_mixes.py` | build the per-fleet `clients_m*.csv` (§ 4d) |
| `make_mix_sweep.sh` | regenerate the whole fleet-mix sweep: mixes, sleep CSVs, `configs_mix/` |
| `mix_sweep.py` | the fleet-mix comparison — capacity vs steering vs dispatch lift |
| `make_gap_sweep.sh` | regenerate all tier + sleep CSVs at once |
| `summarize_sweep.py` | effective-gap / mismatch table |
| `plot_time_to_accuracy.py` | the comparison figures — hours-to-target per arm, per gap |
| `recompute_wall_time.py` | rebuild the simulated clock from the logged schedule, host effects removed (§ 4.8) |
| `verify/run_verify.sh` | 800-round G=40 run that proves the pipeline stays full (§ 4.9) |
| `verify/make_verify_config.py` | derive that run's config from `configs/` — rerun after `make_configs.py` |
| `verify/plot_before_after.py` | the before/after concurrency figure |
| `check_run.py` | check one run (or all) against the § 4b preconditions |
| `../tests/test_flush_refill.py` | regression test for the drain (§ 4.9) |
| `test_predictor_anchor.py` | anchoring regression test |
| `smoke_test_sleep_simulation.py` | `sleep_simulation` A/B |
| `input/`, `output/` | tier specs and per-client sleep CSVs (inputs to runs) |
| `../README_gpu_tier.md` | the longer background write-up |
