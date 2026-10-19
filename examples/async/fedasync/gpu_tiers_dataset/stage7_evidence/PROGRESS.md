,# Stage 7 — progress log

Running record against `prompts/STAGE7_open_population.md`. Appended to as each
step completes; not rewritten in place.

Code lives in a new sibling directory, `sumo_open_loop/`, not in
`sumo_close_loop/` — same reasoning every prior stage's generator gives for
staying separate: Task 4 / Stage 5 / 5b / 6 hold finished, accepted results
produced by `sumo_close_loop/`, and this cell must not be able to disturb them.

---

## Step 1 — the slot allocator. DONE.

`sumo_open_loop/stage7_population.py`:

| piece | what it does |
|---|---|
| `stable_digest()` | blake2b, **not** Python's `hash()` — the latter is randomised per process by PYTHONHASHSEED, so a spawned client would disagree with the server about a vehicle's sleep and shard. Same trap, same fix, as `client_seeded_trainer.derive_seed` |
| `vehicle_sleep_s()` | `T_tier · exp(ε)`, `ε ~ N(0, 0.149166²)` keyed on `(base_seed, veh_id)` — the open-population stand-in for a `sleep_table.csv` that cannot exist for 10,523 vehicles |
| `partition_index()` | `blake2b(veh_id) mod M` — keyed on the **vehicle**, never the slot |
| `SlotAllocator` | M slots, FIFO reuse, `assign`/`release`, `slot_exhausted` counting, `RecycleEvent` per handover |
| `reset_slot_state()` | the `p_recur = 0` wipe |

**`T_TIER = {xavier: 15.0, orin: 1.923077, thor: 1.0}` and `σ = 0.149166`**, per
the stage prompt. The trace's own `t_bar_train_s` (30/18/10, ratio 3:1.8:1,
flagged `provisional`) is **not used** — it would make tier separation 3× instead
of 15× and break comparability with every stage since Stage 3.

### What a recycle resets, and how that was determined

Read out of `fedasync_polaris_vi_selection.py` and
`fedasync_polaris_vi_server.py` rather than assumed. Four places a slot's
history lives:

| state | location | reset to |
|---|---|---|
| reactive `τ̂` | `context.state["polaris"]["local_stalenesses"][slot-1]` | `0.01` (unseen default) |
| `G_i` | `context.state["polaris"]["local_gradient_bounds"][slot-1]` | `0.5` (initial prior) |
| in-flight prediction | `context.state["polaris"]["staleness_predictions"][slot]` | dropped — otherwise it is scored against the **new** occupant's report |
| "never yet selected" | `server.unexplored_clients` (set of zero-based indices) | slot re-added |

That last set is this codebase's `times_selected`: a client in
`unexplored_clients` is handed the proxy `G_i = α·avg_norm` rather than a
measured one (`fedasync_polaris_vi_server.py:288-293`). Re-adding the slot is
what "selection counts return to zero" means here.

### Required evidence — both tests present and passing

`sumo_open_loop/tests/test_stage7_population.py`, **19 tests, all passing**:

- `test_p_recur_zero_wipes_every_piece_of_slot_state` — after a recycle at
  `p_recur = 0`: `τ̂ == 0.01`, `G_i == 0.5`, no pending prediction, slot back in
  `unexplored_clients`
- `test_p_recur_one_keeps_every_piece_of_slot_state` — the opposite at
  `p_recur = 1`: all four survive
- tier follows the incoming vehicle at **both** `p_recur` values
- exhaustion is reported rather than evicting a training vehicle
- determinism: blake2b stability, per-vehicle sleep reproducibility, ε's sd
  matching σ, partition keyed on vehicle not slot

One test bug was found and fixed in the process, worth recording because it
was a real disagreement rather than a typo: the first draft assumed a freed
slot is handed straight back to the next arrival. It is not — `release()`
appends to the free list (FIFO), so recycles spread over all M slots instead of
churning a few hot ones. The tests now fill the allocator before releasing, so
the freed slot is the only one available. That is also the realistic case:
recycles matter when the allocator is near-full, which is when M is doing any
work at all.

Per-recycle log line, so a run is auditable after the fact:

```
[Stage7] recycle slot_id=7 outgoing_veh_id=fEB.412 incoming_veh_id=fWB.913 \
         tier_out=orin tier_in=xavier state_reset=1
```

---

## Step 2 — pre-run gate. All seven reported. No stop condition trips.

`sumo_open_loop/gate_stage7.py` → `stage7_evidence/stage7_gate.json`.
Trace as delivered: 10,523 vehicles, 10,509 coverage windows, 731,142 trace
rows. (The prompt says 10,510 windows; the file has 10,509 data rows. A
one-row difference, reported rather than reconciled.)

### 7. Warm-up and end-of-sim handling *(done first — everything else uses the clip)*

`warmup_s = 150`, so early windows carry negative `t_enter_s`. Applying
`remap_fleet.py`'s rule (drop what ends before `t = 0`, clip what straddles it):

| | |
|---|---|
| windows with `t_enter_s < 0` | 36 |
| dropped entirely (`t_exit_s < 0`) | **22** |
| clipped to start at `t = 0` | **14** |
| usable windows | **10,487** |

The config's own `n_windows_open_at_end_dropped: 9` were already removed by the
trace generator before delivery; nothing further is done about them here.

### 1. Crossing-time distribution

| | open (this trace) | closed loop |
|---|---|---|
| median | **45.0 s** | 45.0 s |
| p10 | **42.0 s** | 42.0 s |
| p90 | **52.0 s** | 52.0 s |
| below 41.142857 s | **13 / 10,487 = 0.12%** | 0.14% |

Identical to three significant figures. Same road, same speed limit, same
radius — so the mobility side of the environment is genuinely unchanged and
any difference this stage measures comes from the identity policy, not from
the geometry.

### 2. Occupancy → M

| median | p90 | p99 | max | mean |
|---|---|---|---|---|
| 13.0 | 18.0 | **22.0** | 29 | 13.44 |

**M requires a judgement call, because the prompt's two rules conflict here:**

- *"Take `M` from the gate's measured p99 occupancy, rounded up"* → **M = 22**
- *"Keep `M × partition_size = 50,000`"* → M must divide 50,000; **22 does not**
  (22 × 2272 = 49,984)

Measured exhaustion across the whole trace at each candidate:

| M | M × partition_size | partition_size | exhausted | rate | divides 50,000 |
|---|---|---|---|---|---|
| 20 | 50,000 | 2500 | 216 | 2.060% | yes — **but fails the ≤1% stop condition** |
| 22 = ceil(p99) | 49,984 | 2272 | 81 | 0.772% | **no** |
| 23 | 49,979 | 2173 | 52 | 0.496% | no |
| **25** | **50,000** | **2000** | **21** | **0.200%** | **yes** |
| 29 = max | 49,996 | 1724 | 0 | 0.000% | no |

**M = 20, partition_size = 2500 — now fixed by the stage prompt itself.**

The first revision of the prompt derived M from p99 occupancy and set the
exhaustion stop condition at 1%. Under that wording M was a judgement call with
a conflict in it (`ceil(p99) = 22` does not divide 50,000), and M = 20 was
chosen deliberately for the one thing it alone buys: `partition_size = 2500`,
*exactly* what Stages 3-6 used, so data-per-client compares directly across
every cell. The revised prompt now **mandates** that same M = 20 / 2,500 and
raises the exhaustion threshold to 15%, so what was a flagged deviation is
simply the specified configuration. The measurements below are unchanged; only
the threshold they are read against moved.

| | exhausted / arrivals | rate |
|---|---|---|
| whole 36,000 s trace | 216 / 10,487 | **2.060%** |
| only the 8,294 s the 4,000-round run consumes | 40 / 2,357 | **1.697%** |

Both are far under the 15% stop condition, and **below** the prompt's own
expectation of ~4-5%. The two "capacity" numbers measure different things and
are both reported because they are easy to confuse: the RSU is **at or above
capacity 5.36% of the time**, but only **2.06% of arrivals** are turned away —
arrivals are Poisson and spread evenly, while at-capacity periods are short
bursts.

For the record, the alternatives measured before M was fixed: M = 22
(`ceil(p99)`) gives 0.772% but 22 × 2272 = 49,984, breaking the 50,000 rule;
M = 25 gives 0.200% and exactly 50,000 but `partition_size = 2000`, which no
earlier stage used.

### 3. Feasible-and-idle pool per round

`T_round = mean_sleep / C = 8.2942 / 4 = 2.0735 s`, so 4,000 rounds span
**8,294 s** — comfortably inside the trace's usable 35,999 s, so **no
wrap-around is needed**, unlike the closed-loop cell.

| median | p10 | p90 | rounds below 2C = 8 |
|---|---|---|---|
| **9.0** | 4.0 | 14.0 | 1,418 / 4,000 = **35.45%** |

**GATE: PASS** — the criterion is on the median, and 9 ≥ 8. But it passes by
one client, and a third of rounds sit below 2C. Selection has freedom on
average and noticeably less than the closed-loop cell had. Recorded plainly
because it bounds how much room the selector has to express a preference at
all, which is the thing this stage is measuring.

### 4. κ — dispatches per vehicle before it exits

| mean | median | p10 | p90 | max |
|---|---|---|---|---|
| **1.71** | 1.66 | 1.55 | 1.92 | 2.55 |

The closed-loop cells ran at **70–188**. Stage 3's measured k\* — the number of
observations a reactive estimator needs before it catches up — is **30–70**.
κ here is **1.71**, roughly **40–110× below k\***, and below even the prompt's
own prior estimate of ≈3. The regime the stage was built to reach is reached:
a reactive arm cannot possibly converge on a vehicle that it sees fewer than
twice.

### 5. Tier mix among in-coverage vehicles

| tier | count | share | config's realised mix |
|---|---|---|---|
| xavier | 5,151 | **49.12%** | 49.11% |
| orin | 3,658 | **34.88%** | 34.91% |
| thor | 1,678 | **16.00%** | 15.97% |

Survives window filtering intact.

### 6. Slot exhaustion at M = 20, and the turned-away tier mix

10,487 arrivals, **216 turned away = 2.060%**, **10,271 recycles**. Stop
condition is >15%; the prompt's own expectation is ~4-5%. **PASS.**

The revised prompt adds a second, sharper check here: if the turned-away mix
departs from the admitted mix, the allocator is keyed to something other than
coverage residency, which is a defect rather than a capacity effect.

| tier | turned away | admitted | delta |
|---|---|---|---|
| xavier | 49.07% | 49.12% | **−0.04 pt** |
| orin | 37.50% | 34.83% | **+2.67 pt** |
| thor | 13.43% | 16.05% | **−2.63 pt** |

**PASS** — on 216 draws these are ~1σ of binomial noise (thor: 29 observed vs
~35 expected, sd ≈ 5.4). Truncation is tier-neutral, so the realised mix that
this cell's claims rest on is preserved.

### Occupancy-sequence digest — the cross-run invariant

```
d1002cd7c986a4f4a32f6b04383e99f8
```

blake2b-128 over the ordered `(t, A|R, veh_id, slot)` stream the allocator
produces. Because a slot is taken on coverage **entry** and released on
coverage **exit** — never on selection — this is a function of the trace
alone, and **all eight runs must reproduce it exactly**. An arm that differs
has tied slot-holding to something the selector influences, which is precisely
the defect the revised prompt added this check to catch. This value is the
reference.

### Sleep model check *(a named stop condition)*

| tier | n | realised mean | CV | lognormal theory | closed-loop fleet |
|---|---|---|---|---|---|
| xavier | 5,168 | 15.172 s | **15.11%** | 15.168 s | 14.37 s |
| orin | 3,674 | 1.9465 s | **14.95%** | 1.9446 s | 1.78 s |
| thor | 1,681 | 1.0166 s | **15.18%** | 1.0112 s | 1.07 s |

**CV matches the closed-loop fleet's 15% on every tier** — that is the robust
check, and σ = 0.149166 implies exactly 15.00%.

The **means sit above the closed-loop fleet's**, and that is expected rather
than a departure: the closed fleet drew only 10 / 7 / 3 vehicles, so its means
carry large small-sample error. With 5,168 / 3,674 / 1,681 draws the realised
means converge on the lognormal expectation `T_tier · exp(σ²/2)` — which they
do, to four significant figures. Reading this row as "materially departs from
14.37 / 1.78 / 1.07" would be reading small-sample noise in the *old* fleet as
signal. **Not treated as a stop condition trip.**

### Gate verdict

Against the **revised** stop conditions:

| stop condition | status |
|---|---|
| median feasible-and-idle pool below 2C | **pass** (9 ≥ 8, by one) |
| slot exhaustion > 15% of arrivals | **pass** (2.06%) |
| turned-away tier mix departs from admitted | **pass** (max 2.67 pt, ~1σ) |
| occupancy-sequence hash differs between runs | reference recorded; checked per run in Step 3 |
| recycle-reset tests fail | **pass** (both present, 19/19 green) |
| per-tier sleep mean/CV departs from closed-loop | **pass** (CV 15%; mean difference explained above) |

**Nothing trips.** The gate is clear to proceed to the 8 runs.

---

## Step 3 — the 8 runs. IN PROGRESS. Server side built, client side outstanding.

### Built and verified

**`sumo_open_mobility.OpenCoverageTimeline`** — the coverage timeline keyed on
string `veh_id`, driving the allocator purely off the simulated clock.
`advance_to(now)` never looks at who was selected; nothing in the module takes
a selector as an argument. That is what makes the assign/release stream a
function of the trace alone.

**Verified against the gate rather than asserted:** the live driver, run over
the real trace, reproduces the gate's replay exactly —

```
digest d1002cd7c986a4f4a32f6b04383e99f8   turned away 216   recycles 10,271
```

— and three regression tests hold it there: the digest is unchanged at
`p_recur = 0` and `p_recur = 1` (slot holding cannot depend on the identity
policy), and unchanged whether the clock arrives in 500 s steps or one jump
(it cannot depend on how the server happens to tick). **22 tests green.**

**`sumo_open_server.py`** — `OpenPopulationMixin` plus the two arm servers.
What it does at each of the three points where a vehicle's physics has to
override the slot's:

| hook | what happens |
|---|---|
| `_select_clients` | advance the population to `sumo_time()`, apply each recycle's `p_recur` policy, idle the clock forward if no slot is occupied and nothing is in flight |
| `process_client_info` | rewrite `report.training_time` with the **dispatched** vehicle's own sleep **before** `super()`, which is where `base.py:1026` turns it into the finish time that drives the whole simulated timeline |
| `customize_server_response` | send `veh_id` / `tier` / `partition_index` to the client |

One thing worth calling out because it is silent when wrong: a slot can be
recycled to a different vehicle **between dispatch and report**. So
`_dispatch_vehicle[slot]` freezes the occupant at dispatch, and both the sleep
rewrite and the coverage verdict are evaluated against *that* vehicle via
`vehicle_in_coverage_at(veh_id, finish_t)` — never against whoever holds the
slot when the report lands. Scoring one vehicle's work against another's
mobility would not raise anything; it would just quietly corrupt the waste
numbers.

The Step 4 instrumentation is emitted from the same path rather than bolted on
later: per dispatch, `observations` (reports held for this slot's *current*
occupant, zeroed on reset), `G_i`, `G_i_at_prior`, `tau_hat`; per recycle, the
six-field log line; at close, the occupancy digest and reset counts.

### Client side — built, and the injection question resolved

The sampler is built at `plato/clients/strategies/defaults.py:121` from
`context.client_id`, i.e. the **slot**. `VehicleShardLifecycle`
(`sumo_open_fedasync.py`) reads `stage7_partition` out of the server response
and swaps `client_id` for the vehicle's partition index around `load_data`,
restoring it immediately after so nothing else sees the substitution.

The open question flagged earlier — whether replacing the strategy post-init
actually takes effect — was **checked rather than assumed**:
`composable.py:176` captures the stack into `_strategies`, but that tuple is
used in exactly one place (`composable.py:190`, calling `setup()` once at
init). Every runtime call reads `self.lifecycle_strategy` fresh. So a bare
attribute swap would work but would skip the new strategy's `setup()`; the
client therefore goes through `_configure_composable`
(`plato/clients/base.py:104`), which is the supported path — it tears the old
stack down and rebuilds the runtime properly.

### Smoke tests — the integration verified end to end

`baseline_open`, then `observed_open` and `observed_closed` side by side. No
tracebacks in any. What they confirm:

| claim | evidence from the smoke logs |
|---|---|
| sleep follows the **vehicle** | orin dispatch reported `sleep=1.767897`, xavier `sleep=14.291404` — each is that vehicle's own `T_tier·exp(ε)`, not a per-slot constant |
| partition follows the **vehicle** | `slot=12 → partition=12` but `slot=7 → partition=12` as well, and `slot=6 → 9`, `slot=10 → 14`. Two different slots holding the same partition at once is exactly what the revised prompt says is correct |
| slot holding is **trace-determined** | `observed_open` and `observed_closed` produced the *same* recycle at the *same* point: `slot_id=1 outgoing=fEB.5 incoming=fEB.19 tier_out=orin tier_in=thor` |
| the identity policy actually bites | that same recycle carries `state_reset=1` at `p_recur=0` and `state_reset=0` at `p_recur=1` |
| tier follows the incoming vehicle | `orin → thor`, `thor → xavier` across recycles |

One cosmetic gap: on the `baseline` arm the dispatch line logs `G_i=nan
tau_hat=nan G_i_at_prior=0`, because plain FedAsync has no `polaris` state to
read. Correct behaviour, misleading field — `G_i_at_prior` should read as
not-applicable rather than 0 for that arm. Noted; it does not affect any
number, since the baseline arm has no V_i ranking to report.

### Wave 1 launched

2026-09-22 19:12 UTC — `baseline_open`, `gpu_open`, `cpu_open`,
`observed_open` (all `p_recur = 0`), each with its own checkpoint cleaner.
**Four at a time, never eight**: Stage 6 launched 8 concurrently, hit CUDA OOM
on the 20 GB GPU, and four runs hung silently for 17 hours. Wave 2 (the
`p_recur = 1` controls) goes after wave 1 lands.

### Wave 1 results, and a false alarm on the occupancy digest

Three of four finished clean (`baseline_open` 00:36, `gpu_open` 00:38,
`observed_open` 00:31 UTC on 2026-09-23); `cpu_open` is slower, as it is in
every cell, because it steers into xavier.

**The end-of-run digests did not match — and that is not a defect.** Logged:

| run | digest | recycles | slot_exhausted |
|---|---|---|---|
| baseline_open | `ee382e77…` | 2,020 | 39 |
| gpu_open | `8f56e7aa…` | 2,070 | 39 |
| observed_open | `8906824…` | 1,891 | 39 |

Read literally that trips the stop condition. Checked before reporting it as
one: **every digest is an exact prefix of the single trace-determined
stream** — replaying the trace event by event and snapshotting the digest
finds each run's value at a specific point:

| run | stops at | events | recycles |
|---|---|---|---|
| baseline_open | t = 7,262 s | 4,107 | 2,020 |
| gpu_open | t = 7,463 s | 4,209 | 2,070 |
| observed_open | t = 6,807 s | 3,847 | 1,891 |

So the stream is identical across arms; each arm simply consumes a different
amount of the trace, because its dispatch mix sets its clock rate and 4,000
rounds therefore end at a different simulated time. `slot_exhausted = 39` being
identical across all three is the corroborating tell.

**The check was measuring the wrong thing**, comparing prefixes of differing
length. Fixed: `sumo_open_mobility.digest_at(csv, slots, until)` replays to a
**common simulated time**, which is where the invariant actually lives. At
t = 6,807 s it returns `8906824024866ff18ae02283e177bceb` for `p_recur = 0`
**and** `p_recur = 1` — identical, confirming the identity policy cannot touch
the stream — and that is exactly what `observed_open` logged, since that is
where it stopped. Three tests now pin this (25 green).

The invariant the stop condition protects — slot-holding is trace-determined,
not selector-dependent — **holds**. The analyser must compare at
`min(final simulated time)` across the eight runs, not at each run's own end.

### Wave 2 (partial) launched

`baseline_closed`, `gpu_closed`, `observed_closed` launched 01:54 UTC, keeping
four runs in flight while `cpu_open` finishes. `cpu_closed` is held back
deliberately so the two slow cpu runs do not overlap.

### Still outstanding

`analyze_stage7.py` for Step 4's tables — κ measured, the reactive arm's
information state per dispatch, `G_i` at-prior vs measured, `s(τ̂)` per tier,
the recycle/occupancy-digest summary, and `gpu` minus `observed` at round
4,000 for both `p_recur` values. The per-dispatch log lines it needs are
already being emitted by the runs now in flight.

The live wiring is a real build, not a config change, because two things that
are static in every previous cell have to become dynamic here. Both integration
points were located and verified to exist before writing any of it, and **none
of them requires touching `plato/`**:

| what must follow the vehicle | where it is static today | the hook that fixes it |
|---|---|---|
| **sleep time** | client-side `Config().client_sleep_times[client_id-1]`, a static array (`plato/clients/strategies/defaults.py:351`) | the server already overrides `process_client_info`; rewrite `self.reports[sid].training_time` with `vehicle_sleep_s(seed, veh_id, tier)` **before** `super()`, which is what `plato/servers/base.py:1026` turns into `finish_time` and hence the whole simulated timeline |
| **data partition** | `dirichlet.Sampler` seeds on `random_seed * client_id` — the slot | server puts `{veh_id, tier, partition_index}` into `customize_server_response` (`base.py:1493`); client reads it in `process_server_response` (`composable.py:310`) and builds its sampler from `partition_index(veh_id, M)` instead of its slot id |

Remaining to build:

1. an open-population coverage oracle keyed on string `veh_id` (the closed-loop
   `CoverageOracle` is keyed on int `client_id`)
2. the server mixin that drives `assign`/`release` off the simulated clock,
   emits the recycle log line, and calls `apply_recycle()`
3. the two hooks in the table above
4. a `sumo_open_loop/` entry point + config generator + runner, mirroring
   `run_stage6.sh` (cleaner per run, `.done` markers, `--dry-run`)
5. **two waves of 4, never 8 at once** — Stage 6 established that 8 concurrent
   runs exhaust the 20 GB GPU and hang four of them silently for 17 hours

The instrumentation Step 4 asks for (κ measured, the reactive arm's information
state at each dispatch, `G_i` at-prior vs measured, `s(τ̂)` per tier, recycle
summary) all hangs off the same per-dispatch log line, so it is designed
alongside (2) rather than bolted on after.

---

## Step 4 — preliminary results (6 of 8 runs; `cpu` pending)

> **⚠ SUPERSEDED IN PART — read `STAGE7_gpu_arm_audit.md` first (2026-09-23).**
> An audit of the `gpu` arm found its `gpu_tops` predictor is keyed on
> `client_id`, which is a **slot** in this stage, not a vehicle. Its tier label
> matches the vehicle actually training only **37%** of the time — chance.
> Therefore **Predictions 1, 2 and 4 as scored below are withdrawn**, along with
> the `s_tau` figures for `gpu_open` / `gpu_closed`. Prediction 3's mechanism,
> Prediction 6 and every soundness check below are unaffected — none of them
> depend on the `gpu` arm. Issue 4 of that audit (`p_recur = 1` never returns a
> vehicle; state is retained onto a physically different one) affects **all**
> arms.
>
> The `cpu` arm has the **same defect** (same lookup, same closed-loop CSV;
> confirmed in the in-flight logs), so the current `cpu` runs cannot score
> Prediction 5. `gpu` and `cpu` must be re-run with the predictor keyed on the
> vehicle. Requirements are in the audit, section "Required for the re-run".

Generated by `sumo_open_loop/analyze_stage7.py` → `stage7_analysis.json`.
`baseline/gpu/observed × p_recur ∈ {0,1}` are complete at 4,000 rounds. Both
`cpu` arms were relaunched 2026-09-23 15:57 UTC after the clock-deadlock fix
(trap 7) and are at round ~130; **Prediction 5 is not yet scored.**

### Run soundness — every stop condition clear

| check | threshold | measured |
|---|---|---|
| conservation-law error | < 3% | **−0.02 … +0.04%** (all six) |
| plateau | 82.1–83.7% band | **82.35 – 83.02%** |
| occupancy sequence identical across runs | stop if any differ | **identical over the common 1,839-recycle prefix** (one digest, `701d0d99…`) |
| state persists at `p_recur = 0` | stop if it does | `state_resets == recycles` exactly (2020/2020, 2070/2070, 1891/1891); `== 20` (slot init only) at `p_recur = 1` |

The occupancy result matters most: all six arms met the **same vehicles in the
same order**. Any time difference below is the selector, not a different
population. (`analyze_stage7.py` now compares each run's *realised* recycle
sequence, not just the trace's expectation — the earlier version only computed
the trace-side digest.)

### Cumulative time at round 4,000 (reconstructed clock)

| run | T (s) | waste | plateau | cons. err |
|---|---|---|---|---|
| baseline_open | 7136.6 | 12.86% | 0.8253 | −0.02% |
| **gpu_open** | **7335.2** | 14.56% | 0.8235 | −0.01% |
| **observed_open** | **6686.9** | 11.68% | 0.8302 | +0.04% |
| baseline_closed | 7155.8 | 12.16% | 0.8293 | +0.00% |
| **gpu_closed** | **7678.4** | 14.06% | 0.8299 | +0.01% |
| **observed_closed** | **6515.3** | 11.92% | 0.8236 | +0.01% |

**`gpu` minus `observed`:** `p_recur = 0` → **+648.3 s (+9.69%)**;
`p_recur = 1` → **+1163.0 s (+17.85%)**. Positive = `gpu` slower.
**`observed` minus `baseline`:** −6.30% (open), −8.95% (closed).

`observed` is the fastest arm in **both** regimes. `gpu` is the slowest in both
— slower than plain FedAsync by 2.78% (open) and 7.30% (closed).

### Prediction scoring

| # | verdict | evidence |
|---|---|---|
| 1 | **NOT HELD — inverted** | `gpu` does not beat `observed` at `p_recur = 0`; it is **9.69% slower**, ~8× the 1.2% noise floor, opposite in sign |
| 2 | **NOT HELD** | they separate by **17.85%** at `p_recur = 1` — a *wider* gap than at `p_recur = 0`. Per the prompt, this means the `gpu`/`observed` difference **is not about identity churn** |
| 3 | **mechanism HELD, outcome NOT HELD** | see below |
| 4 | **mechanism HELD; outcome moot** | `s(τ̂)` median at `p_recur = 0` is **0.9950 for all three tiers** (the unseen default) vs tier-separated **0.447 / 0.707 / 0.707** at `p_recur = 1` — the §4 decay indeed does not appear. The second half ("`gpu`'s advantage should not cross zero") is unanswerable: `gpu` never holds an advantage to lose |
| 5 | **PENDING** | `cpu` at round ~130 of 4,000 |
| 6 | **HELD** | measured κ at `p_recur = 0`: mean **2.49–2.71**, median **2.0**, max ≤19 — far below k\* of 30–70. The intended regime is reached |

### Prediction 3 in detail — the estimator is starved exactly as designed, and it does not matter

Information state of the reactive arm, per dispatch:

| | obs mean | obs max | zero-observation dispatches | scoring on the `G_i` prior |
|---|---|---|---|---|
| `observed_open` (`p_recur=0`) | **1.49** | 12 | **37.2%** | **36.9%** |
| `observed_closed` (`p_recur=1`) | **115.2** | 303 | 0.4% | 0.1% |

And it never accumulates when identities churn — per round-window:

| window | open: obs mean | open: zero | closed: obs mean | closed: zero |
|---|---|---|---|---|
| 1–250 | 1.43 | 39.7% | 7.32 | 7.1% |
| 250–500 | 1.66 | 35.7% | 21.10 | 0.0% |
| 500–1000 | 1.88 | 31.2% | 42.44 | 0.0% |
| 1000–2000 | 1.46 | 37.2% | 86.15 | 0.0% |
| 2000–4000 | **1.39** | 38.6% | **172.72** | 0.0% |

The open-population curve is **flat** — the estimator ends round 4,000 knowing
no more than it did at round 250. The reset works precisely as the stage
designed it.

**And the arm still wins.** `observed` does not degrade toward `baseline`; it
finishes **6.30% ahead** of `baseline` at `p_recur = 0`, and loses only 2.6%
against its own closed-pool self (6686.9 vs 6515.3 s). This is the case the
prompt names in advance: *"the reactive arm is crippled exactly as predicted
and it still does not cost it anything, which is a finding about the scoring
function, not about the environment."*

### Apparent mechanism — the prior steers into the slow tier

Dispatch share by tier (`T_TIER` = xavier 15.0 s, orin 1.923 s, thor 1.0 s):

| run | xavier | orin | thor |
|---|---|---|---|
| baseline_open | 34.2% | 44.1% | 21.7% |
| **gpu_open** | **34.8%** | 45.2% | 20.1% |
| **observed_open** | **32.2%** | 45.0% | 22.8% |
| baseline_closed | 34.8% | 44.0% | 21.2% |
| **gpu_closed** | **37.0%** | 42.5% | 20.5% |
| **observed_closed** | **30.9%** | 45.9% | 23.2% |

`gpu` dispatches xavier — the 15-second tier — **more** often than `observed`
does (+2.6 pt open, +6.1 pt closed), and its waste is correspondingly higher
(14.56%/14.06% vs 11.68%/11.92%). Both terms push `T = rounds × mean_sleep /
(C × (1 − waste))` the same way, and the closed-pool gap in xavier share (6.1 pt)
tracks the larger closed-pool time gap (17.85%). A single observation is
evidently enough to down-rank a vehicle that just spent 15 s, and at κ ≈ 2.7
most vehicles are dispatched more than once — so even a starved reactive
estimator gets that signal, while a fixed tier prior never does.

**Caveat:** single seed (4126) per cell, `cpu` outstanding. Predictions 1–3 are
scored on the six complete runs and will be re-scored when all eight land.
