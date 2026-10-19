# Stage 11: partial recurrence (`p_return`)

Spec: `prompts/STAGE11_p_return_sweep.md`. Nothing under `plato/` is touched.
No run is launched before Stage 10 Wave B has finished.

## Status

| step | state |
|---|---|
| Step 0: `p_return` identity policy + tests | **done**: 81/81 tests (11 new) |
| Real-server wiring check (no training, CPU only) | **done**: 411 / 411 commuter re-entries restored exactly |
| Configs (51 runs + 4 pre-flight sets) | **done** |
| Queue `run_stage11.sh` | **started 2026-09-29 16:18 UTC; waiting for Stage 10 to finish** |
| Step 1: pre-flight (8 smoke runs) | queued (first thing after Stage 10 and the α gate) |
| Phase 1 (seed 4126, 17 runs) | **done** 2026-10-02 06:18 UTC, 17 / 17 PASS; `STAGE11_seed4126_report.md` |
| Phase 2 (34 runs) | **done** 2026-10-04 04:33 UTC, 34 / 34 PASS (`STAGE11_ALL_DONE`); three-seed tables in `STAGE11_analysis.md` / `.json` |

---

## 1. Implementation

### The identity policy: `occupant_identity.ReturningPopulation`

- It is chosen with `[sumo_open] p_return` and requires `p_recur = 0`.
  `p_recur` keeps its meaning and code path.
- On every slot allocation it draws `u`. If `u < p_return` and at least one
  commuter is out of coverage, the occupant is a commuter chosen uniformly
  among them. Otherwise it is the trace vehicle (a transient, as at
  `p_recur = 0`).
- **Commuter pool**: the 20 Stage 3–10 fleet identities `C1…C20`. Commuter
  k has fleet vehicle k's tier and sleep, and partition k − 1.
- **Both RNG streams are seeded from the physics seed only** (4126): one for
  `u`, one for the commuter pick. The run seed never enters. `u` is drawn on
  every allocation from its own stream, so the `u` sequence is **identical at
  every `p_return`** (common random numbers along the curve).
- A commuter never holds two slots; it becomes available again when its
  window closes.
- Every allocation is logged as
  `[Stage11] alloc t slot trace_veh_id identity_id kind tier state_carried exhausted`.
  At the end, a summary gives the counts, the realised share, exhaustion
  events, arrivals by kind and tier, and the **identity-sequence digest**.

### Allocation-time hooks: `stage7_population.SlotAllocator`

`on_assign` / `on_release` are optional observers, `None` by default. They are
called at the exact moment of each allocation or release. They cannot change
which slot a vehicle gets, so the occupancy stream is untouched (tested).
`on_assign` may only mark whether state is carried. That is the event's
`state_reset` flag.

### State follows the identity: `sumo_open_server.install_returning_population`

- **Slot-indexed state** (`G_i`, observed τ, unexplored flag, in-flight
  prediction) is snapshotted when a commuter's window closes. It is restored
  into whatever slot the commuter next occupies.
- **Identity-keyed state** (observation counts, the `observed_per_vehicle`
  duration history) carries over by construction.
- A transient, or a commuter on its first appearance, starts from the Stage 9
  neutral prior (`reset_slot_state`).
- "`times_selected`" in the spec corresponds, in this codebase, to the
  unexplored flag and the observation count. Both are carried.

### Unchanged

The filter, the predictor and the logs already read the occupant identity
(the 7b abstraction), so commuters and transients go through one code path
(tested). Position, direction, `R_i` and window timing always come from the
trace vehicle physically in the slot.

With `p_return` absent, every Stage 7–10 config behaves exactly as before:
- all earlier tests pass;
- the Stage 10 configs regenerate byte-identically;
- the Stage 10 Wave B runs that start after these edits run the same code
  paths (every hook is `None`).

## 2. Tests: `tests/test_stage11_return.py` (11 new, 81/81 in total)

They use the real trace timeline, allocator, aggregation strategy (in
`UploadGate`) and Stage 0 gate, and pin:

- `p_return = 0`: every allocation is a transient whose identity is the trace
  vehicle, occupant for occupant identical to Stage 10 `p_recur = 0`. The
  identity RNG leaves the global `random` and numpy states untouched.
- `p_return = 1`: every occupant is a commuter. A transient appears only when
  all 20 commuters are in coverage (exhaustion).
- A commuter is never in two slots at once.
- The identity digest depends only on physics seed and `p_return`: two replays
  are equal, and a different physics seed gives a different digest.
- Common random numbers: the commuter allocations at `p = 0.25` are a subset
  of those at `p = 0.5` (or exhaustions there).
- The realised share is within 2 points of nominal (7,000 s at `p = 0.5`). The
  transient tier mix matches the trace (49.1% xavier).
- The occupancy digest is identical with and without the policy.
- **State carried:** on every commuter re-entry, `G_i`, τ and the unexplored
  flag equal the values stamped before its exit. A transient gets the neutral
  prior.
- The filter's `T̂` for a commuter is its **identity's** tier (fleet), while
  the trace vehicle's tier differs (the H3 rule).

### Real-server wiring check (no training, no GPU)

The real `OpenPolarisServer` was built from
`configs/stage11f_p050_s4126/gpu_ret.toml` through the entry point's own
steps, stopping before `server.run()`:

- the policy, the allocator hooks, the Stage 0 gate, the selector lookup and
  the occupant predictor are all installed;
- driving `_advance_population` over t ∈ [0, 3,000) s gave **852
  allocations, commuter share 50.6% (nominal 50%), 0 exhaustions, 411
  re-entries**;
- **all 411 re-entries restored the commuter's state exactly**.

The identity digest from the real server (`c8e51c35…`) equals the digest of
the **independent replay** used by the checkers, at the same allocation count.

## 3. Configs

- `make_stage7_configs.py --stage 11 --variant f1|nf --p-return P --seed N`
  writes:
  - `configs/stage11f_p<PPP>_s<seed>/` (filter on; gpu / baseline at all four
    `p`, observed / cpu at 0.25–0.75);
  - `configs/stage11nf_p<PPP>_s<seed>/` (baseline / none at 0.25–0.75).
- Run names are `<arm>_ret`.
- Each config differs from its Stage 10 Wave A partner only in run dir, port,
  `p_return` and the commuter-pool files.
- There are 51 run configs plus 4 pre-flight sets (100 rounds). No port is
  duplicated, and none clashes with Stage 10.

## 4. The queue: `run_stage11.sh` (log `logs/run_stage11.log`)

1. **Waits for Stage 10** (`STAGE10_ALL_DONE`). Any other Stage 10 ending
   means Stage 11 does not start.
2. **α gate** (`stage11_alpha_check.py`): if Wave B shows α = 10 beating
   α = 1 for gpu+F in the open population in every seed, the queue stops
   before launching (spec). A dry run while Wave B was unfinished correctly
   reported "data missing".
3. **Pre-flight**: 8 smoke runs (gpu+F and baseline+F at each `p_return`,
   seed 4126, 100 rounds), then `preflight_stage11.py`, which must pass at
   100%. It checks:
   - every allocation against an **independent replay** of the identity
     sequence;
   - every dispatch and candidate against the occupant identity (identity,
     tier, sleep, partition, predictor tier, `T̂`, `R_i` from the trace
     vehicle, pass, round lines, selection);
   - the training seed;
   - state carried == re-entries, and resets == transients + first entries;
   - occupancy == 7b;
   - the identity digest equal across arms at each `p`;
   - it also reports the realised share and the tier mix by kind.
4. **Phase 1**: 17 cells at seed 4126, in the spec's order. After every run,
   `stage11_runcheck.py` checks the Stage 10 stop conditions plus:
   - realised share within 2 points of nominal;
   - the identity digest equal to the independent replay at that run's
     allocation count.
   A stop kills every running job group.
5. `stage11_phase1_report.py` (to be written before Phase 1 ends) writes
   `STAGE11_seed4126_report.md`.
6. **Stops until `logs/stage11_phase2.go` exists.** The user writes it after
   review.
7. **Phase 2**: 34 runs (4127, then 4128).

- Never more than four at once.
- Each job runs in its own process group with its own cleaner.
- Checkpoints are deleted when each run ends.

## Notes the spec asks to state

- Commuters return within minutes of simulated time, not the next day. This
  is acceptable here because **no server-side quantity decays with elapsed
  time**.
- The commuter pool is 50 / 35 / 15; transients are 49.1 / 34.9 / 16.0. So
  `p_return` does not shift the arrival tier mix. This will be verified from
  the allocation logs.
- If all 20 commuters are in coverage when `u < p_return`, the window goes to
  the transient. These exhaustion events are counted and reported next to the
  realised share.

---

## 5. Launch log

- **2026-09-30 12:25 UTC**: Stage 10 finished (`STAGE10_ALL_DONE`). The
  Stage 11 queue woke up.
- **12:29 UTC**: α gate **PASS**. Wave B comparison I for gpu+F in the open
  population, α = 1 vs α = 10: **−20.1 / −20.9 / −16.9%**, so α = 1 stays.
- **12:47 UTC**: pre-flight reported **FAILED**. The only failing row was
  "identity digest identical across arms", and **that check was wrong**. It
  compared digests of runs that stopped at different allocation counts (46 vs
  44, 47 vs 54, and so on), so it differs by construction. It is the same
  mistake already fixed in `stage11_runcheck.py`, but not ported to the
  pre-flight. In all 8 runs, every allocation matched the independent replay,
  and each run's digest equalled the replay's at its own count. No run was
  wrong.
- **Fix**: per run, logged digest == replay digest at that run's count; across
  arms, the allocation sequences agree on their common prefix. With this, the
  pre-flight passes **100% in all 8 runs**.
- **Second bug found while checking the first:**
  - `stage11_runcheck.py` imports `preflight_stage11`, and so `plato`, but
    the queue called it without `PYTHONPATH`. Every Phase 1 run check would
    have crashed, and the queue would have stopped after the first run.
  - Fixed: `run_stage11.sh` exports `PYTHONPATH`.
  - Tested: the run check now runs end to end on a smoke log, with the digest
    equal to the replay.
- **Realised-share stop condition checked in advance.** The `u` sequence is
  fixed, so the realised share is a deterministic function of the allocation
  count. Over every count a full run can reach (1,000–3,000), it never departs
  from nominal by more than 2 points. The worst case is +1.95 points at
  p = 0.5, N = 1,769. It cannot trip by chance.
- **13:55 UTC**: the queue was relaunched. Pre-flight passed again (the
  completed smoke runs were reused), and **Phase 1 started**. The first four
  runs were healthy, each with one cleaner, and the GPU was at 91%.
  The GPU sat idle from 12:47 to 13:55 because of the checker bug.
- **2026-09-30 18:55 UTC: the queue stopped itself (false alarm, my mistake).**
  - The first four Phase 1 runs all finished (`.done`).
  - The run check for `p050 gpu_ret` crashed. The 2026-09-29 wiring check had
    built the real server from this run's own config, which left two extra
    server logs (and two results CSVs) in its run folder.
    `analyze_stage7.server_log()` refuses a folder with more than one, so
    `stage10_runcheck.py` died without writing a verdict, and the queue read
    that as a stop.
  - The queue killed the two runs that had just started (p025 baseline /
    observed) and stood idle until 04:30 UTC on 1 Oct: **about 9.5 h of GPU
    lost**.
  - The training itself was unaffected.
- **Fix (2026-10-01 04:30 UTC):**
  - `server_log()` now takes the **newest** server log, and `results_csv()`
    the CSV with **the same pid**, so a mismatch still fails. Every Stage 7–10
    folder has exactly one log, so nothing earlier changes.
  - `stage11_runcheck.py` now reports the Stage 10 check's real error instead
    of a bare `FileNotFoundError`.
  - Re-check of the four finished runs: all PASS. `p050 gpu_ret`: share
    0.5087, digest == replay. Tests 81/81.
  - The stale files and the two killed runs' partial folders were **left in
    place** (moving them was not permitted). The parser now ignores them.
- **04:31 UTC: queue relaunched.**
  - α gate PASS; pre-flight PASSED (reused).
  - The four finished runs were skipped and re-checked (PASS).
  - Running: p025 baseline / observed (re-runs), p075 gpu / baseline.
    13 runs left in Phase 1.
- **2026-10-01, with the user's OK:** the 6 stale files were deleted (wiring-check server logs and CSVs for pids 1390099 / 1390231, and the killed run's pid 2447798). Every Stage 11 run folder now has exactly one server log.
- **2026-10-01 15:39 UTC: the user pre-approved Phase 2.** They will not review between phases, so `logs/stage11_phase2.go` was written now. Phase 2 starts right after Phase 1 and the seed-4126 report.
- `stage11_phase1_report.py` written and dry-run on the 22 runs available (the five still running show as "not finished"). The queue runs it at `PHASE1_DONE`; it writes `STAGE11_seed4126_report.md` + `.json`. With `--seeds 4126 4127 4128` the same script gives the three-seed tables for the final `STAGE11_report.md`.
- **2026-10-01 19:50 UTC: second false stop (checker bug).**
  - `stage10_runcheck.py` exempted only `stage10nf_*` from the rule "a
    filtered run must log [Stage0] lines". The first Stage 11 no-filter run
    to finish, `stage11nf_p050 baseline_ret`, was therefore stopped. It was
    sound: plateau 82.77%, conservation +0.002%, occupancy identical, digest
    == replay.
  - The queue killed `stage11nf_p025` / `p075 baseline_ret` and stood idle
    until 01:40 UTC on 2 Oct, **about 6 h lost**.
  - Fix: the exemption now matches any `stage<N>nf_*`. All 15 finished Phase 1
    runs were re-checked: PASS.
  - The killed runs' stale server logs and CSVs were deleted (the user's
    standing OK for stale logs), along with the leftover checkpoints.
  - **01:40 UTC: queue relaunched.** The two remaining no-filter runs are
    running (p025, p075). Phase 1 then the report, then Phase 2 right away.
