# Stage 5b — progress log

Running record against `prompts/STAGE5B_fix_Ri.md`. Appended to as each step
completes; not rewritten in place.

---

## Step 1 — validation (no runs). CLEARED — proceeding to implementation.

New inputs built first (`remap_positions.py`, mirroring `remap_fleet.py`):
`input/positions_n20r400.csv` — 470,985 rows, client_id/t/x_m/direction,
filtered to `in_coverage=1` and `t>=0` from the raw 712,230-row `trace.csv`
(the file `sumo_mobility.py`'s own docstring already flagged as loadable but
never loaded). Checked before using it as ground truth: direction never
flips inside a single contiguous coverage window (0/10,190 checked), and
position moves monotonically in the direction its `direction` label claims
in every window (0 counterexamples) — so a per-window direction and a single
exit boundary per window are both sound assumptions, not simplifications.

### Check 3 — crossing-time distribution (never reported before)

Over all 10,180 windows: **median 45.0 s, p10 42.0 s, p90 52.0 s**.
**99.86%** exceed 41.142857 s (`vmax_elapsed`'s fixed assumption) — this
number was already known from Stage 5's own validation and reproduces here
exactly, cross-checking that both stages are reading the same trace the same
way.

### Check 1 — `vmax_distance` must not violate the lower bound more than `vmax_elapsed` did

Checked at every candidate-second inside a coverage window (480,957 checks,
finer-grained than the per-window 99.86% figure above):

| formula | violations | compliance |
|---|---|---|
| `vmax_elapsed` | 340 / 480,957 | 99.9293% |
| `vmax_distance` | **0 / 480,957** | **100.0000%** |

**Clears the stop condition outright — `vmax_distance` is a perfect lower
bound in this dataset**, not merely a better one. This is expected from the
math (real distance covered can never exceed `v_max × elapsed`, so remaining
distance / `v_max` can never be less than true remaining time), and the 340
`vmax_elapsed` violations are consistent with what Stage 5 already flagged as
its one known weak spot: windows clipped to `t_enter_s = 0` at a run's
clock start, where the assumed entry point is later than the true one, so
`vmax_elapsed` (built from that assumed entry) can overshoot past the truth
near a run's very first rounds. `vmax_distance` reads the actual position
directly and is unaffected by that clipping.

### Check 2 — `vmax_distance` must be tighter than `vmax_elapsed`

| formula | median gap to truth |
|---|---|
| `vmax_elapsed` | 4.857 s |
| `vmax_distance` | **2.302 s** (52.6% smaller) |

By a strict "is it bigger at every point" count, `vmax_distance` is only
tighter at 80.04% of points (376,825 / 470,777) rather than "essentially
all" as anticipated. Investigated rather than accepted at face value: the
remaining 20% (93,952 points) has a **median magnitude of 0.302 s** and
concentrates at a median of 6 s since window start. This is a small,
expected discretisation effect, not a real weakness — at the instant of
entry, `vmax_elapsed` is *exactly* `41.142857 s` by construction, while
`vmax_distance` reflects the first 1 Hz position sample, which has already
moved a few metres past the true boundary crossing (SUMO samples once per
second; the vehicle can cross the boundary at any sub-second offset within
that). Both readings are essentially tied and both remain valid lower bounds
at every one of these points — the aggregate picture (median gap halved,
zero lower-bound violations) is what the run-time behaviour will actually
see, and it clearly favours `vmax_distance`.

### Direction split

49.9% of candidate-seconds travelling toward `x=1000` (direction `+1`),
50.1% toward `x=200` (direction `-1`) — the trace has no directional bias to
correct for, as expected from a closed-loop population on a symmetric road.

### Verdict

All three checks support proceeding: `vmax_distance` is a strictly better
(here, perfect) lower bound, meaningfully tighter on average, with a fully
understood and benign exception.

## Step 1 — implementation. DONE.

| file | change |
|---|---|
| `sumo_close_loop/remap_positions.py` | new — mirrors `remap_fleet.py`; writes `input/positions_n20r400.csv` (470,985 rows, client_id/t/x_m/direction) from the raw trace, filtered to `in_coverage=1` and `t>=0` |
| `sumo_close_loop/sumo_position.py` | new — `PositionOracle`: `x_at()` / `direction_at()`, linearly interpolated between 1 Hz samples, `None` across any gap bigger than one sampling interval |
| `sumo_close_loop/stage0_filter.py` | `stage0_r_i` config knob (`vmax_elapsed` default / `vmax_distance` / `oracle_exit`); `Stage0FeasibilityGate` takes an optional `position_oracle`, required (raises otherwise) only when enabled AND `r_i_mode="vmax_distance"`; `remaining_range_s()` now dispatches on mode |
| `sumo_close_loop/sumo_server.py` | `install_sumo()` / both server constructors thread `position_oracle` through, unchanged signature order (new param at the end, defaults `None`) |
| `sumo_close_loop/sumo_fedasync.py` | reads `stage0_r_i` / `positions_csv` / `rsu_x_m`, builds a `PositionOracle` only when `vmax_distance` is selected |
| `sumo_close_loop/make_stage5b_configs.py` | new — generates the 2 run configs |
| `sumo_close_loop/run_stage5b.sh` | new — mirrors `run_stage5.sh` |
| `sumo_close_loop/validate_r_i.py` | new — Step 1's three validation checks, kept as a standalone script (not deleted) so the validation is reproducible independent of this write-up |

`vmax_elapsed` stays the default — every Stage 5 and Task 4 config, which
omits `stage0_r_i` entirely, is unaffected. Asserted directly:
`TestResolveStage0Config.test_stage0_r_i_defaults_to_vmax_elapsed`.

18 new tests (`test_sumo_position.py`, plus additions to
`test_stage0_filter.py`): `PositionOracle` interpolation/gap-handling against
both a toy trace and the real n20r400 positions table; `resolve_r_i_mode`
validation; `resolve_stage0_config` reading/requiring `rsu_x_m` only for
`vmax_distance`; all three `remaining_range_s` formulas, including the
`oracle_exit` exact-dwell case and both travel directions for
`vmax_distance`; the constructor-time guard that a `vmax_distance` gate
cannot be built without a `PositionOracle` (but a disabled gate never checks,
regardless of `r_i_mode`).

```
160 passed in ~12s   (sumo_close_loop/tests/, 130 pre-existing + 30 new
                       across Stage 5 and 5b's own additions)
```

Smoke-tested both new configs against the live PLATO server for 45 s each
before launching for real: both boot cleanly, `r_i_mode=` appears correctly
in every `[Stage0]` log line, `vmax_distance`'s `R_i_selected` values vary
continuously per client (position-driven) rather than following the old
formula's clean per-round countdown, and `oracle_exit`'s values match
`window_end - t` exactly.

## Step 2 — the two runs. DONE.

Both finished cleanly 2026-09-20 10:47/10:53 UTC (~4h/4h12m each — faster
than Stage 5's runs, as expected with only 2 processes sharing the GPU),
round 4000, no errors.

---

## Steps 3-5 — analysis (`analyze_stage5b.py`).

A bug was caught and fixed before trusting any number from this analyzer:
the conservation law's `mean_sleep(mix)` was first computed from
`TIER_TRAIN`'s noise-free anchor values (15.0 / 1.923077 / 1.0 s) — the
PREDICTOR side, what `T_hat` uses — instead of the fleet's actual jittered
per-tier mean from `sleep_times.csv` (14.37 / 1.78 / 1.07 s), the SIMULATOR
side, which is what actually consumes wall-clock seconds. That bug alone
produced a ~5% conservation-law error, well over the 3% stop condition —
which is exactly why it got caught: Stage 5's own analyzer used the
empirical mean for this (its docstring even says so), so a ~5x-worse error
than Stage 5's ~1% was the signal something upstream had drifted. Fixed by
threading the same empirical `tier_sleep` dict through both
`conservation_check()` and `decompose()`; error dropped to 0.58% / 0.41%.
`T_hat` itself (`TIER_TRAIN`, in `eligibility_window()`) was never wrong —
only the conservation-law bookkeeping was.

### Stop conditions — all clear (after the fix)

| condition | threshold | vmax_distance | oracle_exit |
|---|---|---|---|
| `stage0_fallback` rate | < 10% | 0.15% | 0.03% |
| conservation law error | < 3% | 0.58% | 0.41% |
| median `eligible_after_stage0` | ≥ C = 4 | 6.0 (worst window) | 7.0 (worst window) |
| plateau vs Stage 5 range (82.1-83.7%) | ≤ 2 pts below | 83.06% | 82.81% |

### Table 1 — the R_i ladder

| row | T @ 4000 | vs `baseline/none` | waste | xavier / orin / thor share |
|---|---|---|---|---|
| control (`baseline/none`) | 7525.2 s | — | 13.17% | 39.0% / 41.1% / 19.9% |
| `vmax_elapsed` (Stage 5 control) | 4388.2 s | **-41.69%** | 0.32% | 22.3% / 50.7% / 27.0% |
| **`vmax_distance`** | 4529.6 s | **-39.81%** | 0.20% | 23.5% / 49.9% / 26.6% |
| `oracle_exit` | 4821.7 s | -35.93% | 0.89% | 25.4% / 49.6% / 25.0% |

Useful dispatches per reconstructed second, last window (2000-4000):
`vmax_elapsed` 0.919/s, `vmax_distance` **0.940/s**, `oracle_exit` 0.840/s —
`vmax_distance` is not just close to `vmax_elapsed` on total time, it is
*better* on this measure.

### Table 2 — the decomposition (THE table this stage exists for)

Same method the prompt used to produce -12.9% / -32.9% for `vmax_elapsed`
(verified by reproducing those two numbers from `stage5_analysis.json`
before trusting the method on the new runs):

| row | waste channel | mix channel | sum | actual gap |
|---|---|---|---|---|
| `vmax_elapsed` (given) | -12.9% | -32.9% | -45.8% | -41.7% |
| `vmax_distance` | -13.0% | **-30.4%** | -43.4% | -39.8% |
| `oracle_exit` | -12.4% | -26.6% | -39.0% | -35.9% |

**The waste channel is essentially IDENTICAL across all three R_i formulas**
(-12.4% to -13.0%) — R_i's accuracy has almost no effect on how much waste
the filter removes, because waste is already near-zero everywhere (`T_hat`,
not `R_i`, is what limits it further; see Table 3). **The mix channel is
what varies, and it shrinks monotonically as R_i gets more accurate**
(-32.9% → -30.4% → -26.6%) — consistent with the diagnosis: the flawed
`vmax_elapsed` formula was acting as an extra, unintended speed filter on
top of the real feasibility one, and fixing it recovers *some* but not most
of that channel.

### Prediction 1 — the decision this stage exists to make

**`vmax_distance`'s gap (-39.81%) lands very close to `vmax_elapsed`'s
(-41.69%) — only 1.9 points apart, on a scale where the other reference
point (`baseline/none`, 0%) is 39.8 points away.** Per the prompt's own
framing: this is decisively the "close to -41.7%" outcome, not the "close to
-12.9%" one. **The R_i bug's approximation cost was small. Stage 5's 41.7%
headline stands as reported — the mix-channel effect was mostly real
feasibility, not mostly an artifact of the flawed R_i formula.**

### Table 3 — filter quality, both denominators, per tier

"of rejected, would have completed" (Stage 5's `vmax_elapsed` numbers, for
reference: xavier 22.8% / orin 75.7% / thor 83.9%):

| tier | `vmax_elapsed` | `vmax_distance` | `oracle_exit` |
|---|---|---|---|
| xavier | 22.8% | 12.8% | 5.7% |
| orin | 75.7% | 32.5% | 9.8% |
| thor | 83.9% | 36.1% | 3.8% |

Drops sharply and monotonically for every tier as R_i improves — **Prediction
2 holds cleanly**. "of admitted, fraction wasted" stays low throughout
(0.09-2.34%, no tier above ~2.3% in any run) — admission quality was never
really the problem; over-rejection was.

### Table 4 — eligibility window (the speed-bias diagnostic, "never reported before")

Mean fraction of each tier's own coverage window during which it passes the
predicate:

| tier | `vmax_elapsed` | `vmax_distance` | `oracle_exit` |
|---|---|---|---|
| xavier | 58.7% | 63.3% | 69.6% |
| orin | 87.2% | 95.3% | 97.8% |
| thor | 89.4% | 97.5% | 100.0% |

This is the clearest single number in the whole stage: even under the exact,
non-deployable ceiling, xavier is only eligible for 69.6% of its own window
(never 100%, because much of a window is spent with less than 15 s left by
construction) — some of the "speed bias" is genuine physics, not a filter
artifact. `vmax_distance` recovers about a third of the gap between
`vmax_elapsed` and that ceiling (58.7% → 63.3% → 69.6%).

### Table 5 — fairness

| | `vmax_elapsed` (given) | `vmax_distance` | `oracle_exit` | `baseline/none` (given) |
|---|---|---|---|---|
| thor:xavier ratio | 4.05× | **3.78×** | 3.28× | 1.76× |
| Jain's index | — | 0.769 | 0.798 | — |

**Prediction 6 holds for both runs**: fairness improves relative to
`vmax_elapsed` without returning anywhere near `baseline/none`'s 1.76× —
consistent with Table 4's reading that some of the bias is real feasibility.

### Table 6 — conservation law

See the stop-conditions table above (0.58% / 0.41%, both runs).

### Table 7 — outcome

| run | waste | plateau | TTA→70% | TTA→75% | TTA→78% | TTA→80% |
|---|---|---|---|---|---|---|
| `vmax_elapsed` (Stage 5) | 0.32% | 82.70% | 0.049h | 0.108h | 0.229h | 0.322h |
| `vmax_distance` | 0.20% | 83.06% | (see `stage5b_analysis.json` `curve`) | | | |
| `oracle_exit` | 0.89% | 82.81% | (see `stage5b_analysis.json` `curve`) | | | |

Plateau unmoved either way (82.8-83.1%, inside Stage 5's own 82.1-83.7%
band) — same conclusion as Stage 5: R_i buys time, not accuracy, regardless
of which formula.

### Prediction scorecard

| # | prediction | result |
|---|---|---|
| 1 | `vmax_distance` lands between the controls; where it lands is the answer | **holds — lands close to `vmax_elapsed` (-39.8% vs -41.7%). Stage 5's headline stands.** |
| 2 | rejected-but-would-complete drops well below 75.7%/83.9% for orin/thor | **holds cleanly** — 32.5%/36.1% (`vmax_distance`), 9.8%/3.8% (`oracle_exit`) |
| 3 | xavier share under `vmax_distance` between 22.3% and 39.0% | **holds, barely** — 23.5%, only 1.2 pts above the `vmax_elapsed` end of the range |
| 4 | `oracle_exit` faster than `vmax_distance`, not by much | **does not hold — reversed.** `oracle_exit` is 6.5% *slower* (4821.7s vs 4529.6s). See note below |
| 5 | waste ≤ ~1% in both runs | **holds** — 0.20% and 0.89% |
| 6 | fairness improves vs `vmax_elapsed`, stays above `baseline/none`'s 1.76× | **holds cleanly for both** — 3.78× and 3.28× |

**On Prediction 4, not explained away**: `oracle_exit` is the ceiling on
*filter accuracy*, but `stage0_t_hat` is held at `tier_spec` in both runs —
the belief about training time is identically imperfect either way. The
conservative formulas (`vmax_elapsed`, `vmax_distance`) systematically
*underestimate* remaining dwell, so a client they admit typically has more
true slack past `T_hat` than the predicate implies — an accidental buffer
against a client's own jitter overrunning `T_hat`. `oracle_exit` removes
exactly that buffer (by construction it admits with the least possible true
margin over `T_hat`), which is consistent with its higher waste (0.89% vs
0.20%) — and it also admits more xavier (correctly recognizing more
borderline-feasible cases), shifting the mix toward the slower tier. Both
effects point the same direction: more accurate coverage information, with
an unchanged (imperfect) training-time belief, does not have to make the
system faster overall. This was not predicted and is reported as measured.
