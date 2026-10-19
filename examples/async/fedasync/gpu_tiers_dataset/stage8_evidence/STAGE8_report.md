# Stage 8: seeds 4127 and 4128 for C3

Spec: `prompts/STAGE8_c3_seeds.md`. Code: Stage 7b as it stands, plus the one
change the spec allows (physics/run seed split, §1).

## Status

| step | state |
|---|---|
| Step 1: split seeds, pre-flight (seed 4127, `gpu_open` + `gpu_closed`, 100 rounds) | **PASSED, 100%** (§2) |
| Step 2: 12 runs | **done**, 0 failures (§3) |
| Step 3: report | **done** (§5) |

---

## Seed ledger. Read this before touching any file

| seed | role | what it sets | where |
|---|---|---|---|
| **4126** | PHYSICS, pinned in all 18 runs | every trace vehicle's sleep `vehicle_sleep_s(4126, veh_id, tier)`, and the slot allocator's RNG | `[sumo_open] physics_seed = 4126` (Stage 8 configs). The 7b configs have no key and fall back to `[clients] random_seed = 4126` |
| **4126 / 4127 / 4128** | RUN seed | process seeding, `identity_seed(run, identity, round)` (torch/numpy/random per training run), server dispatch randomness, and the Dirichlet label draw per partition | `[clients]`, `[server]` and `[data] random_seed`, all equal, as in Stage 6 |

Never varied: the trace `e1_open_poisson_seed4126_range400`, the partition
**index** `blake2b(veh_id) mod 20`, the `p_recur = 1` fleet
(`clients_n20r400.csv`, `client_sleep_times_n20r400.csv`), and startup
`T_round`.

| seed | cells | configs | run dirs / logs | ports |
|---|---|---|---|---|
| 4126 | 6 (Stage 7b, reused as is) | `configs/stage7b/` | `stage7b_<arm>_<regime>` | 18660– |
| 4127 | 6 | `configs/stage8_s4127/` | `stage8_s4127_<arm>_<regime>` | 18700– |
| 4128 | 6 | `configs/stage8_s4128/` | `stage8_s4128_<arm>_<regime>` | 18720– |

The seed is part of every directory, log and config name, so no two seeds
can share a file. Arms: `baseline`, `gpu`, `observed` (`cpu` excluded by the
spec).

---

## 1. The seed split (the one code change)

Coupling confirmed by reading the code. `sumo_open_fedasync.main()` read
`[clients] random_seed` once and passed that same value both to
`IdentityResolver` (→ `vehicle_sleep_s`, i.e. physics) and, through
`client_seeded_trainer.base_seed()`, to `identity_seed` (training). Setting
the run seed to 4127 would have silently redrawn every `p_recur = 0`
vehicle's sleep.

The fix is in `sumo_open_fedasync.py` `main()`: a new optional
`[sumo_open] physics_seed` feeds `OpenCoverageTimeline`, `IdentityResolver`
and the server's `base_seed`. `[clients] random_seed` stays the run seed. If
the key is absent, `physics_seed` = run seed, so the Stage 7 / 7b configs
behave byte-identically (regenerated 7b configs diff clean; 45/45 tests pass).
The entry point logs both:

```
[Stage8] seeds: run_seed=4127 (...training, process, dispatch) physics_seed=4126 (...vehicle sleeps, allocator)
```

Tooling added, not changed: `make_stage7_configs.py --stage 8 --seed N`
(configs; refuses 4126 and refuses `cpu`), `run_stage7.sh --stage 8_sN`
(already generic), and `preflight_stage8.py`, a new file that imports the
Stage 7b checker unchanged. The analysis code is unchanged. Cross-seed
aggregation goes in a separate script that only reads its outputs.

**Assumption to note: data partitions.** The spec pins the partition as
`blake2b(veh_id) mod 20`, i.e. the index, and that is pinned. The Dirichlet
label mix *inside* each partition comes from `[data] random_seed`, which
follows the run seed here, as it did in every Stage 6 seed. So each vehicle
keeps its partition index, but partition *k*'s label draw differs per seed
(fingerprints equal to 7b: 0/18 and 0/20, §2). If the intent was to pin the
label draw too, `[data] random_seed` would have to stay 4126. That is a config
value, not code.

---

## 2. Step 1: pre-flight (`run_seed = 4127`, 100 rounds)

`preflight_stage8.py --seed 4127 gpu_open gpu_closed` →
`stage8_evidence/preflight_s4127.json`. **Every check 100%.**

| check | gpu_open | gpu_closed |
|---|---|---|
| all Stage 7b checks (identity, tier, predictor tier, sleep ×2, partition ×2, sampler id, fingerprints, `tau_hat` used, observations, G_i prior, resets / one identity per slot, startup `T_round`, gpu `tau_hat` = Stage 3–6 values) | 100% | 100% |
| **physics pinned**: sleep = 7b (seed 4126) value, dispatch and report | 109/109, 109/109 | 114/114, 114/114 |
| **physics pinned**: `p=1` slot identity / tier / partition = 7b fleet | — | 20/20 |
| **seed applied**: seed = `identity_seed(4127, …)` | 109/109 | 114/114 |
| **seed changed**: seed ≠ `identity_seed(4126, …)` for the same (identity, round) | 109/109 | 114/114 |
| **seed changed**: first-round train_loss ≠ 7b | 1.7894 vs 1.8105 | 1.6732 vs 2.0066 |
| entry point logs run = 4127, physics = 4126 | yes | yes |
| partition label draw equal to 7b (informational) | 0/18 | 0/20 |

Physics check and seed check both hold, so the seed was applied and the
population was not redrawn.

---

## 3. Run ledger

Driver: `sumo_open_loop/run_stage8.sh` (log `logs/run_stage8.log`). At most
four runs at once. Each wave's `checkpoints/` is deleted when the wave ends.

| wave | runs | started (UTC) | finished |
|---|---|---|---|
| 1 | s4127: baseline_open, gpu_open, observed_open, baseline_closed | 2026-09-24 16:52 | 2026-09-24 22:24 |
| 2 | s4127: gpu_closed, observed_closed · s4128: baseline_open, gpu_open | 2026-09-24 22:24 | 2026-09-25 04:01 |
| 3 | s4128: observed_open, baseline_closed, gpu_closed, observed_closed | 2026-09-25 04:01 | 2026-09-25 09:36 |

Wave 1 at start: all four log `run_seed=4127 physics_seed=4126`, with no
tracebacks.

---

## 4. Analysis tooling (ready before the runs finish)

`sumo_open_loop/analyze_stage8.py` imports `analyze_stage7` **unchanged** and
calls its `parse` and metric functions for each seed, pointing it at the right
prefix (4126 → `stage7b`, 4127 → `stage8_s4127`, 4128 → `stage8_s4128`). It
adds, reading only:

- per-run median `G_i` by tier, rounds 2,001–4,000, ≥ 1 observation;
- the occupancy digest across all 18 runs;
- per-cell mean and SD across seeds;
- claims (a)–(g) checked **per seed**. A sign that flips in any seed is
  reported as "undecided";
- C2.

The crossing round (g) is computed per round, not at checkpoints:
`stays_behind_from` is the round after which the closed Polaris arm stays
behind `baseline` to 4,000.

Validated on seed 4126 alone, where it reproduces the Stage 7b figures exactly
(a: +13.16%, d: −6.49 / −4.65%, e: +5.49 / +8.46%, `G_i` xavier : thor open →
closed 1.37 → 3.06 (`gpu`), 1.55 → 2.90 (`observed`)). The 7b report placed
the closed crossing at "≈2,000", which was checkpoint resolution. Per round,
it is **1,690 (`gpu`) and 1,531 (`observed`)**.

---

## 5. Results: 18 runs (3 arms × 2 regimes × seeds 4126 / 4127 / 4128)

`analyze_stage8.py` → `stage8_analysis.json`.

### Soundness, and the physics pin re-checked on every full run

- Conservation error −0.078 … +0.098%; plateau 82.12 – 83.40%, all 18 inside
  the band; predictor tier agreement 100% in every `gpu` run; all 18 reached
  round 4,000.
- **Occupancy invariant: one digest across all 18 runs** (common prefix of
  1,891 recycles).
- `preflight_stage8.py` was re-run on **all 12 full-length runs**
  (`fullrun_checks_s4127.json`, `fullrun_checks_s4128.json`): every check
  100%. This includes sleep = the 7b value on every dispatch, training seed =
  the run seed and ≠ 7b, and first-round loss ≠ 7b. Seed 4128 had not been
  pre-flighted; it is now verified on the full runs.

### T @ 4,000 (s)

| cell | 4126 | 4127 | 4128 | mean ± SD |
|---|---|---|---|---|
| baseline_open | 7238.0 | 7251.2 | 7110.3 | 7199.9 ± 77.8 |
| baseline_closed | 7260.4 | 7339.3 | 7265.1 | 7288.2 ± 44.3 |
| gpu_open | 6768.6 | 6689.9 | 6721.7 | 6726.7 ± 39.6 |
| gpu_closed | 7659.3 | 8021.7 | **7108.4** | 7596.5 ± **459.9** |
| observed_open | 6901.5 | 6706.3 | 6800.8 | 6802.9 ± 97.6 |
| observed_closed | 7874.3 | 8159.9 | **7290.1** | 7774.8 ± **443.4** |

| cell | waste | xavier share | `G_i` xavier : thor (r 2001–4000) |
|---|---|---|---|
| baseline_open | 13.22 ± 0.16% | 34.6 ± 0.5% | — |
| baseline_closed | 13.58 ± 0.18% | 37.2 ± 0.3% | — |
| gpu_open | 8.85 ± 0.35% | 33.7 ± 0.2% | 1.48 ± 0.11 |
| gpu_closed | 13.41 ± 0.66% | 39.2 ± 2.8% | 2.89 ± 0.18 |
| observed_open | 8.35 ± 0.37% | 34.4 ± 0.5% | 1.48 ± 0.06 |
| observed_closed | 13.94 ± 0.57% | 40.2 ± 2.8% | 2.76 ± 0.14 |

The closed Polaris cells vary about 6% across seeds (SD 460 / 443 s). Every
other cell varies 0.6–1.4%.

### The C3 claims, per seed

| # | claim | 4126 | 4127 | 4128 | verdict |
|---|---|---|---|---|---|
| a | `gpu`: open faster than closed (closed − open) | +13.16% | +19.91% | +5.75% | **held** |
| b | `observed`: open faster than closed | +14.10% | +21.67% | +7.19% | **held** |
| c | `baseline`: closed − open | +0.31% | +1.21% | +2.18% | see below |
| d | open: `gpu` vs `baseline` | −6.49% | −7.74% | −5.47% | **held** |
| d | open: `observed` vs `baseline` | −4.65% | −7.51% | −4.35% | **held** |
| e | closed: `gpu` vs `baseline` | +5.49% | +9.30% | **−2.16%** | **undecided** |
| e | closed: `observed` vs `baseline` | +8.46% | +11.18% | +0.34% | held (4128 a near-tie) |
| f | `gpu` `G_i` x:t open → closed | 1.37 → 3.06 | 1.47 → 2.71 | 1.60 → 2.90 | **held** |
| f | `observed` `G_i` x:t open → closed | 1.55 → 2.90 | 1.43 → 2.77 | 1.47 → 2.61 | **held** |
| g | closed `gpu` stays behind `baseline` from round | 1,690 | 1,778 | **never** | — |
| g | closed `observed` stays behind `baseline` from round | 1,531 | 1,760 | **3,902** | — |

**(c) in detail.** `baseline`'s open–closed gap is +0.31 / +1.21 / +2.18%
(mean 1.23%). Its own seed SD is 1.08% (open) and 0.61% (closed), so the gap
is about 1–2 of its SDs. In every seed it is smaller than both Polaris arms'
gaps, but the margin varies: roughly 40× in seed 4126, 16× in 4127, and only
**2.6×** against `gpu` in 4128 (2.18% vs 5.75%). The control holds in
direction everywhere. It is an order of magnitude smaller in two seeds out of
three, not in all three.

**What happens in seed 4128.** The `G_i` divergence is there (xavier : thor
2.90 / 2.61), but the absolute xavier `G_i` is lower (1.07 / 1.04 vs
1.15–1.33 in the other seeds). The closed Polaris arms then dispatch much less
extra xavier: 36.1% / 37.1%, against 39.9–42.5% in seeds 4126 / 4127 and
37.0% for `baseline`. The xavier share of `gpu_closed` by window, 1,000–4,000:
4126 42.2 / 41.2%, 4127 44.5 / 44.2%, 4128 36.9 / 37.9%. So the ratio alone
does not fix how far the selector leans into the slow tier, and in one seed
out of three the lean is too small for `gpu` to fall behind random selection.

### Predictions (written before the runs)

| # | prediction | verdict |
|---|---|---|
| 1 | (a) and (b) hold in all seeds, gaps ~10–15% | **direction held in all seeds; magnitude wider than predicted** (5.8–21.7%) |
| 2 | (c): `baseline` gap within ~2% in every seed, an order of magnitude below Polaris | **partly held**: 0.31 / 1.21 / 2.18% (the last just over 2%); order of magnitude in 2 of 3 seeds, 2.6× in 4128 |
| 3 | (d) and (e) hold in all seeds: the sign of "Polaris vs random" flips between regimes | **NOT HELD for `gpu`** (closed 4128: −2.16%, ahead of `baseline`); held for `observed`, but 4128 is +0.34%, effectively a tie |
| 4 | (f) holds in every seed | **held** |
| 5 | closed crossing round in ~1,500–2,500 | **not held**: in range for 4126 / 4127 (1,531–1,778), but 4128 gives `observed` 3,902 and `gpu` never |
| 6 | `gpu − observed` changes sign in at least one regime (C2 null) | **not held**: the sign is the same in all 6 seed × regime cells (open −1.93 / −0.24 / −1.16%, closed −2.73 / −1.69 / −2.49%). The magnitudes stay small (0.24–2.73%), and no claim rests on C2 |

### What C3 can and cannot say after three seeds

**Supported in every seed:**

- Polaris arms are faster in the open population than in the closed one
  (a, b).
- In the open population they beat random selection (d: −4.35 to −7.74%).
- The closed-pool `G_i` divergence appears in every seed (f).
- `baseline` barely moves between regimes (c), so the effect comes from
  identity persistence, not from the trace or the fleet.

**Not supported as a universal statement:** "the ranking against random
selection *reverses* in the closed population" (e). It reverses for `gpu` in
2 of 3 seeds and for `observed` in 3 of 3, with one of those a near-tie
(+0.34%). The defensible wording on these three seeds is: **closed-population
evaluation removes the Polaris arms' advantage over random selection** (from
−4 to −8% open, to between −2.2% and +11.2% closed), **and in most seeds turns
it into a deficit.** The closed Polaris cells are also 4–10× noisier across
seeds than any open cell, so any single closed-pool result is a poor estimate
of the arm.
