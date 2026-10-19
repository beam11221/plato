# Stage 10 — the full method in the open population: Stage 0 filter + scoring. Audit, then 30 + 18 runs.

Stages 7b–9 ran with Stage 0 **off**, on purpose: they tested the scoring claim alone.
Stage 5–6 tested the filter alone, and only in the SUMO closed loop. The method the thesis
proposes uses the GPU spec twice — as `T̂` in the Stage 0 filter and as `s(τ̂)` in the
score — and **that combination has never been run in the open population.** This stage
runs it, with every arm, both regimes, three seeds.

Do not touch anything under `plato/`.

---

## Why this is a real question — written before any result

The two halves are known separately:

| | closed | open |
|---|---|---|
| filter only (random selector + spec filter) | −39.4% vs random (Stage 6, SUMO closed loop) | **not run** |
| scoring only (`gpu`, α = 1) | +5.3% vs random (Stage 9) | −20.6% vs random (Stage 9) |
| filter + scoring | `gpu` + filter **slower** than random + filter by +3.9 / +14.7 / +10.6% (Stage 6) | **not run** |

Stage 6 found the two **work against each other**: the filter starves xavier, `G_i`
rises on the starved clients, and `V_i` then chases exactly the vehicles the filter
excluded. Stage 8 found that identity churn caps the `G_i` divergence in the open
population (xavier : thor 1.4–1.6× open vs 2.6–3.1× closed). So the open population
might weaken the antagonism and let the two effects add, or `gpu` α = 1 and the filter
might remove the same xavier dispatches and overlap. Which one happens is not known.

Stage 9 also found that α = 10 lowers waste in the open population (8.4–10.1% vs
random 13.2%) because "newest first" acts as an implicit residency filter. With a real
filter on, that advantage should disappear, so the α comparison has to be re-read with
the filter present.

---

## Step 0 — audit the open-loop server's Stage 0 path. Report before any run.

The Stage 0 filter was built and validated on the SUMO closed-loop entry point. The
open-loop server (`sumo_open_loop`, occupant identity from Stage 7b) may not have it
wired, or may have it wired with a `client_id` lookup — the H3 failure mode from Stage 7.
Assume a sibling of H3 exists until shown otherwise.

Report, for the open-loop path:

1. whether Stage 0 exists there at all, and where
2. every lookup the filter makes — `T̂` source, tier, position, coverage boundary — and
   whether it keys on the **occupant identity** or on `client_id`. Everything must read
   the occupant, in both `p_recur` modes, through the same abstraction as 7b
3. where `R_i = |x_exit_boundary − x_i| / v_max` gets `x_i` in the open trace, and
   whether the trace carries positions at the resolution Stage 5b used. If it does not,
   **stop and report** — do not substitute an elapsed-time formula
4. that the gate order is unchanged from Stage 5:
   `eligible → busy → in-range-now → Stage 0 (R_i ≥ T̂_i) → score → lottery`,
   with `T_comm = 0` and the Stage 5 fallback rule (empty pool → unfiltered pool for
   that round, log `stage0_fallback`)

Fix only what the audit finds, through the occupant abstraction. Add tests.

## Step 1 — one new `T̂` source for the reactive arm

A reactive server has no spec, but it can observe how long each vehicle took. Stage 6
gave `observed` a single fleet-mean `T̂` for every vehicle, which throws that away. The
fair reactive filter uses the vehicle's own observed training time:

| `stage0_t_hat` | value |
|---|---|
| `tier_spec` (existing) | tier-mean `T̂` from the arm's own spec file — GPU spec for `gpu` and `baseline`, `tiers_cpu.csv` for `cpu` |
| **`observed_per_vehicle`** (new) | for a vehicle with ≥ 1 completed dispatch: the mean of its own realised training times (`return − dispatch`, `T_comm = 0`). For a never-seen vehicle: the **fleet running mean** of all realised training times so far, initialised at the fleet mean 8.323077 s before the first report |

Notes:

- a wasted dispatch (vehicle left coverage before returning) produces **no
  observation** — the server never learns its duration. Do not impute one
- the never-seen rule is the filter's analogue of Stage 9's
  `observed_unseen_tau = fleet_running_mean`, and it is used in **every** `observed` cell,
  α = 1 and α = 10 alike. It is part of the filter, not of the Polaris prior
- default stays as before; existing configs must behave exactly as before. Test it

## Step 2 — pre-flight, 100 rounds per filtered arm, both regimes, seed 4126

All Stage 7b and Stage 9 pre-flight checks at 100%, plus, per dispatch and per
rejection:

- filter's tier == occupant tier; `T̂` used == the value the source above defines at
  that moment (for `observed_per_vehicle`: equals the occupant's own running mean, or
  the fleet running mean if never seen)
- `R_i` from the occupant's position at that round
- `T̂`, `R_i`, pass/reject logged == what the gate actually used
- `eligible_after_stage0` median ≥ `C = 4`, fallback < 10% of rounds

Report pool sizes after each gate for open and closed. The open population has ~13
vehicles in coverage, fewer than the Stage 6 cell, so the fallback rate is the first
thing to look at.

Report, then launch.

---

## Step 3 — the runs, in two waves

Same as Stage 9 in everything not listed: same code, physics seed pinned at 4126, run
seed 4126 / 4127 / 4128, trace `e1_open_poisson_seed4126_range400`, `M = 20`, `C = 4`,
Dirichlet α = 0.8, 4,000 rounds, `coverage_gated`, startup `T_round` = 2.0808 s,
`stage0_r_i = vmax_distance`.

### Wave A — priority. α = 1 and α-independent cells. 30 runs.

| arm | Stage 0 | `stage0_t_hat` | `g_prior_alpha` / unseen `τ̂` | `p_recur` | seeds | runs |
|---|---|---|---|---|---|---|
| baseline | **off** (pure random) | — | — | 0, 1 | 3 | 6 |
| baseline | on | `tier_spec` (GPU) | — | 0, 1 | 3 | 6 |
| gpu | on | `tier_spec` (GPU) | 1 / — | 0, 1 | 3 | 6 |
| observed | on | `observed_per_vehicle` | 1 / `fleet_running_mean` | 0, 1 | 3 | 6 |
| cpu | on | `tier_spec` (CPU) | 1 / — | 0, 1 | 3 | 6 |

Launch order, at most four at once, so the headline arrives first:

1. open: `baseline + filter`, `gpu + filter` (6 runs)
2. open: `observed + filter`, `cpu + filter`, `baseline / none` (9 runs)
3. closed: all five arms (15 runs)

`baseline / none` is re-run deliberately even though Stage 7b/8 cells exist: it puts a
pure-random control inside the same batch and gives a second measurement of run-to-run
execution noise (Stage 9 found 1.29%). Keep the 7b/8 cells too and report both.

When Wave A finishes, write `STAGE10A_report.md` with the Wave A comparisons below,
then **start Wave B without waiting**, unless a stop condition has tripped.

### Wave B — after Wave A. α = 10 with the filter. 18 runs.

| arm | Stage 0 | `stage0_t_hat` | `g_prior_alpha` / unseen `τ̂` | `p_recur` | seeds | runs |
|---|---|---|---|---|---|---|
| gpu | on | `tier_spec` (GPU) | 10 / — | 0, 1 | 3 | 6 |
| observed | on | `observed_per_vehicle` | 10 / `const_0.01` | 0, 1 | 3 | 6 |
| cpu | on | `tier_spec` (CPU) | 10 / — | 0, 1 | 3 | 6 |

Wave B uses the Polaris default prior package exactly as in Stage 7b/8. The filter's own
never-seen rule does not change.

**Total: 48 runs.** No-filter partners of the scoring arms are reused from Stage 9
(α = 1) and Stage 7b/8/9 (α = 10) — do not re-run them.

---

## Step 4 — what to report

Per run, the Stage 9 set (T at rounds 100 / 250 / 500 / 1,000 / 2,000 / 4,000, waste,
dispatch share per tier per window, TTA 70/75/78/80, plateau, membership and
never-dispatched per tier, first-dispatch tier mix, Jain's index, `G_i` at dispatch by
tier and seen / never-seen per window), plus the Stage 5–6 filter set:

- dispatches attempted, rejected by Stage 0, fallback rate
- pool sizes after each gate, median / p10 / p90, per window
- rejection accuracy per tier: fraction of rejections that would have completed in range
- of admitted dispatches, fraction wasted, per tier
- eligibility-window fraction per tier
- **waste / mix channel decomposition** of each filtered run against `baseline / none`
  of the same seed and regime, using the conservation law with empirical tier sleeps
  14.37 / 1.78 / 1.07 s
- for `observed`: the fleet running-mean `T̂` over time, and the share of dispatches
  whose `T̂` came from the never-seen rule, per tier
- **the antagonism mechanism:** median `G_i` at dispatch, xavier : thor, rounds
  2,001–4,000, for each filtered Polaris arm next to its no-filter partner

### The comparisons

Every comparison paired by seed and checked in **each seed individually**. A sign that
flips in any seed is reported as undecided, never as a mean ± SD that covers both signs.

| # | comparison | regime | wave |
|---|---|---|---|
| **A** | `gpu + F` vs `baseline + F`, α = 1 — **the headline** | open | A |
| B | same | closed | A |
| C | `gpu + F` vs `observed + F`, α = 1 — C2 with the filter on | open, closed | A |
| D | each filtered arm vs `baseline / none`, with the waste / mix split | both | A |
| E | `baseline + F` vs `baseline / none` — does C1 replicate in the open population? | both | A |
| F | each scoring arm, filter on vs its Stage 9 no-filter partner — additive or sub-additive? | both | A |
| G | `cpu + F` vs `gpu + F` | both | A |
| H | `baseline / none` re-run vs Stage 7b/8, same seed | both | A |
| I | each filtered arm, α = 1 vs α = 10 | open, then closed | B |

For **F**, state additivity explicitly: `(filter-only gain) + (scoring-only gain)` against
the measured combined gain, per seed.

---

## Written down before the runs start

| # | prediction |
|---|---|
| 1 | **C1 replicates in the open population.** `baseline + F` beats `baseline / none` by 30–42% in every seed, both regimes, with a waste channel of about −12 to −13% |
| 2 | **The headline (A).** `gpu + F` beats `baseline + F` in the open population in all three seeds, but by less than Stage 9's −20.6% without the filter: the combination is sub-additive because both remove xavier dispatches |
| 3 | **Closed (B).** `gpu + F` is slower than `baseline + F` in at least two of three seeds, as in Stage 6 — the antagonism survives under persistent identity |
| 4 | **Mechanism.** The filtered `gpu` arm's `G_i` xavier : thor ratio is larger than its no-filter partner's in the closed regime and close to it in the open regime |
| 5 | **C2 with the filter (C).** `gpu + F` beats `observed + F` in the open population in all three seeds, and `observed + F` has the higher waste of the two in every seed: on a first dispatch it filters a xavier newcomer with the fleet running mean (~8 s), not ~14 s, and admits doomed dispatches the spec would reject |
| 6 | `cpu + F` is the slowest filtered arm in both regimes; its waste channel is intact and its mix channel is ≈ 0 or positive, as in Stage 6 |
| 7 | **α under the filter (I).** `gpu` α = 1 beats `gpu` α = 10 with the filter in the open population in all three seeds; α = 10's waste advantage from Stage 9 disappears (waste within 1 point of α = 1) |
| 8 | `baseline / none` re-runs land within 2% of Stage 7b/8 in every seed (H) |
| 9 | **Fairness price.** Xavier never-dispatched under `gpu + F` in the open population exceeds Stage 9's 12–15%. Reported next to prediction 2, never alone |

**Prediction 2 is the point.** If it fails — `gpu + F` ties or loses to `baseline + F` in
the open population — report it plainly: the filter carries the method, and the score
adds nothing once the filter is on. That is a clear thesis result, not a failure.

---

## Stop conditions

- any Step 2 check below 100%, or Step 0 finds positions unavailable in the open trace
- `stage0_fallback` in more than 10% of rounds, or median `eligible_after_stage0` below
  `C = 4`, in any run
- the occupancy digest differs from 7b/8/9 in any run
- conservation-law error > 3%
- plateau below 80.1% in any run
- `baseline / none` re-run differs from 7b/8 by more than 3% in any seed — stop Wave B
  and report; that would mean the harness has drifted since Stage 9

## Do not

- change `V_i`, the selection rule, the GPU equation, startup `T_round`, sigma, tier
  ratio, `M`, the trace, physics seed, dataset, model or optimiser
- change `stage0_r_i` away from `vmax_distance`, the fallback rule, or `T_comm`
- give `observed` any spec information, including in the filter
- use a `T̂` source for `cpu` other than its own CPU spec
- re-run the Stage 9 or Stage 7b/8 no-filter scoring cells
- start Wave B before Wave A has finished and `STAGE10A_report.md` is written
- try any α other than 1 and 10
- report the headline without the fairness numbers next to it
- tune anything to make a prediction come true
