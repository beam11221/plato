# Stage 11 — partial recurrence: where does the method's advantage turn over? Build, then one seed (17 runs), review, then two more seeds (34 runs).

**Do not launch any run before Stage 10 Wave B has finished** (GPU capacity, and Wave B
may change the choice of α). Step 0 and Step 1 (code and tests) may start now.

Do not touch anything under `plato/`.

---

## Why this stage exists — written before any result

Stage 10 Wave A found the full method (`gpu + F`, α = 1) against `baseline + F`:

| regime | 4126 | 4127 | 4128 |
|---|---|---|---|
| open (every arrival a new identity) | −15.2% | −13.3% | −9.4% |
| closed (identity bound to slot) | +11.2% | +5.7% | +3.9% |

The sign reverses between the two regimes in every seed. The mechanism is the `G_i`
feedback loop: under persistent identity the filter starves xavier, xavier's `G_i`
rises (xavier : thor 3.7–4.5× closed vs 1.7–2.3× open), and the score chases the
vehicles the filter excluded.

Real traffic at an RSU is neither regime. Commuters pass the same RSU repeatedly;
through traffic passes once. The question a reader will ask is: **at what share of
returning vehicles does the method stop paying?** This stage measures the curve between
the two endpoints.

---

## No new SUMO run is needed — and why

The trace supplies **only coverage timing** (when a window opens, how long it lasts,
the position inside it). Identity — tier, sleep, data partition, server-side state — is
assigned at the PLATO layer through the Stage 7b occupant abstraction. SUMO motion is
tier-agnostic (one `vType`). Stage 7b already runs the same trace as both endpoints by
changing only the identity policy. Partial recurrence is one more identity policy on
the same trace, `e1_open_poisson_seed4126_range400`, unchanged.

Do **not** regenerate the trace, change `M`, or touch the slot allocator. The occupancy
digest must stay identical to Stages 7b–10.

---

## Step 0 — the identity policy `p_return` (new knob; `p_recur` is left as it is)

`p_recur` (0 / 1) keeps its current meaning and code path. Add a separate knob:

| knob | meaning |
|---|---|
| `p_return` ∈ [0, 1] | probability that a **new coverage window** (a slot allocation) is occupied by a **returning commuter** rather than a **transient** vehicle |
| `commuter_pool` | the 20 fixed identities of the Stage 3–10 fleet (`clients_n20r400.csv`, `client_sleep_times_n20r400.csv`): 10 xavier / 7 orin / 3 thor, each with its fixed tier, sleep and partition. `K = 20`, fixed in this stage |

On every slot allocation:

1. draw `u` from the **identity RNG** (below). If `u < p_return` and at least one
   commuter is **not currently in coverage**, the occupant is a commuter drawn
   uniformly from those not in coverage. Otherwise the occupant is the **trace
   vehicle** of that window, as at `p_recur = 0` (own tier, sleep,
   `blake2b(veh_id) mod 20` partition)
2. position, direction, `R_i` and window timing always come from the trace vehicle
   physically in the slot, as in Stage 10
3. **server-side state follows identity.** A commuter keeps everything the server
   learned about it across absences: `G_i`, `times_selected`, the reactive arm's
   staleness history, and its `observed_per_vehicle` training-time history. A transient
   starts from the Stage 9 neutral prior, as in Stage 10's `p_recur = 0`
4. a commuter never occupies two slots at once

**The identity RNG must be seeded from the physics seed (4126), not the run seed.**
The identity sequence is part of the population, like the trace. It must be identical
across arms and run seeds at a given `p_return`. Log per allocation:
`slot_id, trace_veh_id, identity_id, kind ∈ {commuter, transient}, tier, state_carried`,
and report one **identity-sequence digest per `p_return` value**.

Notes to state in the report, not to change:

- commuters return within minutes of simulated time, not the next day. This is
  acceptable because no server-side quantity decays with elapsed time. Say so
- commuter pool mix 50/35/15 vs transient mix 49.1/34.9/16.0, so `p_return` does not
  shift the tier mix of arrivals. Verify it from the allocation log
- if all 20 commuters are in coverage when `u < p_return`, the window goes to the
  transient. Count these events. The realised commuter share must be reported next to
  the nominal `p_return`

### Tests (real selector, real aggregation strategy, as in 7b / 9 / 10)

- `p_return = 0`: identity sequence identical to Stage 10 `p_recur = 0`, and the
  identity RNG is never consumed in a way that changes any other stream
- `p_return = 1`: every occupant a commuter (apart from logged exhaustion events),
  state carried on every return
- for any `p_return`: a commuter's `G_i`, `times_selected`, `τ̂` history and training-time
  history after re-entry equal their values at its last exit; a transient's equal the
  neutral prior
- identity-sequence digest identical across two run seeds at the same `p_return`
- the filter, predictor and logs read tier and `T̂` from the occupant identity (the H3
  rule), for commuters and transients alike

## Step 1 — pre-flight, 100 rounds, seed 4126, `gpu + F` and `baseline + F` at each `p_return`

All Stage 10 pre-flight checks at 100%, plus:

- realised commuter share of allocations vs nominal `p_return`, and exhaustion events
- tier mix of arrivals by kind
- state carried == commuter re-entries; state reset == transient allocations
- occupancy digest identical to Stages 7b–10

Report, then wait for Stage 10 Wave B to finish before launching.

---

## Step 2 — the runs

All settings exactly as Stage 10 Wave A: filter on (`stage0_r_i = vmax_distance`),
`g_prior_alpha = 1`, `observed_unseen_tau = fleet_running_mean`, `C = 4`, Dirichlet
α = 0.8, 4,000 rounds, `coverage_gated`, startup `T_round` = 2.0808 s, physics seed 4126,
run seeds 4126 / 4127 / 4128.

If Stage 10 Wave B shows α = 10 beating α = 1 for `gpu + F` in the open population in
every seed, **stop and report before launching**: the α for this stage would then need
to be chosen again.

### The cell set — 17 cells per seed

| arm | `stage0_t_hat` | `p_return` | cells |
|---|---|---|---|
| gpu + F | `tier_spec` (GPU) | 0.25, 0.5, 0.75, **1.0** | 4 |
| baseline + F | `tier_spec` (GPU) | 0.25, 0.5, 0.75, **1.0** | 4 |
| observed + F | `observed_per_vehicle` | 0.25, 0.5, 0.75 | 3 |
| cpu + F | `tier_spec` (CPU) | 0.25, 0.5, 0.75 | 3 |
| baseline / none | — | 0.25, 0.5, 0.75 | 3 |

The endpoints are reused: `p_return = 0` is Stage 10 Wave A `p_recur = 0` (identical
identity sequence, per Step 0). The `p_return = 1.0` cells are an **anchor**: they check
that the commuter model at its endpoint lands where the slot-bound closed regime of
Stage 10 did. If it does, Stage 10's closed cells serve as the `p = 1` end of the curve.
If it does not, report the difference and use the anchor.

`baseline / none` reads no identity-dependent state, so it is expected to be flat
across `p_return`. It is a control for the harness. `cpu + F` completes the arm set.

### Phase 1 — the whole pipeline on seed 4126 only. 17 runs. Then STOP.

Run all 17 cells at run seed 4126, at most four at once, in this order:

1. `p_return = 0.5`: `gpu + F`, `baseline + F`, `observed + F`
2. `p_return = 0.25`: the same three
3. `p_return = 0.75`: the same three
4. anchor `p_return = 1.0`: `gpu + F`, `baseline + F`
5. `cpu + F` and `baseline / none` at 0.25 / 0.5 / 0.75

When all 17 have finished, write **`STAGE11_seed4126_report.md`**: the full Step 3 set for
one seed, the three gap curves, the linear-mixing comparison, `p*`, the anchor check, and
a verdict on each prediction labelled **provisional (1 seed)**.

**Then stop and wait.** Do not start seeds 4127 / 4128 until the file
`logs/stage11_phase2.go` exists; the user writes it after review. The seed-4126 result
may change what Phase 2 runs (for example, adding a `p_return` point near a crossing).
Any such change will arrive as an amendment to this prompt, not as an improvisation.

Rules for reading a single seed, to write into the report:

- a gap smaller than the ±3.5% execution noise is reported as "inside noise", not as a
  sign
- no claim is quoted from Phase 1. It exists to catch a wrong design early

### Phase 2 — seeds 4127 and 4128. 34 runs. Only after the go-file.

The same 17 cells at run seeds 4127 and 4128, in the same order, seed 4127 finishing
its first three groups before seed 4128 starts. Then write `STAGE11_report.md` with the
three-seed analysis below.

**Total: 51 runs** (17 + 34).

---

## Step 3 — what to report

Per run, the Stage 10 set. Also, **split by occupant kind (commuter / transient)**:

- dispatch share per tier, per window
- median `G_i` at dispatch per tier, rounds 2,001–4,000, and the xavier : thor ratio
- prior observations held for the identity at dispatch (the information state):
  distribution and mean. Relate it to the Stage 3 k\* ≈ 30–70
- ever-dispatched per tier and never-dispatched, for transients and commuters separately
- share of all dispatches going to commuters, against their share of allocations

### The curve

For each arm: T @ 4,000 against `p_return` ∈ {0, 0.25, 0.5, 0.75, 1}, each seed as a
point. Then the three gap curves, per seed:

| # | gap | question |
|---|---|---|
| **A** | `gpu + F` − `baseline + F` | **where does it cross zero?** (the headline) |
| B | `gpu + F` − `observed + F` | how fast does the reactive arm catch up as identities recur? |
| C | `observed + F` − `baseline + F` | does reactive scoring ever pay on top of the filter? |

For A, also report the **linear mixing line** between the two endpoints of each seed,
`(1 − p)·gap(0) + p·gap(1)`, and the measured gap's departure from it at each `p`.
Report the crossing `p*` per seed by linear interpolation between the adjacent points
that bracket zero, or "no crossing".

Checked in **each seed individually**. A sign that flips between seeds at a given `p` is
undecided. Differences against a baseline arm must clear the ±3.5% execution noise
accepted in Stage 10.

---

## Written down before the runs start

| # | prediction |
|---|---|
| 1 | **The gap A is monotone in `p_return`** in every seed: from −9 to −15% at `p = 0` to +4 to +11% at `p = 1` |
| 2 | **Worse than linear mixing.** The measured gap A lies above the linear line (worse for `gpu`) at `p = 0.25, 0.5`. Commuters carry an inflated xavier `G_i` that also outranks transients, so the antagonism contaminates the transient share too. The linear line crosses zero near `p ≈ 0.65`, so **`p*` is predicted in [0.4, 0.65]** |
| 3 | **The mechanism is visible in the commuters.** Their xavier : thor `G_i` ratio rises with `p_return` and exceeds the transients' at every `p > 0` |
| 4 | **`baseline + F` is flat** across `p_return`: within ±3.5% of its `p = 0` value in every seed. The filter reads tier, not history |
| 5 | **Gap B shrinks** monotonically with `p_return` (the reactive arm learns the commuters) and is undecided by `p = 0.75` |
| 6 | Gap C is never negative beyond noise at any `p`: reactive scoring does not pay on top of the filter |
| 7 | The `p_return = 1` anchor lands within ±3.5% of Stage 10's closed cells for both arms |
| 8 | Plateau stays within 81.7–83.6% everywhere |

**Prediction 2 is the point.** If the curve is linear or better (`p* ≥ 0.65`), the method
tolerates a majority of returning traffic. If `p*` is low, the claim has to be
restricted to transit-dominated RSUs, and a remedy for the `G_i` loop becomes the next
piece of work. Either answer is reported as it comes.

---

## Stop conditions

- any Step 0 test or Step 1 check below 100%
- identity-sequence digest differs between runs at the same `p_return`, or occupancy
  digest differs from Stages 7b–10
- realised commuter share departs from nominal `p_return` by more than 2 points (report
  exhaustion events)
- fallback > 10% of rounds, or median `eligible_after_stage0` < `C = 4`
- conservation-law error > 3%, or plateau below 80.1%

## Do not

- re-run SUMO, regenerate the trace, or change `M`, the allocator, positions or `v_max`
- seed the identity draw from the run seed
- change what `p_recur` means, or re-run any Stage 10 cell
- vary `K`, the commuter pool's composition, or add any time decay to server state
- change `V_i`, the selection rule, the GPU equation, the filter, `T̂` sources, α,
  startup `T_round`, sigma, tier ratio, dataset, model or optimiser
- add `p_return` values beyond 0.25 / 0.5 / 0.75 / 1.0
- launch before Stage 10 Wave B is finished
- start seed 4127 or 4128 before `logs/stage11_phase2.go` exists
- quote any Phase 1 result as a claim
- report the headline gap without the per-seed signs and the fairness numbers by kind
- tune anything to make a prediction come true
