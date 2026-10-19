# Stage 7b — fix the occupant binding, re-run. Audit first, then 2 runs, then 4.

The audit (`STAGE7_gpu_arm_audit.md`) is accepted in full: H3 — the `gpu` and `cpu`
predictors look up tier by `client_id` against the closed-loop fleet file, and in Stage 7
`client_id` is a slot. Predictor tier matched the occupying vehicle on 37.2% of
dispatches against a chance rate of 36.2%.

## Decisions taken

**1. Kill `cpu_open` and `cpu_closed` now.** Their results cannot be used.

**2. `p_recur = 1` means: identity is bound to the slot; the trace supplies only
coverage timing.** Each slot is a persistent vehicle — fixed tier, sleep, data partition
and server-side state, taken from the Stage 3–6 fleet (`1–10 xavier, 11–17 orin,
18–20 thor`) — and it re-enters coverage on the schedule of the trace windows that flow
through that slot. This is the original design of `p_recur` ("identity persistence
controlled at the PLATO layer"; "tier identity must be preserved on slot recycle").
SUMO motion is tier-agnostic (one `vType`), so assigning a slot's tier to any window
does not violate the physics.

At `p_recur = 0` identity comes from the trace vehicle, as now.

All four current `*_closed` runs were produced under the wrong semantics — a stranger's
state pasted onto a different vehicle — and are **invalid in every arm**, including
`baseline` and `observed`.

## Step 0 — one abstraction, not two patches

Introduce a single notion of **occupant identity** for each slot at each moment:
`(identity_id, tier, sleep_s, partition)`. At `p_recur = 0` it is the trace vehicle; at
`p_recur = 1` it is the slot's fixed fleet identity.

**Everything that currently keys on `client_id` must read the occupant identity
instead**, or state explicitly why it should stay slot-keyed. Do not fix the predictor
by wiring it to "the trace vehicle's tier" directly: that is correct at `p_recur = 0` and
wrong at `p_recur = 1`. One code path, both cases.

The five requirements listed under "Required for the re-run" in the audit all stand:
candidate scoring by occupant, busy-pool terms by the identity frozen at dispatch,
startup pool median from actual initial occupants, auditable log fields
(`identity_id`, tier used, the `tau_hat` actually used), and a pre-flight at 100% match.

## Step 1 — hunt for siblings of H3. Report before changing anything else.

H3 is a closed-loop assumption that survived into the open-loop server silently. Assume
there are others. List **every** piece of state or lookup in the Stage 7 path keyed on
`client_id`, and for each say: what it is, whether it should follow the slot or the
occupant, and what it currently does.

Two to check first:

- **Data partitions.** The Stage 7 prompt required `hash(veh_id) mod 20`. PLATO usually
  binds a datasource to `client_id` at initialisation. If partitions are in fact
  slot-bound, then `baseline_open` and `observed_open` also deviate from specification,
  and whether they must be re-run needs a decision. Report which it is, with evidence
  from the logs, not from the code alone.
- **Torch seeding in `sumo_open_loop`.** Stage 6 found the SUMO entry point never seeded
  torch. Confirm the Stage 7 server seeds `random`/numpy/torch and re-seeds per client
  from `(base_seed, identity, round)` — note *identity*, not `client_id`, for the same
  reason as everything else here.

**Stop after Step 1 and report.** If partitions turn out slot-bound, the set of runs to
redo changes, and that decision comes back here.

## Step 2 — pre-flight

Short smoke run (~100 rounds) of each arm at both `p_recur` values. Required, per
dispatch, all at 100%:

- predictor tier == occupant tier
- `sleep_s` == occupant's sleep
- partition == occupant's partition
- at `p_recur = 1`: the occupant of slot *k* has the same identity, tier and partition
  in every window for the whole run
- at `p_recur = 0`: `state_reset == recycles`
- `tau_hat` in the dispatch log equals the value the selector used

## Step 3 — the runs, in two waves

**Wave A — the thesis test. 2 runs.** `gpu_open`, `cpu_open`, `p_recur = 0`, seed 4126,
all other settings as Stage 7. `baseline_open` and `observed_open` are reused *only if*
Step 1 clears them.

Report Wave A before launching Wave B.

**Wave B — the closed control. 4 runs.** `baseline`, `gpu`, `cpu`, `observed` at
`p_recur = 1` under the new semantics.

## Written down before Wave A

Stage 7 established that a **single observation** is enough to classify a vehicle's tier
in this fleet (tiers are 15× apart with no overlap at CV 15%), so `observed` is blind
only on each vehicle's first dispatch — 37% of dispatches. Any proactive advantage is
confined to that channel.

| # | prediction |
|---|---|
| 1 | **On zero-observation dispatches**, `gpu` selects xavier markedly less often than `observed` does. Among never-seen candidates every one carries the same 10× `G_i` prior, so `gpu` ranks them by `s(τ̂)` (0.822 / 0.721 / 0.349) while `observed` gives all of them 0.995 and in effect picks at random. **This is the mechanism test and is read first** |
| 2 | On dispatches with ≥ 1 observation, `gpu` and `observed` have similar tier mixes — both know the tier by then |
| 3 | `gpu_open` finishes ahead of `observed_open` on cumulative time at round 4,000. The expected margin is modest, bounded by the first-dispatch channel, and may not clear the scoring arms' seed variance (`gpu` spread 4.88% in Stage 6) |
| 4 | `cpu_open` is slowest of the four open arms |

If 1 holds and 3 does not, report that plainly: the mechanism works as designed and the
channel it operates through is too narrow in this fleet to move the outcome. That is a
finding about the fleet's tier separation, not a failure of the estimator.

## Do not

- patch the predictor to read the trace vehicle's tier directly
- reuse any `*_closed` run
- reuse `baseline_open` or `observed_open` if Step 1 finds them affected
- change the GPU equation, `V_i`, the selection rule, sigma, the tier ratio, `M`, the
  trace, the dataset, the model or the optimiser
- launch Wave B before Wave A is reported
- tune anything to make a prediction come true
