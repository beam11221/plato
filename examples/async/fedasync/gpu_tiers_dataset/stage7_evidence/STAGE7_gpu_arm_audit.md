# Stage 7 — audit of the `gpu` arm's estimator

Date: 2026-09-23. Scope: the four checks requested before continuing analysis,
plus the follow-up question *"how can the GPU equation predict wrong when every
vehicle reports its GPU-TOPS?"*

**Verdict: H3.** The `gpu_tops` predictor is live and its arithmetic is correct,
but it is keyed on `client_id`, which in Stage 7 is a **slot**, not a vehicle.
It reads a static 20-row fleet table that has not described reality since the
first recycle. Its tier label matches the vehicle actually training in **37%**
of dispatches — exactly the chance rate.

Nothing here has been changed or re-run. This is a report only.

---

## 1. Does the `gpu` arm actually use `gpu_tops`? — **yes, not H1**

It did not fall through to the `else` (observed) branch. Each arm declares its
predictor at startup:

| run | declared predictor |
|---|---|
| `gpu_open`, `gpu_closed` | `gpu_tops (proactive, T_round recomputed each round from live busy pool)` |
| `observed_open`, `observed_closed` | `observed-τ (rounds, default)` |
| `cpu_open` | `cpu (proactive, ...)` |

`[GpuTierSleep]` lines appear in **every** arm including `baseline`, so their
presence proves nothing — that tag belongs to the client-side `sleep_time`
assignment (the physics), not to the estimator. There is no `T_pred` string in
the codebase; the predictor's output surfaces as
`[PolarisVi] Staleness prediction accuracy: ... predicted=`.

`gpu_open` emits genuinely continuous predictions, but only **three distinct
values** across all 4,000 rounds:

| value | = | tier |
|---|---|---|
| 7.2089 | 15.000 / 2.081 | xavier |
| 0.9242 | 1.923 / 2.081 | orin |
| 0.4806 | 1.000 / 2.081 | thor |

(`observed_open` by contrast emits 23 distinct values: `0.0100`, `0`, `1`, `2`, …)

## 2. Is the tier bound to the slot or to the vehicle? — **to the slot (H3)**

Every `cid` receives exactly **one** predicted value for the entire run:

| cid (slot) | predicted | implied tier |
|---|---|---|
| 1–10 | 7.2089 | xavier |
| 11–17 | 0.9242 | orin |
| 18–20 | 0.4806 | thor |

Joined against the real `tier` of the `veh_id` occupying that slot at that
moment (from the `[Stage7] dispatch` line):

| run | predictor tier == vehicle's tier | **wrong** |
|---|---|---|
| `gpu_open` | 1743 / 4684 = **37.2%** | **62.8%** |
| `gpu_closed` | 1720 / 4659 = **36.9%** | **63.1%** |

Slot 1 — predictor says `xavier` on every one of its dispatches; the vehicles
that actually occupied it were `xavier 78, orin 56, thor 28`.

**37% is chance.** With the slot map at 10/7/3 (= .50/.35/.15) and the realised
dispatch mix at 34.8/45.2/20.1, blind agreement is
`.50(.348) + .35(.452) + .15(.201) = 36.2%`. Measured 37.2%. The `gpu` arm's
signal carries **no information about the vehicle it is scoring**.

## 3. Is `s_tau` in `stage7_analysis.json` computed from what `gpu` used? — **no; those numbers are withdrawn**

`analyze_stage7.py:304` computes `s(τ̂)` from `d.tau_hat`, parsed from the
`tau_hat=` field of the `[Stage7] dispatch` line. That field carries the
**observed-τ array**, not the value the `gpu` selector used:

- `tau_hat` in `gpu_open` dispatch lines: `{0.01, 0, 1, 2, 3, 4, …}` —
  indistinguishable in character from `observed_open`
- what `gpu` actually used: `{7.2089, 0.9242, 0.4806}` and nothing else

Recomputed from the real selector input, `s(τ̂) = 1/√(1+τ̂)`:

| tier | reported earlier (wrong array) | **`gpu`'s real value** |
|---|---|---|
| xavier | 0.7092 (median 0.9950) | **0.3490** |
| orin | 0.8460 (median 0.9950) | **0.7209** |
| thor | 0.9099 (median 0.9950) | **0.8218** |

**Consequence:** the earlier Prediction 4 verdict ("mechanism held — `s(τ̂)`
median flat at 0.9950 across tiers at `p_recur = 0`") is true **only of the
`observed` arm**. The `gpu` arm's `s(τ̂)` is strongly tier-separated even at
`p_recur = 0` — a 2.35× spread. The 0.995 figure was read off an array `gpu`
never consulted.

## 4. At `p_recur = 1`, does the tier follow the new vehicle or the returning one? — **the new vehicle**

| run | distinct incoming `veh_id` | reading |
|---|---|---|
| `gpu_closed` | 2168 / 2168 | no vehicle ever returns |
| `observed_closed` | 1839 / 1839 | no vehicle ever returns |

Slot 1, `gpu_closed`:
`fEB.5(orin) → fEB.19(thor) → fEB.30(thor) → fWB.37(orin) → fWB.45(xavier)`,
each with `state_reset=0`.

**Tier changes across a recycle while the state is retained: 60.7%**
(`gpu_closed`), **61.0%** (`observed_closed`).

The recycle sequence at `p_recur = 1` is byte-identical to `p_recur = 0` (this
is the same fact the occupancy invariant reports). So `p_recur = 1` is not a
closed-population control: it is the **same open population**, with the previous
occupant's Polaris state pasted onto a physically different vehicle. This
affects **every arm**, not just `gpu`.

---

## Why the GPU equation can be wrong even though TOPS is a known quantity

The intuition behind the question is that each vehicle reports its GPU-TOPS, so
with `dataset_size` and utilisation identical across the fleet, the prediction
should at minimum separate xavier from orin/thor. The equation does do that —
cleanly, into exactly three groups. The failure is upstream of the arithmetic.

**In the implementation, no vehicle reports TOPS** — there is no TOPS field in
any payload. The CSV lookup is the implementation's *stand-in* for the beacon
reply of the system model (next section): it answers "what TOPS would the
vehicle in this position have told the RSU?" The stand-in is faithful only as
long as `client_id` identifies a vehicle.

Its TOPS comes from:

```
gpu_tops_predict_seconds(client_id, dataset_size, local_epochs)
  └── _client_seconds(client_id, ...)            gpu_tier_sleep.py:487
        └── _predicted_seconds(tier_for(client_id), ...)          :506
              └── tier_for(client_id) -> self._clients[client_id] :126
                    loaded once at startup from clients_csv
```

and `configs/stage7/gpu_open.toml` points that at:

```
clients_csv = ".../sumo_close_loop/input/clients_n20r400.csv"
```

— the **closed-loop** fleet file, 20 rows, `1–10 xavier, 11–17 orin, 18–20 thor`.
Exactly the map recovered from the logs in check 2.

That file is a correct description of the world in Stages 3–6. There,
`client_id` **is** a vehicle, permanently: 20 clients, 20 vehicles, fixed for
the whole run. `tier_for(client_id)` means "the tier of that vehicle", the
equation is handed the right TOPS, and `gpu` separates the tiers perfectly.

Stage 7 silently changed what `client_id` denotes. It is now a **slot** — a
parking space at the RSU that 10,523 vehicles flow through, 2,070 recycles in
`gpu_open` alone, a new occupant roughly every 2.3 rounds. The CSV still says
"client 1 = xavier". Slot 1 is not a xavier; it is a parking space that held 78
xaviers, 56 orins and 28 thors.

**So the equation is not wrong — the binding is.** This is an identity error,
not a modelling error: `tier_for(client_id)` used to mean *the tier of the
vehicle* and now means *the tier of the parking space*. The predictor keeps
answering the question it was asked, confidently and with three clean tiers,
having never asked which vehicle is in the slot.

**Why it is silent.** The physics side was ported correctly and does follow the
vehicle — `sumo_open_server.py:295` overwrites `report.training_time` with
`vehicle_sleep_s(base_seed, veh_id, tier)`, where `tier` is the real tier
captured at dispatch in `self._dispatch_tier`. So the simulated clock, the
waste, the conservation law and the occupancy invariant are all correct, and
nothing errors or looks anomalous. The server **has** the true tier — it is just
never wired into the predictor. Only the ranking is misaligned, and a
misranking has no error signature.

---

## System model (as presented) vs implementation

How the selector is described in the presentation:

1. **Beacon.** The RSU broadcasts a beacon to find which vehicles are in range
   and available to train. *Implemented:* SUMO coverage windows → the slot
   allocator admits a vehicle on coverage entry and releases it on exit.
2. **Beacon reply.** Each vehicle that hears the beacon answers with metadata:
   that it is ready, **its GPU-TOPS** and **its `dataset_size`**. *Implemented:*
   not a message but a lookup, `tier_for(client_id)` into a tiers CSV.
   `dataset_size` is not looked up per vehicle because it is identical (2,500)
   for everyone.
3. **Selection.** The RSU scores the ready vehicles with the GPU equation and
   `V_i = G_i / √(1 + τ̂_i)`, exactly as implemented.

This model is fully compatible with a *proactive* predictor: TOPS arrives in
step 2, **before** selection in step 3. Proactive means "does not wait for the
vehicle's training history", not "cannot use anything the vehicle says". The
distinction that matters is **pre-selection metadata** (TOPS, `dataset_size`,
carried on the beacon reply) versus **post-training observations** (measured
staleness, only available after a round completes, which is what `observed`
uses).

**Where Stage 7 breaks the correspondence.** In the system model the beacon
reply comes *from the vehicle*, so the TOPS it carries is always the vehicle's
own. In the implementation the lookup is keyed on `client_id`:

| | Stages 3–6 (closed) | Stage 7 (open) |
|---|---|---|
| `client_id` denotes | a vehicle, permanently | a slot, re-occupied ~2,000 times |
| `tier_for(client_id)` returns | that vehicle's tier | the slot's row in a closed-loop file |
| agrees with what the beacon reply would carry | **always** | **37% (chance)** |

So Stages 3–6 implement the presented system faithfully; Stage 7's `gpu` and
`cpu` arms do not. The RSU is scoring each vehicle on a TOPS figure that
vehicle never sent.

---

## Required for the re-run

The `gpu` arm must be re-run, and so must `cpu`.

**`cpu` has the same defect.** It takes the identical code path
(`gpu_tops_predict_seconds(cid, …)`, with `tiers_cpu.csv` in place of the GPU
tiers) against the same `clients_n20r400.csv`. Confirmed from the in-flight
logs: `cpu_open` and `cpu_closed` give slots 1–17 a fixed 4.1434 and slots
18–20 a fixed 3.1873 for the whole run. The two `cpu` runs relaunched at
15:57 UTC are therefore invalid as they stand.

What a faithful implementation of the beacon reply needs:

1. **Candidate scoring keyed on the occupant.** When the selector scores slot
   `cid`, the TOPS (tier) must be that of the vehicle occupying `cid` at
   selection time, which is what its beacon reply would carry. The occupant's
   tier is already known server-side (the coverage timeline, the same source
   that fills `_dispatch_tier`).
2. **Busy-pool terms keyed on the dispatched vehicle, not the occupant.** The
   live `T_round` and the busy residuals also call
   `gpu_tops_predict_seconds(busy_cid, …)`. For a client that is training, the
   right tier is that of the vehicle frozen at dispatch (`_dispatch_tier`),
   because the slot may recycle while it trains. This is the same distinction
   as HANDOFF trap 1.
3. **The startup pool median** (`T_round per-tier: xavier n=10 …` in the init
   log) is also computed from the CSV. It must come from the actual initial
   occupants, or be shown not to matter.
4. **Auditability.** The `Staleness prediction accuracy` line should carry
   `veh_id` and the tier used, and the `[Stage7] dispatch` line's `tau_hat`
   must log the value the selector **actually used** (check 3 above).
   Otherwise `analyze_stage7.py` cannot measure the arm.
5. **A pre-flight check before any launch:** predictor tier == vehicle tier
   on 100% of dispatches in a short smoke run.

None of this changes the GPU equation, the scoring function `V_i` or the
selection rule, which the Stage 7 "do not" list protects. It changes only
which vehicle's TOPS is fed into them. The fix belongs in the Stage 7 server
(outside `plato/`).

**Runs affected by this defect:** `gpu_open`, `gpu_closed`, `cpu_open`,
`cpu_closed`. **Not affected:** `baseline_*` (no predictor) and `observed_*`
(never calls the tier lookup).

**Open decision, separate from this defect: issue 4.** At `p_recur = 1` no
vehicle ever returns. The retained state lands on a different vehicle, with a
different tier 61% of the time. If the intended control is "the same vehicle
comes back", then no `*_closed` run implements it, in any arm, including
`baseline` and `observed`. Re-running only `gpu_closed` and `cpu_closed` under
the current `p_recur = 1` semantics would reproduce the same mis-specified
control. This needs a decision before the `closed` re-runs are launched.

---

## Scope of impact

**Invalidated:**

- Predictions 1, 2 and 4 as scored in `PROGRESS.md`; Prediction 5 (`cpu`)
  cannot be scored from the current `cpu` runs either. The `gpu` vs `observed`
  comparison is not a proactive-vs-reactive test; it is a correctly-specified
  estimator against one pointed at the wrong entity.
- The `s_tau` block for `gpu_open` / `gpu_closed` in `stage7_analysis.json`.
- The in-flight `cpu_open` / `cpu_closed` runs (same defect, see above).

**Unaffected** (none of these depend on the `gpu` arm):

- Prediction 3's mechanism: `observed` at `p_recur = 0` is starved as designed
  (obs mean 1.49, 37.2% zero-observation dispatches, flat across windows) and
  still finishes 6.30% ahead of `baseline`.
- Prediction 6: measured κ 2.49–2.71, far below k\* of 30–70.
- All soundness checks: conservation error ≤ 0.04%, plateau 82.35–83.02%,
  occupancy sequence identical across all six finished runs over the common
  1,839-recycle prefix, `state_resets == recycles` at `p_recur = 0`.
- **Stages 3–6.** In a closed pool slot ≡ vehicle permanently, so the same
  lookup is correct there. The defect requires recycling to appear, and only
  Stage 7 recycles.

**Note:** issue 4 is separate from the `gpu` defect and affects all arms.


---

## Reproducing these checks

```bash
cd examples/async/fedasync/sumo_open_loop

# 1 — declared predictor per arm
grep -m1 -o 'Staleness predictor:.*' logs/stage7_gpu_open.log

# 1 — the three distinct predicted values
grep -oE 'predicted=[0-9.]+' logs/stage7_gpu_open.log | sort | uniq -c

# 2 — one fixed value per cid  (see the audit script for the dispatch join)
grep -oE 'cid=[0-9]+  predicted=[0-9.]+' logs/stage7_gpu_open.log | sort -u

# 3 — the field analyze_stage7.py actually parses
grep -oE 'tau_hat=[0-9.]+' logs/stage7_gpu_open.log | sort | uniq -c | head

# 4 — identity continuity across recycles
grep -oE 'incoming_veh_id=\S+' logs/stage7_gpu_closed.log | sort -u | wc -l

# the static table behind all of it
cat ../sumo_close_loop/input/clients_n20r400.csv
```
