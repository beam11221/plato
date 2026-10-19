# Stage 1 — code changes and verification. DO NOT run the sweep yet.

Three changes, then stop and report with evidence files. No full training runs in
this stage. The next stage launches ~37 runs, and if any of these three changes is
wrong, all of them are wasted — so this stage exists to prove the changes are right
before that happens.

---

## Change 1 — Seed torch

Identical configs currently vary ~6% on the simulated clock between runs because
torch is not seeded. That variance has been sitting inside our measured noise floor
and being read as natural variation.

Seed every RNG the run touches: python `random`, `numpy`, `torch` (CPU and CUDA),
and set cuDNN to deterministic. Derive them all from the config's existing `seed`
field — do not introduce a second seed knob.

Report which RNGs were previously unseeded and where the seeding now happens.

---

## Change 2 — Per-client jitter in the sleep generator

**The problem.** The `gpu` arm reads the same tier file that the sleep generator
uses to produce sleep times. Its belief about training time therefore equals ground
truth exactly, by construction. It is an oracle, not an estimator — so the thesis
claim it is meant to test (that a spec-derived estimate is good enough) has never
actually been tested.

**The fix — change the generator only, not the predictor.** In reality
T = W / (P · η): the throughput P is on the datasheet and genuinely knowable, but
the achieved efficiency η is not — it depends on memory bandwidth, thermals, batch
size and background load. Two units of the same model do not train at identical
speed. Make the simulator reflect that:

```
sleep_i = T_tier(tier_i) * exp(eps_i)
eps_i ~ Normal(0, sigma)
```

Rules:

- `eps_i` is drawn **once per client when the fleet is created** and stays fixed for
  the whole run. It models a persistent per-unit offset, not transient load. Do not
  redraw it per round.
- Draw it from the seeded RNG so the fleet is reproducible.
- Pick `sigma` so the realized coefficient of variation of sleep time within a tier
  is **15-20%**. State the sigma you used and the CV you measured.
- The selector still reads `T_tier` from the unchanged tier file. Do not give any
  arm access to `eps_i` except `gpu_oracle` below.

**Sanity requirement:** jitter must not blur the tiers. Xavier and Thor differ by
~15x; at CV 15-20% the tier distributions should stay clearly separated. Report the
min/max sleep per tier and confirm no overlap between Xavier and Orin, or between
Orin and Thor.

---

## Change 3 — Split `gpu` into two arms

| arm | what T-hat it holds | role |
|---|---|---|
| `gpu` | the **tier mean** from the datasheet tier file | the deployable method — what the thesis actually claims |
| `gpu_oracle` | the **per-client true** sleep, jitter included | upper bound; not deployable, used only to show how much headroom the spec-derived version gives up |

`gpu` is the existing code path and needs no change beyond the generator now
differing from it. `gpu_oracle` is a new arm that reads per-client truth.

Keep `cpu` and `observed` exactly as they are.

---

## Evidence to produce (this is the deliverable)

Put everything in `stage1_evidence/`:

**`CHANGES.md`** — every edit, as file:line plus a short why. Nothing else changed.

**`sleep_table.csv`** — one row per client, for a fleet built with seed 4126:
```
client_id, tier, T_tier_s, eps, sleep_s
```

**`jitter_check.txt`**
- sigma used, and realized CV per tier (target 15-20%)
- min / mean / max sleep per tier
- explicit statement of whether Xavier/Orin and Orin/Thor ranges overlap
- ratio of tier means before and after jitter (should still be ~15x Xavier:Thor)

**`arm_beliefs.csv`** — for the same fleet, what each arm believes:
```
client_id, tier, true_sleep_s, gpu_belief_s, gpu_oracle_belief_s, cpu_belief_s
```
This is the file that proves `gpu` is no longer an oracle. `gpu_belief_s` must be
constant within a tier; `gpu_oracle_belief_s` must equal `true_sleep_s`.

**`determinism.txt`** — run the same config (N=20, C=4, alpha=0.8, arm gpu,
seed 4126) twice for **200 rounds** each, and report:
- whether the two accuracy curves are identical row for row
- whether `sim_time_s` at round 200 is identical
- if not identical, the largest per-round difference and where the remaining
  nondeterminism comes from

200-round runs only. Do not run 4,000 rounds in this stage.

---

## Then stop

Report the five files and wait. Do not launch the sweep, do not tune sigma to make
any result look better, and do not touch the tier file, the fleet mix, the dataset,
the model, or the selector code.

If `determinism.txt` still shows a difference after seeding, **say so plainly rather
than working around it** — a remaining source of nondeterminism is something we need
to know about before spending 37 runs on top of it.

---

## What comes next (context only — do not start)

Once these five files check out: re-run every arm on the new sleep traces.

- Phase 1, C sweep at N=20: C in {1, 2, 4, 6, 8, 10}, arms baseline / gpu / cpu /
  observed, alpha=0.8, 4,000 rounds, seed 4126 — 24 runs
- Phase 2, fleet-size sweep at C=4: N in {30, 40} with partition_size 1667 and 1250
  so that N x partition_size stays at CIFAR-10's 50,000 — 8 runs
- Last: `gpu_oracle` at C=4 and C=8 only — 2 runs

All previous results become an oracle branch: kept for reference, not poolable with
anything produced after these changes.
