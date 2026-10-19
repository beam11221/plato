# ADR-0001: Communication latency in the tier-speed model

Status: accepted, sweep in progress. See `BRIEF_comm_sweep.md` for the full
experiment design and `gpu_tiers/HANDOFF.md` for the machinery this reuses.

## Context

Every run so far (H-sweep, fleet-mix sweep) models a client's training time as
compute only:

```
T_i = W_model / (TOPS_i * eta_i)
```

The thesis claim is that a GPU-TOPS-derived staleness signal beats CPU- or
reactively-derived ones because GPU throughput varies ~15x across vehicular
tiers while CPU clock varies ~1.2x. That claim is incomplete without a wireless
link: real vehicles upload the trained model over one, and that cost is the
same for every vehicle (they all ship the same 42.69 MB ResNet-18), so it
enters as an additive constant on top of the tier-dependent compute term:

```
T_i = W_model / (TOPS_i * eta_i)  +  D / B
      \_____ tier-dependent _____/    \_ identical for all _/
```

An additive constant common to both sides of a ratio pushes that ratio toward
1 as the constant grows:

```
(T_xavier + C) / (T_thor + C)  ->  1   as C -> infinity
```

So as bandwidth falls, the GPU speed signal is diluted, not degraded, and
`V_i = G_i / sqrt(1 + tau_hat_i)` collapses toward `V_i ~= G_i` — a pure
data-quality selector. This is a limitation study: a result showing the
method's advantage vanish at realistic V2N bandwidth is a **success**, being
measured honestly, not a failure to be tuned away.

## Decision

### 1. The equation

`C = D / B` seconds, added identically to every client within a run.

| symbol | value | notes |
|---|---|---|
| `D` | 42.69 MB = 341.52 Mbit | fixed; the measured payload (HANDOFF § 4.8) |
| `B` | the swept variable | Mbps |
| `C` | `341.52 / B` seconds | identical for every client in one run |

Implemented as two new `[gpu_tier]` keys, `comm_bandwidth_mbps` (B) and
`payload_mbit` (D, defaults to 341.52), read once per process by
`gpu_tier_sleep.comm_latency_seconds()`. Absent `comm_bandwidth_mbps` -> C = 0,
byte-identical to every run made before this feature existed — verified by
regenerating the H-sweep and fleet-mix sweep configs after this change landed
and diffing byte-for-byte against what was already on disk. A key that IS
present but <= 0 raises rather than silently falling back to C = 0 (same
discipline as `t_round_model`, HANDOFF § 4.10 vs § 4.11): a silent fallback
here would burn hours of compute on a run that looks like the real sweep but
answers a different question.

### 2. Uplink only

Downlink is a broadcast from a base station at far higher transmit power and is
conventionally neglected in this literature (Information Sciences 664 (2024),
project papers, § 3.3). Modelled as `D / B`, not `2 * D / B`. Stated here and in
the config comment (`comm_bandwidth_mbps` block in the rendered TOML) so it is
never silently assumed to be symmetric.

### 3. Two injection points, kept independent

The codebase already keeps BELIEF (what the selector thinks) and REALITY (what
the simulated clock charges) on paths that meet at exactly one line
(`residual = max(0, t_full - (wall_now - start_time))`, HANDOFF § 1). This
decision preserves that split — C is added on both paths independently, never
derived from one on the other:

| path | where | how |
|---|---|---|
| reality | `generate_sleep_times.py` -> the per-client sleep CSV | `SleepTimeGenerator.sleep_times(..., comm_latency=C)`, added AFTER the anchor `scale` |
| belief | `gpu_tier_sleep.gpu_tops_predict_seconds()` | `_predicted_seconds()`, added AFTER `predictor_scale()`, gated by `comm_in_predictor()` |

A new `[gpu_tier] comm_in_predictor` boolean (default `true`) controls whether
BELIEF includes C at all — see § 6, the ablation.

### 4. C is added AFTER the anchor scale, never before

The sleep CSVs are anchored (xavier = 20 s) while the raw predictor formula
gives xavier = 0.055 s — a 363x scale difference that `predictor_scale()`
reconciles at every predictor-side call site (HANDOFF § 4.1). `C` is already a
real-seconds quantity in the *anchored* simulator's units, so:

```python
t_pred = predictor_scale() * raw_compute_prediction + C   # correct
t_pred = predictor_scale() * (raw_compute_prediction + C) # WRONG — scales C by 363x
```

Getting this backwards would make `residual` saturate or collapse the same way
the un-anchored predictor did before HANDOFF § 4.1 was fixed. Pinned by
`test_predictor_anchor.py` §§ 6-8 (extended for this decision): C on the
anchored scale reproduces `gpu_tops_predict_seconds(xavier) = (C=0 prediction)
+ C` exactly, not `(C=0 prediction + C) * scale`.

Single definition (`gpu_tier_sleep._predicted_seconds`) is used by every
predictor-side caller — `gpu_tops_predict_seconds`,
`compute_pool_T_round_seconds`, `compute_pool_mean_T_seconds` — so `T_round` is
always derived from the same augmented times individual clients are predicted
from. No second, un-augmented path exists to drift out of sync (BRIEF § 4.4).

### 5. `t_round_model` MUST be `"queue"`

`"min"` computes `E[min of P draws]`; as `C` grows, every client's predicted
time converges toward `C`, so `E[min of P]` and `E[T]/P` diverge further apart
along this sweep's own x-axis (BRIEF § 4.1, HANDOFF §§ 4.11/4.14). The reused
`C=0` control cell (the `m50_35_15` cell of the fleet-mix sweep at G=30) was
already run under `"queue"`, so this is also a compatibility requirement, not
just a correctness one.

Verified numerically before any run: `pref(gpu)` — the speed term's preference
for thor over xavier — reproduces the fleet-mix sweep's already-reported 3.42 at
C=0 (HANDOFF § 4.14) and decays as `C` grows:

| B (Mbps) | C (s) | T_round (s) | pref (gpu) | pref (cpu) |
|---|---|---|---|---|
| inf | 0 | 1.138 | 3.42 | 1.13 |
| 100 | 3.415 | 1.479 | 2.12 | — |
| 50 | 6.830 | 1.821 | 1.75 | — |
| 20 | 17.076 | 2.845 | 1.39 | — |
| 10 | 34.152 | 4.553 | 1.22 | — |

(`comm_sweep.py predicted_pref()`, computed directly from `gpu_tier_sleep`
against each cell's own rendered config — not re-derived by hand — so this
table and the code that produced it cannot drift apart. `cpu`'s pref is
constant in B at 1.13, unaffected by the GPU-side gap, matching HANDOFF §
4.14's control-arm reasoning.)

### 6. The belief-vs-reality ablation

At the lowest bandwidth (B=10 Mbps), `gpu` runs twice:

| | reality has C | belief has C | |
|---|---|---|---|
| A | yes | yes | correctly calibrated selector |
| B | yes | no (`comm_in_predictor=false`) | believes thor is ~8x faster when it is ~1.5x faster |

Arm B reuses arm A's sleep CSV (reality is identical) under a distinct run
identity (`..._b010nc`) and a distinct port, so it never collides with arm A.
Implemented as the `comm_in_predictor` config flag rather than a code branch —
the same function (`_predicted_seconds`) serves both, gated by one boolean read
once per process.

## Consequences

- Every config generated by `make_comm_sweep.py` carries a `[gpu_tier]` comment
  block stating C = D/B, uplink-only, and citing this document.
- `make_configs.py`'s `SHARED_GPU_TIER_KEYS` gained `comm_bandwidth_mbps` and
  `payload_mbit`: they must match between the `gpu` and `cpu` arm of the same
  cell (same treatment as `t_round_model`), but are exempt from the mix-label
  comparability check via a bandwidth-suffix strip (`_b\d+[a-z]*$`) so a
  bandwidth cell can reuse the fleet-mix sweep's `m50_35_15` clients file
  without tripping the "clients_csv and sleep CSV must name the same fleet"
  check.
- `comm_in_predictor` is deliberately NOT a shared key: the ablation is a
  single `gpu`-only cell that intentionally differs from its own sibling, not
  something compared across arms.
- This sweep's identity (`t_round_model="queue"`, `target_accuracy=1.01`,
  `rounds=5000`, seed 8888) matches the fleet-mix sweep's, which is what makes
  reusing its `C=0` cell valid — and is exactly why it must NOT be pooled with
  the H-sweep (`"min"`, `target_accuracy=0.6`, `rounds=10000`, HANDOFF § 4.7).
