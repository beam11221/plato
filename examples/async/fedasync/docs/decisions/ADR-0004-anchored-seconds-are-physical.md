# ADR-0004: Anchored seconds are physical seconds, and `eta_nominal = 1.0` is a placeholder

Status: accepted 2026-09-06, before the SUMO runs launched. Experiment in
`sumo_close_loop/`; the anchor mechanism in `gpu_tiers/gpu_tier_sleep.py`; the
surgical experiment's view of the anchor in `gpu_tiers/HANDOFF.md` § 4.1 and
§ 4.4.

## Context

Adding SUMO mobility put a question to the anchor that no earlier experiment had
to answer.

The tier model computes a training time from a FLOPs budget:

```
raw = W_MODEL_FLOPS * dataset_size * epochs / (gpu_tflops_fp16 * 1e12 * eta_nominal)
```

`tiers_estimated_15x.csv` sets `eta_nominal = 1.0` for every tier, so for
`partition_size = 2500` the raw times are **275 ms / 35.3 ms / 18.3 ms** for
xavier / orin / thor. `[gpu_tier] anchor_sleep` then rescales all tiers uniformly
so xavier sleeps a chosen number of seconds — 100 s in the surgical experiment,
15 s here.

Until now the anchor was **inert with respect to every measured quantity**. The
scale cancels in `tau_hat = T_i / T_round`, so the four arms select identically at
any anchor; `HANDOFF.md` § 4.4 calls it load-bearing only because PLATO adds a
~2 s per-client overhead, and an additive constant compresses the tier ratio when
the sleeps are small. That is a simulator artifact, not physics.

Mobility breaks that. Coverage windows last ~34.7 s of **trace** time, and the
experiment maps trace time onto PLATO's simulated clock 1:1
(`sumo_time() = wall_time - initial_wall_time`). Whether a vehicle can finish
training before it leaves RSU range — the quantity the whole experiment measures
— is therefore decided by the ratio

```
T_train (anchored seconds)  :  coverage window (trace seconds)
```

So one of two things had to be true: either the anchored seconds are physical and
the 1:1 mapping is right, or the raw seconds are physical and the trace should be
divided by the anchor scale (54.5455x here) as well.

## Decision

### 1. Anchored seconds are the physical seconds. The trace maps 1:1.

`eta_nominal = 1.0` means *one hundred per cent of peak FP16 throughput*, which no
real training workload achieves — least of all 32x32 images at batch 64 on an
embedded GPU, where the tensor cores are largely idle and the work is memory- and
launch-bound. The raw number is a theoretical lower bound, not a duration.

The repository contains its own refutation. The lab GPU — far faster than a Jetson
Xavier — measured **4.06-4.89 s** for the identical workload, logged on every
round as

```
[SleepSimulation] measured_training_time=4.058048 s  reported_training_time=15.000000 s
```

If the raw 275 ms were physical, a Xavier would train ~15x faster than a
datacenter GPU.

So the anchor is not an inflation applied to a real number. It is a **calibration
of effective utilisation**: at `anchor_sleep = 15` the implied efficiency is

```
eta_effective = 275 ms / 15 s = 1.83 %   (uniform across tiers)
```

and 15 s for one epoch over 2500 CIFAR-10 images on a Jetson Xavier is the claim
the experiment is actually making.

### 2. `eta_nominal = 1.0` is recorded as a placeholder, not a belief

Nothing in the tier CSVs currently carries a measured efficiency. The 1.0 is a
neutral value that pushes all of the calibration into the anchor. That was
harmless while the anchor was inert; it is not harmless now, because the anchor
now sets a physical ratio that determines the headline result.

This is accepted **for the current runs** — the alternative is to block the
experiment on hardware measurements nobody has yet — but it is recorded here so
the assumption is visible in the write-up rather than buried in a CSV column.

### 3. The migration path, for when real efficiencies exist

The anchor and `eta_nominal` are the same knob written two ways:

```
T_i = W * D / (TFLOPS_i * 1e12 * eta_i)  *  scale        where scale = anchor / raw(anchor_tier)
```

so setting `eta_nominal = 0.0183` in `tiers.csv` and dropping `anchor_sleep`
reproduces today's numbers **exactly**. Prefer that form once real values arrive:
it is self-documenting and it allows a *per-tier* efficiency, which the anchor
cannot express.

What changes when per-tier efficiencies are used, and must be re-derived rather
than assumed:

* **The tier gap stops being 15x.** It becomes
  `(TFLOPS_thor * eta_thor) / (TFLOPS_xavier * eta_xavier)`. Newer architectures
  generally reach a *lower* fraction of peak on small workloads, so the realistic
  gap is likely narrower than the nominal one.
* **`tau_hat` changes**, and with it every predicted staleness in the `gpu` arm.
  `preflight.py` prints the new values; they must be re-checked against the log.
* **The waste rates change**, because `T_i` moves against a fixed 34.7 s window.
  Re-run `preflight.py` section 4 before assuming the current 43.5 / 5.6 / 2.9 %
  still hold.
* **`tiers_cpu.csv` must be regenerated too**, or the `cpu` arm's wrongness stops
  being the wrongness the experiment intends to measure.
* Results produced under the old efficiencies are **not poolable** with results
  produced under the new ones. It is a different fleet.

## Consequences

* The 1:1 trace mapping in `sumo_mobility.py` and `SumoRsuMixin.sumo_time()` is
  correct as written; no code change follows from this ADR.
* `preflight.py` prints the implied `eta_effective` every run, so the assumption
  is restated each time rather than remembered.
* Any write-up of the SUMO experiment must state the assumed training time and
  its implied efficiency alongside the waste rates, because the waste rates are
  a direct function of it.
* If a published or measured Jetson training benchmark becomes available, the
  cheapest useful improvement to this experiment is to adopt it — the anchor is
  currently the least-evidenced number in the whole setup.
