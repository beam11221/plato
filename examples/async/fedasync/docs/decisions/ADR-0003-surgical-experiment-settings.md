# ADR-0003: Settings for the surgical speed-estimator experiment

Status: accepted 2026-08-28, runs launched the same day. Design in
`surgical/BRIEF_surgical.md`; the arm itself in ADR-0002; the machinery this
reuses in `gpu_tiers/HANDOFF.md`.

## Context

The supervisor asked for a surgical comparison of three speed estimators — GPU
spec, CPU spec, and history (Polaris's observed-τ) — on a 20-client CIFAR-10 /
ResNet-18 fleet with Dirichlet α = 1.0. Fixing the fleet size and the partition
turned out to disturb several settings that had been stable for three sweeps,
because they were calibrated against 100 clients × 500 samples. Each is recorded
below with what it displaced.

## Decisions

### 1. Fleet: 20 clients, 10 / 7 / 3, `partition_size = 2500`

50/35/15 over 20 divides exactly. `surgical/make_fleet.py` writes the client→tier
map in contiguous ID blocks — a layout that is load-bearing (`HANDOFF.md` § 4d):
PLATO derives a client's shard from `random_seed × client_id`, so blocks keep
client *i*'s data fixed when a future fleet sweep changes the mix, and make
fleets nested.

`partition_size = 2500` is 50 000/20. Note what it is **not**: the `noniid`
sampler is `plato/samplers/dirichlet.py`, which draws `partition_size` indices per
client with `WeightedRandomSampler(replacement=False)` over the whole training
set. A client's 2500 samples are unique *within* that client but may overlap
another client's — this is **not a disjoint 1/20 partition**. The 100 × 500 setup
had the same property; keeping PLATO's default was an explicit choice, not an
oversight.

### 2. `concentration = 1.0` (was 0.9)

Milder label skew narrows the spread of `G_i`, so `V_i = G_i / sqrt(1 + τ̂)` is
driven more by the staleness term — the term this experiment is about.

### 3. `per_round = max_concurrency = 4` — **20 % of the fleet**

The earlier sweeps ran 10 of 100 = 10 %. This is double that ratio, and both
consequences matter:

* **It makes steering possible.** thor is 3 clients; 4 slots can hold all of
  them, so the selector can reach a 75 % dispatch share against a 15 % pool
  instead of running out of fast clients the way the 70/20/10 mix cell did.
* **It weakens the effect being measured.** Every client is re-dispatched roughly
  every 20 rounds instead of every 100, so the `observed` arm's stored τ is ~5×
  fresher than in the runs that motivated this experiment. Pulling the other way,
  a 4-slot busy pool changes composition more violently per selection, so
  `T_round(t)` may be noisier. The net sign is unknown; it is pre-registered in
  BRIEF § 6 so a small observed-vs-gpu gap is not misread as the mechanism
  failing.

### 4. Tier spec: `tiers_estimated_15x.csv` (10 / 78 / 150 TFLOPS)

The project's own hardware estimate, a 15× xavier→thor gap, matching the thesis
claim. Deliberately **not** `tiers_geo_gap15.csv` (10 / 38.73 / 150), whose
geometric mid-tier the gap sweep used. The consequence is recorded because it is
not obvious: with 78 TFLOPS, orin→thor is only 1.37× *effective*, so the fleet
behaves closer to "one slow tier + two fast" than to three evenly separated
tiers. Per-tier MAE should be read with that in mind.

### 5. `avg_training_time = 0.0` (was 1.5) — the largest change

Under `sleep_simulation`, the client measures a real training time and then
reports `(avg_training_time + sleep) × epochs` instead. 1.5 s was the real
ResNet-18 compute on the tuning machine at 500 samples, kept only so the
simulated clock matched even older runs that really slept — a
backward-compatibility constant, which a new experimental identity does not owe.

Three reasons it goes to zero here:

1. **It double-counts.** The sleep formula and the predictor formula are the same
   formula, `T_i = W_model/(TOPS_i·η)` — a whole training time, not an increment.
   Adding a lab GPU's compute on top mixes the host into the vehicles' clock.
2. **It dilutes the independent variable.** Being an additive constant identical
   for every client, it pushes every tier ratio toward 1: at the measured ~4.2 s
   (see § 9) the effective xavier→thor gap would fall from 14.3× to ~9.3×. Same
   mechanism the comm sweep exists to study (ADR-0001).
3. **It would contaminate the headline metric.** The predictor does *not* know
   about `avg_training_time`, so with it non-zero even the "correct" gpu arm
   under-predicts — and worst on the fastest tier (−53 % on thor at 1.5 s). This
   experiment's primary metric is MAE of τ̂ per tier; that bias would land
   directly in it.

At 0.0, belief and reality are identical by construction and the tier sleep is
the sole determinant of training time.

The key must remain **present** and zero: `defaults.py:356` reads it with no
guard, so deleting it is an AttributeError on the first client report.

### 6. `comm_simulation = true` (was false)

The name is misleading: it means "do not charge communication time", not "turn
communication on". It routes the payload through a file instead of a socket and
sets `comm_time = 0` (`plato/servers/base.py:1007`). With `false`, `comm_time` is
a `time.time()` delta measured on whichever host ran the cell (~0.3 s/report),
which lands in the simulated clock and is what made the earlier sweeps
incomparable across machines (`HANDOFF.md` § 4.8). That section already
recommended this change; the surgical experiment is the first identity free to
take it.

**Two communication models exist in this codebase and both stay off:** PLATO's
own (`[clients] compute_comm_time` with `[server] uplink_bandwidth` /
`downlink_bandwidth`) and this project's (`[gpu_tier] comm_bandwidth_mbps`, C =
D/B, ADR-0001). Neither key appears in any surgical config. They are independent,
and turning on the wrong one would silently answer a different question.

What remains host-dependent is `processing_time` alone
(`defaults.py:252`), which no config can switch off.

Disk cost, measured: the payload files use a fixed name per client and are
overwritten each round (`{model}_{cid}.pkl` and `{model}_client_{cid}.pkl` under
`checkpoint_path`), so they are bounded at 40 files ≈ 1.7 GB for this fleet, not
growing. `clean_checkpoints.sh` does not touch them — its `round_of()` matches
only `prng_state_*`, `numpy_prng_state_*` and `*.safetensors.pkl`.

### 7. `anchor_sleep = 100` (was 20), `max_sleep_time = 300` (was 120)

**What the anchor is.** The raw formula gives xavier 0.275 s at 2500 samples
because `eta_nominal = 1.0` — efficiency is not modelled at all. Against the
framework's own un-scalable per-report overhead, a 15× hardware gap would be
*experienced* as 1.66×. The anchor is a uniform multiplier applied to every tier,
so it never touches the ratios between them; it sets where those ratios sit
relative to that overhead. It is a **design knob, not a physical claim** — if the
overhead were zero, the anchor would change nothing at all.

**Why it had to move.** The raw time scales with `partition_size`, so holding 20 s
while the partition grew 500 → 2500 would silently make the simulated hardware 5×
faster. 100 s keeps the implied efficiency at 0.275 % of peak — the same 363×
scale factor every earlier sweep ran at (the "363x" of `HANDOFF.md` § 4.1) — and
lifts the effective gap from 11.96× to **14.26×**.

**The rule, so future fleet sweeps stay consistent:**

```
anchor_sleep = 0.04 s × partition_size        (20 s @ 500, 100 s @ 2500)
```

`max_sleep_time` rises to 300 because `Config.simulate_client_speed()` clips
every sleep at it *silently* (`HANDOFF.md` § 4.10) and 120 would leave a 100 s
anchor only 20 % of headroom.

Nothing else moves with the anchor: τ̂ per tier (7.21 / 0.92 / 0.48), `pref`
(2.35), mean staleness (C/M = 4) and every arm-vs-arm ratio are unchanged,
because they are ratios. Only the framework overhead's share, and the absolute
simulated hours, differ.

### 8. `staleness_bound = 25` unchanged — and it is now inert

`staleness_i ≈ (C/M)·(T_i / T_mean_selected)`; with C = 4, M = 1 and this fleet's
E[T] = 55.49 s, xavier lands at ≈ 7.2 rounds and the other tiers below 1. The
bound never fires, so nothing is censored — the opposite of the C = 10 situation
where a bound of 11 truncated 68 % of xavier dispatches and penalised exactly the
behaviour under measurement (`HANDOFF.md` § 4.9). Keeping the historical value is
therefore safe *here*; re-derive it if `per_round`, the tier spread or
`minimum_clients_aggregated` change.

### 9. The real measured compute is logged, not lost

`plato/clients/strategies/defaults.py` now logs the measured training time
immediately before discarding it:

```
[Client #12] [SleepSimulation] measured_training_time=4.100936 s
             reported_training_time=12.820513 s (sleep=12.820513 avg=0.000000 epochs=1)
```

Without this line the host's actual per-round cost is unrecoverable after a run,
since the reported value is synthetic. Measured on the pilot: **≈ 4.1–4.4 s** per
client per round at 2500 samples with 4 concurrent — the number to use if actual
training cost is ever wanted, and the number that would have collapsed the
effective gap had it been fed back into `avg_training_time` (§ 5).

### 10. Run identity

`rounds = 10000`; `target_accuracy = 0.9`, which is **no early stop in practice**
— every full-length run in this project plateaus near 0.60 at 500 samples/client,
and 0.9 is out of reach for FedAsync + non-IID on CIFAR-10/ResNet-18 even with 5×
the data. Every arm therefore runs exactly `rounds` rounds and stays comparable
(`HANDOFF.md` § 4.13). **Do not report "did not reach 0.9" as a result**; pick a
reporting target from the finished curves.

Single seed **4126**, shared by every arm. `t_round_model = "queue"` — the correct
model (§ 4.11), free to use because this identity has no backward-compatibility
burden, and it affects the gpu and cpu arms only since the observed path never
computes `T_round`.

Four arms including `baseline`: every ratio this project reports is normalised by
its own cell's baseline.

Ports **17115 / 17215 / 17315 / 17415**, via `make_configs.py --port-offset 9000`,
clear of every existing block (gap 8100–8440, multi-seed 9000+, mix 11000+, comm
14000+). The uniqueness check only sees one render at a time, so it cannot catch a
collision *across* sweeps — separate blocks are what prevent it.

## Consequences

* A fourth experimental identity. Nothing here pools with the gap, multi-seed,
  fleet-mix or comm sweeps — different fleet size, partition, concentration,
  concurrency, anchor and clock model.
* The clock is now almost entirely simulated: `processing_time` is the only
  host-dependent term left, against tier sleeps of 100 / 12.82 / 6.67 s. Time-to-
  accuracy from this experiment is reproducible across machines in a way the
  earlier sweeps' was not.
* Accuracy is still fully real — only *time* is simulated. That is what makes the
  approach valid, and it belongs in any write-up.
* Single seed means no claim about small differences. The ±10 % evaluation-noise
  floor (`HANDOFF.md` § 4.13) applies to every time-to-accuracy number; the
  mechanism metrics (MAE of τ̂, dispatch share) do not carry it and should be
  preferred.
