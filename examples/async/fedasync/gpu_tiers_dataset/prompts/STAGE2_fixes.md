# Stage 2 — three fixes and a verification report. Still no sweep.

Stage 1 passed on the jitter and the arm split. Two things remain before a sweep is
worth launching, plus one lowered parameter. As before: make the changes, produce
the evidence, and **stop**. The sweep is ~32 runs; if any of this is wrong they are
all wasted.

Do not touch anything under `plato/`. All three changes live in the example folder.

---

## Change A — Lower the jitter to CV 15%

`sigma = 0.18` gives a nominal CV of 18.15%, and at that value the orin and thor
sleep ranges clear each other by only **5%** in the seed-4126 draw. A different seed
could close that gap — and orin/thor is exactly the pair `cpu-freq` already cannot
separate, so an overlap there would quietly shrink the `gpu` advantage we are trying
to measure.

Set sigma for a nominal CV of **15%**: `sigma = sqrt(ln(1 + 0.15^2)) = 0.1492`.

Build fleets and report the separation margin (`slower.min / faster.max`) for:

| | seed 4126 | seed 4127 | seed 4128 |
|---|---|---|---|
| N = 20 | | | |
| N = 40 | | | |

Report `xavier/orin` and `orin/thor` for all six. **If any orin/thor margin is below
1.10, stop and report** rather than lowering sigma further on your own — that is a
decision to take with the numbers in hand, not a parameter to search.

Also report, per fleet: realised CV per tier (population and sample), tier mean
ratio xavier:thor, and the belief-vs-reality gap on `E[T]`.

**Sigma is locked after this.** It is being chosen on the tier-separation margin —
a property of the fleet — before any accuracy result exists. It must not be revisited
once results start arriving, whatever they show.

N = 30 is dropped: 50/35/15 does not divide into 30, and the realised 15/10/5 puts
thor (the fastest tier) at 16.7% instead of 15%. The fleet-size sweep is N ∈ {20, 40}.

---

## Change B — Make randomness belong to the client, not the worker

`determinism.txt` §4 found that torch's RNG lives in the PLATO **worker process**,
and which worker serves which client depends on the order the client processes
happen to connect over socket.io — a real-time race. A client trained by worker 2
in one run and worker 4 in another draws a different augmentation sequence, computes
different gradients, gets a different `G_i`, and the selector then picks a different
client. That is how two runs of one config diverged at dispatch 18.

Re-seed torch at the start of each local training run from
`(base_seed, client_id, round)` instead of once per process.

This is not a change to the model being simulated — a worker is an implementation
detail of the simulator, a client is not. Right now the parallelism strategy leaks
into the experiment. Tying randomness to `client_id` puts it where the model already
says it belongs.

Do it in a trainer subclass in this folder. Do not edit `plato/`.

---

## Change C — Reconstruct the clock outside PLATO

`determinism.txt` §3 found that PLATO adds the host's `processing_time` — a real
`perf_counter()` measurement — into the simulated clock on every report. So the
simulated clock is inflated by whatever else the machine was doing, which means
**two runs on a loaded box are not comparable**, and parallel runs contaminate each
other. It is also the reason the ×0.15 anchor rescale failed earlier: host overhead
does not scale with the anchor.

We are not fixing PLATO. We are computing the clock ourselves afterwards, from the
dispatch order in the logs and the per-client sleep times we already know.

### The model

```
slots = [0.0] * C
for (client) in dispatch_order:            # in the order the log records them
    i        = argmin(slots)
    t_start  = slots[i]
    t_finish = t_start + sleep[client] + comm_time
    slots[i] = t_finish
    record(client, t_start, t_finish)
```

`sleep[client]` comes from the fleet's `sleep_table.csv`. `comm_time` must be **read
from the config**, not hard-coded to zero — it is zero under
`comm_simulation = true` today, and silently wrong the day that changes.

Accuracy needs no adjustment: it is indexed by round, and rounds do not move. Only
the time axis is recomputed.

### Output

A script that takes a run directory and emits
`accuracy_curves_reconstructed.csv`:

```
experiment, arm, seed, round, sim_time_plato, sim_time_reconstructed, accuracy
```

Keep **both** clocks in the file. The PLATO one is what we are replacing, and we
will want to be able to look back at it.

### Verification — the deliverable

**C1 — is the queue model right?** Reconstruct the clock for a finished run and plot
`sim_time_plato - sim_time_reconstructed` against round.

That difference is accumulated `processing_time`, so it must be **positive and
monotonically increasing, and reasonably smooth**. Report its value at the final
round, both absolute and as a percentage. If it jumps, plateaus, or ever goes
negative, **the queue model is wrong** — say so rather than adjusting the model to
fit; a negative difference in particular means the reconstruction is not reproducing
what the server did.

**C2 — does it actually fix the contamination?** Take the two determinism runs
(0.80% spread on PLATO's clock) and reconstruct both.

| result | meaning |
|---|---|
| spread falls to ~0% | the contamination was host overhead alone |
| spread remains | it comes from the differing dispatch order — i.e. Change B is load-bearing |

Report the number either way. Do this **before** and **after** Change B if you can,
because the pair of numbers separates the two causes cleanly.

**C3 — is the dispatch order actually recoverable?** Confirm the logs record the
**order** of dispatches, not only the round number: several dispatches can land in
one round, and the queue replay needs the sequence. Say which log line or CSV column
carries it. If the order is ambiguous anywhere, report where — do not guess an
order and proceed.

---

## Then stop

Produce:

- `stage2_evidence/jitter_check_cv15.txt` — the six-fleet margin table and the
  per-fleet statistics
- `stage2_evidence/CHANGES.md` — every edit, file:line, and why
- `stage2_evidence/clock_reconstruction.md` — C1, C2, C3 with numbers
- `stage2_evidence/determinism_after.txt` — two 200-round runs repeated after
  Change B, reporting both clocks, and whether the dispatch sequence now matches

Then wait. Do not launch the sweep.

---

## Reminders

- No tuning to make a number land anywhere. If something misses a target, report
  the miss — Stage 1 did this correctly with the CV band and it was the right call.
- Do not edit `plato/`, the tier files, the fleet mix, the dataset, the model, the
  optimiser, or the selector.
- If a fix does not work, say it did not work. A residual source of nondeterminism
  that we know about is far cheaper than one we discover after 32 runs.

## What follows (context only — do not start)

Phase 1, C sweep at N=20: C ∈ {1, 2, 4, 6, 8, 10} × {baseline, gpu, cpu, observed},
α=0.8, 4,000 rounds — 24 runs.
Phase 2, fleet size at C=4: N=40 — 4 runs.
Last: `gpu_oracle` at C=4 and C=8 — 2 runs.

Whether one seed is enough for those depends on what C2 and
`determinism_after.txt` report.
