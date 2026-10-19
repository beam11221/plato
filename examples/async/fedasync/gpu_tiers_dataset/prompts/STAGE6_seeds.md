# Stage 6 — complete the `vmax_distance` table, then add seeds. 12 runs, one gate.

Stage 5b settled the open question: `vmax_distance` is a perfect lower bound on
remaining dwell (0 violations in 480,957 checks), halves the median gap to truth
(2.30 s vs 4.86 s), is deployable, and costs only 1.9 points of speedup against the
flawed `vmax_elapsed` (−39.8% vs −41.7%).

**Decision taken: `vmax_distance` is the method.** `vmax_elapsed` becomes a named
ablation — the naive approximation — and `oracle_exit` the non-deployable ceiling.
Stage 5's runs stay on record under those labels and are not re-run.

That decision leaves a hole. Stage 5's six runs all used `vmax_elapsed` (the default);
Stage 5b ran only `baseline` under `vmax_distance`. So at the method's own `R_i` the
table has exactly one row.

| cell | at `vmax_elapsed` | at `vmax_distance` |
|---|---|---|
| baseline + none | exists (Task 4) — **valid for every `R_i`**, the filter is off so `R_i` is never evaluated | same run |
| baseline + tier_spec | exists (Stage 5) | exists (Stage 5b) |
| baseline + fleet_mean | exists (Stage 5) | **missing** |
| gpu + tier_spec | exists (Stage 5) | **missing** |
| cpu + tier_spec | exists (Stage 5) | **missing** |
| observed + fleet_mean | exists (Stage 5) | **missing** |

Batch A fills the hole. Batch B adds seeds to the four cells that carry a claim.

Environment, identical in every run and to Stage 5/5b: SUMO closed population,
radius-400 trace `e1_closed_seed4126_range400`, N=20, C=4, α = 0.8, 4,000 rounds,
`upload_policy = coverage_gated`, `stage0_r_i = vmax_distance`, `V_i`-weighted lottery,
Stage 0 fallback rule unchanged.

Do not touch anything under `plato/`.

---

## The one thing that must not vary

**The fleet stays `n20_x10-o7-t3_xa15_p2500_j0p149_s4126` in all 12 runs.** Only the
run seed changes — the lottery draws, torch, dispatch order. The per-client `sleep_i`
values must be byte-identical across every run in this stage; verify by MD5 before
launching and report it.

Seeds 4127 and 4128 were used in Stage 2 to check *fleet* separation margins. Here they
are *run* seeds on the 4126 fleet. Different role, same numbers — say so in the write-up
so nobody later reads these as different fleets.

Varying the fleet seed is a separate robustness question and is **not** part of this
stage. Every comparison here is paired within one fleet, and changing the fleet would
break that pairing.

---

## Batch A — fill the table at `vmax_distance`, seed 4126. 4 runs.

| # | selector | `stage0_t_hat` | what claim it serves |
|---|---|---|---|
| A1 | **gpu** | tier_spec | does the selector help behind the filter? Stage 5 measured `gpu` **11.9% slower** than `baseline` at `vmax_elapsed`; this is the same test at the method's own `R_i` |
| A2 | cpu | tier_spec | a wrong spec degrades the filter as well as the score |
| A3 | baseline | fleet_mean | the low rung of the `T̂` ladder — 21 points below `tier_spec` at `vmax_elapsed` |
| A4 | observed | fleet_mean | the filter does not depend on which selector sits behind it |

Existing rows at `vmax_distance` that complete this table and must **not** be re-run:
`baseline / none` (Task 4, 7,525.2 s) and `baseline + tier_spec` (Stage 5b, 4,529.6 s).

### Gate — report Batch A before launching Batch B

Report the six-row table (T @ 4,000, waste, dispatch share per tier, the mix/waste
decomposition, rejection accuracy per tier, fallback rate) and **stop**.

Two things could change which cells deserve seeds, and both are cheap to check now and
expensive to discover later:

- **If A1 no longer shows `gpu` slower than `baseline`**, the selector claim changes
  shape and Batch B's `gpu` rows may need a different partner.
- **If A3's gap to `baseline + tier_spec` has collapsed from 21 points**, the `T̂`
  ladder is weaker at this `R_i` than at the old one, and that is the claim most in
  need of seeds rather than least.

Also stop and report if any Stage 5 stop condition trips: fallback > 10% of rounds,
conservation-law error > 3%, median `eligible_after_stage0` < C = 4, or plateau more
than 2 points below the 82.1–83.7% band.

---

## Batch B — seeds. 8 runs, seeds 4127 and 4128.

| cell | why it needs an error bar |
|---|---|
| baseline + none | every percentage in the thesis is measured against it |
| baseline + tier_spec + `vmax_distance` | the headline, −39.8% |
| gpu + tier_spec + `vmax_distance` | the claim that the selector does not help behind the filter — the smallest effect of the four and the most seed-sensitive |
| baseline + fleet_mean + `vmax_distance` | the spec-versus-no-spec gap, which is what makes the hardware signal non-redundant |

`cpu` and `observed` stay at one seed. `cpu`'s separation is 24–30% on total time —
far outside anything the noise floor could flip — and `observed + fleet_mean` answers a
structural objection rather than a numerical comparison. Say this explicitly in the
write-up rather than leaving them looking under-powered.

---

## What to report

### Per run
Everything Stage 5b reported: conservation law, T @ 4,000, waste overall and per tier,
dispatch share per tier per window, mix/waste decomposition, rejection accuracy in both
denominators, eligibility-window fraction per tier, fallback rate, pool sizes after each
gate, plateau, noise, TTA and rounds-to-target at 70 / 75 / 78 / 80%.

### Across seeds — this is the point of Batch B

For each of the four seeded cells: mean and sample standard deviation over the three
seeds of T @ 4,000, waste, xavier dispatch share, and the mix and waste channels.

Then, and more importantly:

**Ordering stability.** For every pairwise comparison that the thesis will make, state
whether the ordering holds in **all three seeds individually**, not merely on the mean:

| comparison | claim |
|---|---|
| `baseline + tier_spec` vs `baseline / none` | the filter is faster |
| `baseline + tier_spec` vs `baseline + fleet_mean` | the spec beats a spec-free `T̂` |
| `gpu + tier_spec` vs `baseline + tier_spec` | the selector does **not** help behind the filter |

A claim whose ordering flips in any seed is reported as undecided, not as a mean with a
standard deviation wide enough to cover both. Say which of the three it is for each row.

**Seed noise floor on T @ 4,000.** The spread across seeds of `baseline / none` is the
reference against which every other spread should be read. It has never been measured —
Stage 3's `_rep2` measured reproducibility (0.00%), not variance.

**Conservation law per seed.** If the law holds per seed at the same tightness
(≤ ~1%), that is independent evidence the seeds differ only in the way intended.

---

## Written down before the runs start

| # | prediction |
|---|---|
| 1 | Seed spread on T @ 4,000 is **small — under ~5%** — because the conservation law makes it a function of the dispatch mix, which is an average over ~4,000 dispatches. TTA spread will be much larger (Stage 3 measured ±6–21% on rounds-to-target) |
| 2 | All three orderings in the table above hold in **every individual seed** |
| 3 | A1 reproduces Stage 5's finding: `gpu + tier_spec` is slower than `baseline + tier_spec`, by roughly the same margin, and dispatches more xavier |
| 4 | A3's gap to `baseline + tier_spec` stays close to the ~21 points measured at `vmax_elapsed` — a better `R_i` should not change how much the `T̂` signal is worth |
| 5 | Waste stays ≤ ~1% in every filtered run, and the per-tier waste rates stay within the ranges Stage 5 measured |

Report each as held or not held. A failed prediction is a result.

Prediction 1 decides how every other number in the thesis may be quoted. If the seed
spread on T @ 4,000 turns out large, then three seeds are not enough for any of this
and the conclusion is that more are needed — report that plainly rather than quoting
means over three noisy draws.

---

## Do not

- change the fleet file, sigma, the fleet mix, the tier files, the training anchor, the
  trace, the radius, `v_max`, the dataset, the model, the optimiser, the scoring
  function, or the selection rule
- change `stage0_r_i` away from `vmax_distance` in any run of this stage
- vary the fleet seed — only the run seed
- re-run `baseline / none` at seed 4126, or `baseline + tier_spec` at seed 4126, or any
  Stage 5 run
- launch Batch B before Batch A has been reported
- report a mean across seeds without also reporting whether the ordering held in each
  seed separately
- tune anything to make a prediction come true
