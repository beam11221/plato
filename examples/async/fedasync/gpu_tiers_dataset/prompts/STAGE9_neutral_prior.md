# Stage 9 — neutral priors for never-seen vehicles. Log check, then 22 runs + 1 invariance run.

Launch after Stage 8 has finished, or alongside it only if the four-concurrent-runs
ceiling allows. Stage 9 pairs every cell with a Stage 7b / Stage 8 cell of the same
seed, so it needs Stage 8's seeds 4127 and 4128 to exist.

Do not touch anything under `plato/`.

---

## Step 0 — a log-only check on Stage 7b/8, before anything else. No runs.

Stage 8 found `gpu − observed` has the same sign in all six seed × regime cells
(open −1.93 / −0.24 / −1.16%, closed −2.73 / −1.69 / −2.49%), and that the gap is
**larger closed than open** — so it is not the cold-start channel. Candidate
explanation: `observed`'s learned staleness cannot separate orin from thor (both settle
at τ = 1, s = 0.707) while the spec gives 0.721 vs 0.822, and realised staleness is a
noisy queue-dependent signal where the spec value is exact per tier. If so, `gpu`'s
edge is **signal precision among the fast tiers**.

From the existing logs of `gpu`, `observed` and `baseline`, both regimes, seeds
4126 / 4127 / 4128 (α = 10), report:

1. dispatch share xavier / orin / thor, overall and restricted to dispatches with
   ≥ 1 observation, per window (1–1000 / 1000–2000 / 2000–4000)
2. **thor : orin dispatch ratio** per arm, and each arm's ratio relative to
   `baseline`'s (which reflects availability only)
3. `observed`'s `τ̂` distribution by tier at dispatch — median and IQR for orin and
   for thor, and how often an orin and a thor candidate in the same round get the same
   `τ̂`
4. `gpu − observed` decomposed with the conservation law into the mix channel and the
   waste channel, and the mix channel split into the part from xavier share and the
   part from the orin/thor split

Hypothesis confirmed if `gpu`'s thor : orin ratio exceeds `observed`'s in every seed,
and the orin/thor part of the mix channel accounts for most of the gap. Refuted if the
ratios match and the gap comes from xavier share or waste. Report either way.

This does not gate the runs below. Report it, then proceed.

---

## Why this stage exists — written before any result

Stage 7b found that in the open population **99.4–100% of admitted vehicles receive a
first dispatch in every Polaris arm**, whatever their tier. The cause is the treatment
of never-seen vehicles:

- every arm gives a never-seen vehicle `G_i = α × mean(measured G_i)` with the Polaris
  default **α = 10**
- `observed` additionally gives it `τ̂ = 0.01`, so `s = 0.995` — higher than any vehicle
  it has already seen

Both are **optimistic priors**: an exploration device. In Polaris/Oort they are
designed for a closed pool, where the exploration phase ends once every client has been
tried — in Stage 4 that happened by round 44. **In an open population the exploration
phase never ends**, because new vehicles keep arriving. The optimistic prior then
dominates selection for the whole run: any newcomer outranks every known vehicle
(a new xavier at `V = 0.349 × 10 = 3.49` beats a known thor at `0.822 × 1 = 0.82`), and
selection degenerates to "newest first". Spec information can then only reorder the
newcomers, never decide which of them train.

This is a closed-pool calibration carried into an open-pool setting — the same theme
as C3. The principled correction is a **neutral prior**: assume a never-seen vehicle is
average until observed. This is argued from the design of the exploration mechanism,
not from any outcome, and it is recorded here before the runs.

It is applied to every arm equally. The Polaris-default cells (α = 10, from Stage 7b and
Stage 8) are kept and reported alongside, so the result is a comparison of two priors,
not a replacement of one.

---

## Step 1 — two config knobs, defaults unchanged

| knob | default (= Stage 7b) | Stage 9 value |
|---|---|---|
| `g_prior_alpha` | 10 | **1** — a never-seen vehicle's `G_i` is the current mean of measured `G_i` |
| `observed_unseen_tau` | `const_0.01` | **`fleet_running_mean`** — the mean of every staleness observation delivered so far, across all vehicles; initialised at `C − 1 = 3` before the first delivered report |

Notes:

- `C − 1` as the initial value is knowable from configuration and is what Stages 3–7b
  measured mean staleness to be. It avoids re-introducing the optimistic `0.01` during
  the first rounds.
- `fleet_running_mean` is the reactive arm's spec-free analogue of the `fleet_mean` `T̂`
  used in Stage 5–6 for the filter. For never-seen vehicles, the question becomes the same
  one C1 answered for filtering: **does a spec beat a fleet average?**
- `initial_gradient_bound = 0.5` before any report is a start-up transient and stays as is.
- `gpu` and `cpu` compute `τ̂` from spec for every vehicle, seen or not; only
  `g_prior_alpha` affects them.
- `baseline` reads neither knob.

With both knobs at their defaults, every existing config must behave exactly as before.
Add tests that assert this, using the real aggregation strategy and selector, as in 7b.

## Step 2 — pre-flight, 100 rounds per arm and regime, seed 4126

All Stage 7b pre-flight checks at 100%, plus:

- for every dispatch of a never-seen vehicle under `g_prior_alpha = 1`: logged `G_i` ==
  the current mean of measured `G_i` (not 10×)
- for every `observed` dispatch of a never-seen vehicle: logged `τ̂` == the running mean
  at that moment
- `tau_hat` and `G_i` logged == the values `V_i` actually used
- **baseline invariance:** one full 4,000-round `baseline_open` run at seed 4126 with
  the Stage 9 config reproduces Stage 7b's `baseline_open` (T @ 4,000, dispatch
  sequence). This proves the knobs do not touch the arm that ignores them, and lets all
  six `baseline` cells from 7b/8 be reused

Report, then launch.

## Step 3 — the runs

| arm | `p_recur` | seeds | runs |
|---|---|---|---|
| gpu | 0, 1 | 4126, 4127, 4128 | 6 |
| cpu | 0, 1 | 4126, 4127, 4128 | 6 |
| observed | 0, 1 | 4126, 4127, 4128 | 6 |

**18 runs**, all with `g_prior_alpha = 1` and `observed_unseen_tau = fleet_running_mean`.
Everything else identical to Stage 7b/8: same code, physics seed pinned at 4126, run
seed varying, trace, `M = 20`, `C = 4`, α = 0.8, 4,000 rounds, `coverage_gated`,
Stage 0 off, startup `T_round` = 2.0808 s.

`baseline` is reused from Stage 7b (4126) and Stage 8 (4127, 4128), conditional on the
invariance check.

`cpu` at seeds 4127/4128 under α = 10 was not run in Stage 8. **Run those four as
well** (`cpu_open`, `cpu_closed` × 4127, 4128, α = 10, Stage 8 code and seed split) so
every Stage 9 cell has an α = 10 partner of the same seed. **Total: 22 runs.**

Run at most four at once.

---

## Step 4 — what to report

Per run, the Stage 7b / Stage 8 set, plus:

- **membership:** fraction of admitted vehicles ever dispatched, per tier. Under α = 10
  this was 99.4–100% everywhere. This is the quantity the stage exists to move
- **first-dispatch tier mix:** xavier / orin / thor share among each vehicle's first
  dispatch
- **wait to first dispatch** per tier, mean and median
- **per-vehicle fairness:** distribution of dispatch count per admitted vehicle, share
  of vehicles never dispatched, Jain's index over admitted vehicles
- `G_i` at dispatch by tier and by seen / never-seen, per window
- for `observed`: the running-mean `τ̂` over time

### The comparisons

Every comparison paired by seed and checked in **each seed individually**. A sign that
flips in any seed is reported as undecided.

| # | comparison | regime |
|---|---|---|
| A | `gpu` vs `observed`, α = 1 | open — **the C2 test** |
| B | `gpu` vs `observed`, α = 1 | closed |
| C | each Polaris arm, α = 1 vs α = 10 | open, then closed |
| D | each Polaris arm vs `baseline`, α = 1 | open and closed — does C3's sign flip survive the neutral prior? |
| E | `cpu` vs `baseline` | both |

---

## Written down before the runs start

| # | prediction |
|---|---|
| 1 | **Membership moves.** Under α = 1 the ever-dispatched fraction drops well below 99% in the Polaris arms in the open regime, and under `gpu` it drops most for xavier |
| 2 | **C2 appears.** In the open regime `gpu` beats `observed` on T @ 4,000 in all three seeds, by more than Stage 7b's −1.93%. Mechanism: `gpu`'s first dispatches are markedly less xavier-heavy than `observed`'s, because `gpu` ranks newcomers by spec while `observed` gives them all the fleet-average `τ̂` |
| 3 | **Closed is almost unaffected.** At `p_recur = 1` a vehicle is never-seen only on its slot's first use, so α = 1 and α = 10 give results within seed noise |
| 4 | `cpu` remains worst in both regimes |
| 5 | **Fairness cost.** Under α = 1, `gpu` in the open regime leaves a measurable share of xavier vehicles never dispatched; plateau stays inside 82.1–83.7% (the partition is redundant at α = 0.8, as noted in §13) |
| 6 | C3's sign flip against `baseline` persists under α = 1: churn still caps the `G_i` divergence, so the Polaris arms stay ahead of `baseline` in open and behind it in closed |

**Prediction 2 is the point.** If it fails — `gpu` still ties `observed` even when
newcomers must compete — then the proactive signal has no outcome value in this fleet,
the reason is the 15× tier separation (one observation classifies a vehicle), and C2 is
closed as a null with a complete explanation. Report that plainly.

**Prediction 5 is the price.** A result for Prediction 2 is reported together with its
fairness cost, never on its own.

---

## Stop conditions

- any pre-flight check below 100%, or the baseline invariance run differs from 7b
- the occupancy digest differs from 7b/8 in any run
- conservation-law error > 3%
- plateau more than 2 points below 82.1% in any run — dropping newcomers would then be
  damaging the model, which is a different and more serious finding

## Do not

- change `α` or the unseen-`τ̂` rule for one arm only
- change `V_i`, the selection rule, the GPU equation, startup `T_round`, sigma, tier
  ratio, `M`, the trace, physics seed, dataset, model or optimiser
- enable Stage 0
- try any value of `g_prior_alpha` other than 1 in this stage
- report the C2 result without the fairness numbers next to it
- tune anything to make a prediction come true
