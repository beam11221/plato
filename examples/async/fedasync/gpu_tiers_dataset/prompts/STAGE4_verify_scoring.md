# Stage 4 — verify the scoring function, and test one prediction. No sweep.

Stage 3's 23 runs are analysed and accepted. Three questions about `V_i = s(τ̂_i) · G_i`
and about the closed-pool conservation law are now blocking. Answer them, produce the
evidence, and **stop**. Do not start any campaign beyond the runs listed here.

Tasks 1–3 are verification and analysis. Task 4 is a small pre-registered experiment
cell and does not depend on Tasks 1–3; it may run in parallel with them.

Do not touch anything under `plato/`. Everything here lives in the example folder.

Context you need, from the Stage 3 analysis:

- Every run issues exactly **one dispatch per round**. Total simulated time is
  predicted to within 0.1–1.3 points by the **dispatch tier mix alone** — so in this
  closed pool the selector changes the *order* of work, not the *amount*.
- Consequently `gpu`, `observed` and `gpu_oracle` run ~20% ahead of `baseline` at
  round 250 and end up ~2% behind at round 4,000. The advantage is a decaying
  transient. `cpu` is the only arm that changes the mix, and its penalty is permanent.
- At C=1 measured report staleness is **exactly 0.000 for all 4,000 reports in every
  arm**, yet `gpu` still runs 11.6% faster per round than `baseline` and picks a
  different tier mix (Xavier 44.1% vs 49.1%).

---

## Task 1 — what quantity actually feeds `s(·)`? (code reading, no runs)

If staleness is identically zero for every client, `s(τ̂_i) = 1/√(1+τ̂_i)` should be
identical for every client and `V_i` should reduce to `G_i` alone — in which case all
four arms would select identically at C=1. They do not. So `τ̂` is not the quantity the
scoring function's definition says it is, or the two are conflated somewhere.

Trace the value from where it is computed to where it enters `s(·)` and report, with
file and line numbers:

1. The **expression** each arm computes for `τ̂_i`, written out.
2. Its **unit** — predicted rounds of staleness, predicted training seconds, a
   normalised score, or something else — and, if it is a time, what converts it to
   rounds and what `T_round` that conversion uses.
3. Whether `gpu`, `cpu` and `observed` differ **only** in the estimate of training
   time, or also in the transformation applied afterwards. `observed`'s estimate is
   known to live in round-space while `gpu`'s is derived from seconds; say exactly
   where those two meet.
4. Whether anything **other** than `s(τ̂)` and `G_i` influences selection —
   normalisation, exploration, tie-breaking, a rotation or fairness constraint, an
   eligibility window, a cooldown. If a fairness or rotation mechanism exists, quote it
   in full; it is a candidate explanation for the decay described above.

Write this to `stage4_evidence/scoring_trace.md`.

**Do not change any behaviour in this task.** If the implementation turns out to
disagree with the written definition of `V_i`, report the discrepancy and stop — the
decision whether to change the code or change the definition is not yours to take
alone.

---

## Task 2 — log the two terms of `V_i` separately (short instrumented runs)

Working hypothesis to test: `s(τ̂)` pushes selection toward fast clients, while `G_i`
rises on clients that have been skipped and pushes back, so the dispatch mix relaxes
toward the fleet mix and the early advantage decays.

Add per-dispatch logging in the example folder — a new file, not a change to the
selector's decisions:

```
round, client_id, tier, selected, s_tau_hat, tau_hat_raw, G_i, V_i,
rounds_since_last_selected, times_selected_so_far, g_is_prior, g_prior_value
```

Log **every candidate scored in that round**, not only the winner. Without the
non-selected candidates the margin cannot be reconstructed.

`g_is_prior` is a boolean: true when `G_i` is the default/unmeasured value used for a
client that has not yet returned an update, false when it is a measured gradient norm.
`g_prior_value` is whatever that default is. Both matter for a decision that comes
after this task — whether the exploration prior on `G_i` is worth sweeping — and
neither can be recovered later without re-running, so log them now. Report what the
prior is and where it is set.

Runs: `N=20, C=4, seed 4126, 4,000 rounds`, arms `baseline` and `gpu` only — two runs.
The full budget is used so the log spans the crossover at round ≈2,400, where the time
advantage returns to zero. Every other parameter stays at the Stage 3 values.

Report, in `stage4_evidence/vi_terms.md`:

Windows throughout: 1–100 / 100–250 / 250–500 / 500–1000 / 1000–2000 / 2000–4000.

- **A.** Coefficient of variation across candidates of `s(τ̂)` and of `G_i`, separately,
  per window. Whichever term varies more across candidates is the one deciding the
  ranking. Also report the raw range of each.
- **B.** Spearman ρ between `V_i` and each of `s(τ̂)`, `G_i`, per window. Report both;
  do not conclude from ρ alone.
- **C.** Mean `G_i` grouped by `rounds_since_last_selected` (buckets 1–5, 6–15, 16–50,
  >50). If `G_i` rises with time since selection, the rebalancing hypothesis is
  supported; if it is flat, it is not, and something else drives the decay.
- **D.** Mean `s(τ̂)` per tier per window — does the fast-tier preference weaken over
  time, or does it hold while `G_i` overrides it? These two are different mechanisms
  and the distinction matters.
- **E.** Share of candidates sitting at the `G_i` prior, per window, and the CV of
  `G_i` among prior-valued candidates versus measured ones. Separately: how many rounds
  pass before every client in the fleet has been selected at least once.
- **F.** Counterfactual ranking, per window: what fraction of rounds would have chosen
  a **different** client if the winner had been picked by `s(τ̂)` alone, and what
  fraction if picked by `G_i` alone. This measures directly how often the staleness
  term changes the decision, which is the quantity the thesis claim rests on.

State whether the hypothesis is supported, contradicted, or undecided. "Undecided" is
an acceptable answer.

---

## Task 3 — time-resolved dispatch share (existing logs, no runs)

Aggregate dispatch share over 4,000 rounds is now known to be blind to the transient:
`gpu` and `baseline` look identical on it while their cumulative time differs by 19% at
round 250.

From the existing `dispatch_ledger.csv` of the Stage 3 runs, recompute dispatch share
per tier in windows 1–250 / 250–500 / 500–1000 / 1000–2000 / 2000–4000, for every arm
in the N=20 C=4 and C=8 cells.

Report as `stage4_evidence/dispatch_share_windowed.md`, plus a CSV.

Expected if the conservation law holds: `gpu`, `observed` and `gpu_oracle` start with a
fast-tier-heavy mix and drift toward the fleet mix, with the cumulative share crossing
50/35/15 around the same round where the time advantage crosses zero (≈2,400 at C=4,
≈1,900 at C=8). `cpu` should stay Xavier-heavy throughout. **If the windowed shares are
flat instead, the conservation law has a different explanation and Task 2's hypothesis
is wrong — say so plainly rather than fitting the result to the expectation.**

---

---

## Task 4 — does the conservation law survive coverage gating? (SUMO, closed population)

The §2 conservation law — total simulated time is set by the dispatch tier mix alone,
so the selector reorders work rather than reducing it — was derived from a closed pool
in which **every dispatch eventually returns a usable update**. Under coverage gating,
a vehicle that leaves the RSU before its upload completes produces nothing, so a
selector that avoids doomed dispatches changes the *amount* of useful work.

That is currently an inference, not a measurement. This task tests it.

**This cell keeps identity persistence** (closed-loop SUMO, vehicles respawn with the
same ID). That is deliberate: it isolates the effect of wasted work while leaving the
reactive estimator at full strength. Open population is a separate cell and is not part
of this task.

### Step A — pre-run gate, no FL training

From the SUMO trace alone, compute per round the number of vehicles satisfying
`R_i ≥ T̂_i + T_comm`, where `R_i = (x_exit − x_i)/v_i` and `T̂_i` is the tier-mean
training time.

Report: median, 10th and 90th percentile of feasible-pool size per round; the
distribution of `R_i / T̂_i` across the fleet; and the fraction of rounds in which the
feasible pool is smaller than `2 × C`.

**Stop and report if the median feasible pool is below `2 × C`.** At that point
selection is near-forced, every arm behaves identically by construction, and the cell
measures nothing. The fix is to change the *mobility* timescale — road length or speed
— never the training anchor. Bring the numbers back before changing anything.

### Step B — keep the training-time model byte-identical to Stage 3

Use the **same fleet file** as the Stage 3 N=20 cell
(`n20_x10-o7-t3_xa15_p2500_j0p149_s4126`), so client *i* has exactly the same `sleep_i`
in both cells. Same tier table, same sigma, same anchor, same partition size, same
alpha, same seed, same 4,000 rounds.

If the existing SUMO config uses a different anchor, **do not rescale the anchor to fit
the trace.** Rescale the trace instead, and say what was changed. The surgical-to-SUMO
comparison is only interpretable if the training-time model is untouched.

### Step C — the runs

`N=20, C=4, seed 4126, 4,000 rounds`, arms `baseline, gpu, cpu, observed`, Stage 0
feasibility filter **OFF**, under both upload policies:

| upload_policy | arms | runs |
|---|---|---|
| `coverage_gated` | baseline, gpu, cpu, observed | 4 |
| `always_accept` | baseline, gpu, cpu, observed | 4 |

**8 runs.** `always_accept` is not optional. SUMO changes two things at once — the
available pool shrinks, and work can be wasted. Without `always_accept` the two cannot
be separated and no result from this cell is attributable.

Stage 0 stays off so that any waste reduction is attributable to the scoring function
itself, and so that the Stage 0 counterfactual can be recomputed from these logs. The
earlier Stage 0 counterfactual (98.6% waste captured, 0.0% false positives) was
computed on oracle-branch runs and is **no longer valid**; recomputing it here on the
fixed harness is part of this task.

### Step D — what to report

Per run, alongside the Stage 3 summary fields:

- dispatches, **wasted dispatches** (upload rejected), useful dispatches, and the
  wasted fraction — overall, per tier, and per window (1–250 / 250–500 / 500–1000 /
  1000–2000 / 2000–4000)
- cumulative `sim_time_reconstructed` vs the `coverage_gated` baseline at rounds
  100 / 250 / 500 / 1000 / 2000 / 4000, the same table as the Stage 3 analysis
- **useful work per unit time**: useful dispatches divided by reconstructed seconds,
  per window
- the Stage 0 counterfactual replayed from these logs, per arm: waste caught, viable
  work killed, waste remaining

### Step E — written down before the runs start

| # | prediction |
|---|---|
| 1 | Under `always_accept`, all four arms reproduce the Stage 3 pattern — advantage decays to ≈0 by round 4,000. Mobility alone does not break the conservation law |
| 2 | Under `coverage_gated`, `gpu`'s advantage **does not return to zero** by round 4,000, because avoided waste is not repaid |
| 3 | `gpu` and `observed` separate on wasted fraction but not on tier mix — the reactive estimator works in round-space and cannot express a feasibility predicate |
| 4 | `cpu` wastes the most, because it steers into the slow tier and slow vehicles are the ones that miss their window |

**Prediction 2 is the decision gate for the whole open-population plan.** If `gpu`
still converges to the baseline under `coverage_gated`, then wasted work does not break
the conservation law, and the argument that open population will produce a durable
advantage is wrong. Report that plainly if it happens — it would save weeks.

---

## Reporting

Four evidence files, one short summary message. State for each task: what was found,
what remains unknown, and whether any finding contradicts the Stage 3 analysis
summarised at the top of this prompt. A contradiction is a result, not something to
reconcile.

## Do not

- change sigma, the fleet mix, the tier files, the training anchor, the dataset, the
  model, the optimiser, or any selector's decision logic
- launch any run beyond the two in Task 2 and the eight in Task 4
- turn on the Stage 0 filter in Task 4
- edit anything under `plato/`
- tune anything to make a hypothesis come true
