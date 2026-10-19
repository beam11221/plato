# Task 2 — log the two terms of `V_i` separately

Two runs, `N=20, C=4, seed 4126, 4,000 rounds`, arms `baseline` and `gpu`,
otherwise identical to the Stage 3 cell. Both finished cleanly, zero
tracebacks, `.done` markers written. Raw per-candidate logs:
`stage4_evidence/vi_terms_raw_baseline.csv` (68,004 rows) and
`vi_terms_raw_gpu.csv` (68,004 rows) — every eligible candidate, every round,
not only the winner, via `vi_term_logger.py` (see that file's docstring for
how it guarantees the logged numbers cannot drift from what selection
actually used).

**`baseline` never scores anything** — it is plain random selection with no
`τ̂`/`G_i`/`V_i` at all, so its columns in the CSV are blank by design and it
plays no further part in this analysis. Items A–F below are about the two
terms of `V_i`, which only exist for `gpu`.

## The prior, restated from Task 1 (required here too)

`g_is_prior` = true while a client is still in
`FedAsyncPolarisAggregationStrategy.unexplored_clients`
(`fedasync_polaris_vi_server.py:279-293`). `g_prior_value` in that state
equals `G_i` itself: **`alpha × mean(measured G_i of clients that reported
this round)`**, `alpha = polaris_alpha = 10.0` (config default, same in this
run). It is not a fixed constant — it moves with whatever the currently-
reporting clients' norms average to. Before any client has ever reported,
every client starts at the separate fixed default `initial_gradient_bound =
0.5` (`fedasync_polaris_vi_server.py:169`).

## A. Coefficient of variation across candidates, `s(τ̂)` vs `G_i`, per window

Per round: CV = stdev/mean across that round's scored candidates, for each
term separately; table shows the window-average of the per-round CV, plus
the pooled min/max (range) of each term across the whole window.

| window | mean CV(s) | mean CV(G) | range(s) | range(G) |
|---|---|---|---|---|
| 1-100 | 0.364 | 0.536 | [0.349, 0.822] | [0.500, 41.204] |
| 100-250 | 0.364 | 0.133 | [0.349, 0.822] | [1.424, 2.793] |
| 250-500 | 0.366 | 0.220 | [0.349, 0.822] | [0.979, 3.140] |
| 500-1000 | 0.361 | 0.392 | [0.349, 0.822] | [0.317, 3.231] |
| 1000-2000 | 0.361 | 0.602 | [0.349, 0.822] | [0.115, 3.123] |
| 2000-4000 | 0.361 | 0.696 | [0.349, 0.822] | [0.056, 4.120] |

**CV(s) is dead flat at ~0.36 for the entire run.** CV(G) is U-shaped: high
early (0.536, still settling out of the initial explore phase — the range
[0.5, 41.2] includes the 10× prior at its most extreme before the observed-
norm mean stabilises), drops to its lowest in the 100-500 window (0.13-0.22,
once every client has been explored at least once and norms haven't yet
diverged), then **climbs past CV(s) and keeps climbing** through the rest of
the run, ending nearly double CV(s) (0.696 vs 0.361) in the last window.
**`G_i` is the term deciding the ranking late in the run; `s(τ̂)` is not** —
it has essentially no room left to vary, having had none all along.

## B. Spearman ρ: `V_i` vs `s(τ̂)`, `V_i` vs `G_i`, per window

Pooled over all candidate-round rows in the window.

| window | ρ(V, s) | ρ(V, G) |
|---|---|---|
| 1-100 | 0.546 | 0.632 |
| 100-250 | **0.874** | -0.222 |
| 250-500 | 0.756 | -0.152 |
| 500-1000 | 0.269 | 0.456 |
| 1000-2000 | -0.041 | **0.778** |
| 2000-4000 | -0.053 | **0.838** |

A clean crossover, not a gradual one: `V_i`'s ranking is driven almost
entirely by `s(τ̂)` in the 100-500 window (ρ up to 0.87, `G_i` actually
*negatively* correlated there), and almost entirely by `G_i` from round 1000
on (ρ up to 0.84, `s(τ̂)` correlation collapses to ~0). Do not conclude from
this table alone (per this task's own instruction) — see D and F below for
the two readings that disambiguate it.

## C. Mean `G_i` by `rounds_since_last_selected` — **the direct test is undecided, and the confound matters**

Whole run, pooled:

| bucket | n | mean G_i | median G_i |
|---|---|---|---|
| 1-5 | 9,697 | 1.084 | 0.677 |
| 6-15 | 22,930 | 1.113 | 0.869 |
| 16-50 | 28,332 | 0.988 | 0.726 |
| >50 | 7,044 | 0.678 | 0.457 |

Pooled across the whole run this is **not** monotonically rising with
neglect — it peaks at the 6-15 bucket and *falls* for the most-neglected
clients. But `G_i`'s overall scale shrinks by ~3.5× from early to late
training (a separate, large effect — see the split below), which confounds
any pooled read. Split by phase:

| phase | 1-5 | 6-15 | 16-50 | >50 |
|---|---|---|---|---|
| early (round ≤ 1000) | 2.281 (n=2654) | 1.974 (n=5808) | 1.836 (n=7197) | 1.698 (n=1344) |
| late (round > 1000) | 0.633 (n=7043) | 0.820 (n=17122) | 0.700 (n=21135) | 0.437 (n=5700) |

Neither phase shows a clean rise with neglect either — early is **monotonically
decreasing** with `rounds_since_last_selected` (the opposite of the
hypothesis), late is hump-shaped (peaks at 6-15, falls on both sides). **The
literal mechanism "a client's `G_i` rises the longer it goes unselected" is
not supported by this direct test, in either phase.**

## D. Mean `s(τ̂)` per tier per window — **does not weaken; it is a constant**

| window | orin | thor | xavier |
|---|---|---|---|
| every window, 1-100 through 2000-4000 | 0.7209 | 0.8218 | 0.3490 |

Identical to four decimal places in **every** window. `s(τ̂)` for `gpu` is a
fixed per-tier constant for the entire 4,000-round run — it never once moves,
because the tier-mean training-time belief and the steady-state `T_round`
it's divided by both stabilise within the first few dozen rounds. This
directly answers this item's question: **the fast-tier preference does not
weaken over time — it holds at full, unchanging strength throughout, and
whatever changes ranking late in the run is entirely on `G_i`'s side.** This
is the cleanest single finding in this task: A and B's crossover cannot be an
`s(τ̂)` effect, by construction — there is no `s(τ̂)` dynamic to have one.

## E. Share at the `G_i` prior; CV among prior vs. measured; time to full coverage

| window | share at prior | CV(G｜prior) | CV(G｜measured) |
|---|---|---|---|
| 1-100 | 12.3% | 0.504 | 0.299 |
| 100-250 | 0.0% | n/a | 0.140 |
| 250-500 | 0.0% | n/a | 0.225 |
| 500-1000 | 0.0% | n/a | 0.413 |
| 1000-2000 | 0.0% | n/a | 0.637 |
| 2000-4000 | 0.0% | n/a | 0.813 |

Every client has been selected (and therefore measured) at least once by
**round 44** (first-selected rounds ranged 1–44 across the 20 clients, most
within the first 25). So the 10× prior only exists in the first ~44 rounds of
a 4,000-round run — it cannot be the mechanism behind the mid/late-run
dynamics in A/B/C, which all happen long after every client has a real
measured `G_i`. What *does* grow steadily after exploration finishes is the
**spread among measured values themselves** — CV(G｜measured) rises
monotonically from 0.14 to 0.81 across the run, a 5.8× increase, which is the
real driver behind A and B's late-run numbers.

## F. Counterfactual ranking per window

For each round, top-`C` by `s(τ̂)` alone and top-`C` by `G_i` alone,
compared against the actual selected set.

| window | frac. differ (s alone) | frac. differ (G alone) |
|---|---|---|
| 1-100 | 0.970 | 0.920 |
| 100-250 | 0.927 | 0.953 |
| 250-500 | 0.916 | 0.952 |
| 500-1000 | 0.940 | 0.924 |
| 1000-2000 | 0.943 | 0.891 |
| 2000-4000 | 0.942 | 0.876 |

**Read this table's trend, not its absolute level.** Task 1 found real
selection is a `V_i`-weighted lottery, not deterministic top-`C`
(`fedasync_polaris_vi_selection.py:276-294`), so a deterministic top-`C`-by-
one-factor set will disagree with an actual stochastic draw most of the time
regardless of which factor dominates `V_i` — that mechanical noise floor
(87-97% "differ" throughout) is why this table cannot answer "how often does
the staleness term change the decision" as cleanly as B does. What it can
still show is *direction*: `frac_differ(G_alone)` falls from 0.92-0.95 in the
first three windows to 0.876 in the last — `G_i`-alone gets *closer* to
matching actual outcomes over time — while `frac_differ(s_alone)` stays flat
around 0.92-0.97 throughout. That direction corroborates B (G_i becoming the
better single predictor of the real outcome late in the run) but should be
read as a weak secondary confirmation, not primary evidence — B is the
cleaner signal here.

## Verdict: **supported in aggregate mechanism, undecided on the specific causal story**

The working hypothesis had two parts. Taking them separately:

1. *"`s(τ̂)` pushes selection toward fast clients"* — **supported, and
   stronger than stated**: it is not just a push, it is a fixed, unmoving
   ~2.35× preference for thor over xavier (0.822 vs 0.349) that holds for
   all 4,000 rounds without any decay of its own (D).

2. *"`G_i` rises on clients that have been skipped, pushing back, so the
   mix relaxes toward the fleet mix"* — **the literal causal mechanism is
   not supported**: `G_i` does not rise with `rounds_since_last_selected` in
   a direct test, in either training phase (C). What *is* clearly true, and
   matches the decay Stage 3 observed, is that `G_i`'s cross-client spread
   and its correlation with `V_i` both grow steadily through the run, while
   `s(τ̂)`'s do not move at all (A, B, D) — so **something about `G_i`
   growing more decisive over time is real and well-evidenced**, it is just
   not the "neglect raises this client's score" story as literally stated.
   A plausible alternative this data cannot rule in or out: measured
   gradient norms simply diverge more across clients as local models drift
   further from the global one under non-IID data as training progresses,
   independent of each client's own selection history.

One further limitation, carried over from Task 1 and not resolved here:
because real selection is a weighted lottery, some of the "decay" Stage 3
measured could also be the lottery mechanism itself (a slow client's nonzero
selection probability, not a change in its `G_i`) rather than `G_i` dynamics
at all. This task's logging cannot separate that contribution from `G_i`'s
growing correlation — both operate on every round simultaneously. Flagged as
open, per the task's allowance that "undecided" is an acceptable answer.
