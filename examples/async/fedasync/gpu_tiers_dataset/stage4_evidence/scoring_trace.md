# Task 1 — what quantity actually feeds `s(·)`?

Code read only. Nothing executed, nothing changed. All line numbers refer to
the current checkout of `examples/async/fedasync/` (paths given relative to
that directory).

## 1. The expression each arm computes for `τ̂_i`

Two completely different code paths, selected by `[server] staleness_predictor`
(`fedasync_polaris_vi_server.py:395-417`):

**`gpu`, `cpu`, `gpu_oracle`** (the three members of `SECONDS_PREDICTORS`,
`fedasync_polaris_vi_server.py:61`) all run the *same* selector code and differ
only in which `tiers.csv` / `oracle_sleep_csv` `gpu_tier_sleep.py` resolves
underneath them:

1. `T_pred_i` (seconds) = `gpu_tier_sleep.gpu_tops_predict_seconds(cid, dataset_size, epochs)`
   → `_client_seconds()` (`gpu_tiers/gpu_tier_sleep.py:487-514`) → either the
   tier-spec formula `_predicted_seconds()` (`gpu_tiers/gpu_tier_sleep.py:397-413`,
   `gpu`/`cpu`) or the per-client oracle CSV lookup (`gpu_oracle`,
   `gpu_tiers/gpu_tier_sleep.py:504-514`).
2. `τ̂_i` (rounds) = `gpu_tier_sleep.gpu_tops_predict_staleness_rounds(T_pred_i, busy_residuals, T_round)`
   (`fedasync_polaris_vi_selection.py:256-259` calling
   `gpu_tiers/gpu_tier_sleep.py:861-911`) — **not** a plain division. It is a
   two-phase discrete-event simulation: phase 1 counts how many of the
   *currently busy* clients' known residual times finish before `T_pred_i`;
   phase 2 adds a steady-state tail `remaining * phase2_scale / T_round`. Only
   when `busy_residuals` is empty (first round, or `wall_time` not yet set)
   does it fall back to the plain ratio `seconds_to_rounds(T_pred_i, T_round) =
   T_pred_i / T_round` (`fedasync_polaris_vi_selection.py:260-262`,
   `gpu_tiers/gpu_tier_sleep.py:810-824`).

**`observed`** (`fedasync_polaris_vi_server.py:415-417`, the
`else` branch — the only remaining value `resolve_staleness_predictor` can
return):

`τ̂_i = _observed_staleness(cid, context)` (`fedasync_polaris_vi_selection.py:60-69`)
— reads `context.state["polaris"]["local_stalenesses"][cid-1]`, i.e. the raw
`τ_actual` PLATO attached to that client's *most recent report*
(`update.staleness`, stored verbatim at
`fedasync_polaris_vi_selection.py:136-137`, no offset added). Default `0.01`
for a client that has never reported. There is no simulation and no seconds
anywhere in this path.

`baseline` computes no `τ̂` at all — it is
`fedasync_selection.FedAsyncClientSelectionStrategy`
(`fedasync_selection.py:20-60`), plain uniform random sampling over clients not
currently training. `V_i` is never evaluated for this arm.

## 2. Unit

`gpu` / `cpu` / `gpu_oracle`: `T_pred_i` starts as **predicted training
seconds** (from a hardware spec or an oracle CSV), then is converted to
**predicted rounds of staleness** by
`gpu_tops_predict_staleness_rounds()`. The `T_round` it divides by (in the
steady-state tail, and in the seconds→rounds fallback) is
`compute_pool_T_round_seconds()` (`gpu_tiers/gpu_tier_sleep.py:709-807`), which
implements the **"queue" model**: `T_round = E[T] / min(P, n)`
(`gpu_tiers/gpu_tier_sleep.py:622-623`), `E[T]` the arithmetic mean training
time over the supplied pool and `P = clients_per_round`. (A "min" model also
exists, `t_round_model()`-selected, but every config in this project uses
`"min"` only as the pre-`t_round_model()`-key legacy default; the logs
(`[GpuTierSleep] T_round model: queue`) confirm `"queue"` is what actually ran.)

`observed`: `τ̂_i` is **already in rounds** — it is literally
`current_round − round_when_selected` from the previous cycle
(`update.staleness`, set by PLATO's base server, read back unmodified). No
conversion, no `T_round`, no seconds anywhere in this arm's estimate.

## 3. Do `gpu`, `cpu`, `observed` differ only in the training-time estimate, or also in the transformation?

**Also in the transformation — the two families meet only once, at the score
itself.** `gpu`/`cpu` share 100% of the code from `_predicted_seconds()`
through `gpu_tops_predict_staleness_rounds()`; they differ only in the
*content* of `tiers.csv` (a data difference, not a code difference — the
README's claim holds). `observed` never touches any of that code. The two
families are computed by disjoint functions and only become comparable at
`fedasync_polaris_vi_selection.py:266`:

```python
v_i = g_i / math.sqrt(1.0 + max(tau_hat_rounds, 0.0))
```

Everything upstream of that one line — whether `tau_hat_rounds` came from a
residual-aware discrete-event simulation seeded by hardware specs, or from a
raw memory read of last round's outcome — is invisible to the score. This is
exactly the asymmetry the Stage 3 summary already inferred (`observed`'s
estimate "lives in round-space", `gpu`'s "is derived from seconds"); this
trace locates the meeting point precisely at line 266, with no shared
intermediate representation before it.

## 4. Does anything other than `s(τ̂)` and `G_i` influence selection?

**Yes — two things, and one of them contradicts this module's own docstring.**

### 4a. Selection is not deterministic top-K; it is V_i-weighted random sampling without replacement

The module docstring says (`fedasync_polaris_vi_selection.py:4-17`):

> "The top-K clients by V_i are selected (deterministic)"

and the class docstring repeats it (`fedasync_polaris_vi_selection.py:76-81`):

> "Deterministic top-K selection by V_i = G_i / sqrt(1 + τ̂_i)... then replaces
> random sampling with a deterministic sort on V_i scores."

The actual `select_clients()` body does not do this. Quoted in full
(`fedasync_polaris_vi_selection.py:276-294`):

```python
# Stochastic V_i-weighted sampling (without replacement).
# Sampling proportional to V_i avoids the tier-clustering that arises
# when all G_i are equal and deterministic top-K always fills every slot
# with the fastest tier (Thor).  High-scoring clients are still selected
# with higher probability, preserving the Polaris intent.
scores_arr = np.array(
    [max(scores[cid], 0.0) for cid in eligible], dtype=float
)
total_weight = float(scores_arr.sum())
if total_weight > 0 and effective_count < len(eligible):
    probs = scores_arr / total_weight
    chosen = np.random.choice(
        len(eligible), size=effective_count, replace=False, p=probs
    )
    selected = [eligible[int(i)] for i in chosen]
else:
    # All scores zero, or selecting the entire pool: fall back to top-K.
    sorted_eligible = sorted(eligible, key=lambda c: scores[c], reverse=True)
    selected = sorted_eligible[:effective_count]
```

So every round, every eligible client — including low-`V_i` slow-tier
clients — has a strictly positive chance of being dispatched, proportional to
its share of total `V_i` mass across the eligible pool. This is a genuine
disagreement between the code's own documentation and its behaviour. **Per
this task's instructions, this is reported and not resolved here**: whether
the intended design is the documented deterministic top-K or the implemented
weighted sampling is not this task's decision.

This is directly relevant to the Stage 3 decay: weighted-by-V_i sampling
(instead of strict top-K) is itself a rebalancing mechanism independent of
`G_i` dynamics — slow-tier clients are picked with a probability that never
goes to zero, at a rate set by their share of total `V_i`. Task 2 should
attribute the decay between this mechanism and the `G_i`-rises-with-neglect
hypothesis, not assume the latter is the only candidate.

### 4b. `G_i` carries a 10× "unexplored" proxy, computed from other clients, not from τ

`FedAsyncPolarisAggregationStrategy._update_gradient_bounds()`
(`fedasync_polaris_vi_server.py:239-309`) tracks `self.unexplored_clients`, the
set of client indices that have never yet reported. On every aggregation
event (`fedasync_polaris_vi_server.py:288-293`):

```python
if count > 0 and self.unexplored_clients:
    avg_norm = sum_norms / count
    proxy = self.alpha * avg_norm
    for idx in self.unexplored_clients:
        norm_deltas[idx] = proxy
```

`alpha` defaults to `10.0` (`fedasync_polaris_vi_server.py:168`, overridable via
`[server] polaris_alpha`). So an unexplored client's `G_i` is **10× the mean
observed gradient norm among clients that reported this round** — not a fixed
constant, and not related to staleness at all. Before any client has ever
reported, `G_i` starts at `initial_gradient_bound = 0.5` for everyone
(`fedasync_polaris_vi_server.py:169`, `:196-198`, and independently
re-declared as the selection strategy's own fallback at
`fedasync_polaris_vi_selection.py:104-107` — two separate `np.full(..., 0.5)`
defaults that happen to agree numerically but are set in two different places;
worth knowing about, not a bug).

This proxy is a second, independent rebalancing force on top of 4a: a client
that has gone unselected long enough to still be "unexplored" is boosted to
10× the observed mean `G_i`, which (combined with 4a's weighted sampling)
gives it a large — not just non-zero — chance of being picked regardless of
its `τ̂`.

### 4c. Nothing else found

No cooldown, no rotation schedule, no explicit tie-breaking rule, no
per-client eligibility window beyond the plain busy-client filter (a client
already training is removed from `clients_pool` before scoring —
`fedasync_polaris_vi_selection.py:170-182` and, under SUMO,
`sumo_gates.py:102-124` additionally removes clients out of RSU range). The
busy filter changes *who is eligible*, not how eligible clients are ranked
relative to each other, so it is not a fourth ranking mechanism. No fairness
constraint of the kind Task 1 asked to quote in full — none exists in this
code path.

## Summary for the questions this task exists to answer

`τ̂` is not one quantity computed one way, wired to `s(·)` uniformly across
arms — it is two structurally different pipelines (a hardware-spec-seeded
discrete-event simulation in seconds→rounds for `gpu`/`cpu`/`gpu_oracle`, vs. a
raw round-space memory read for `observed`) that only meet at the final
`v_i = g_i / sqrt(1+τ̂)` line. And selection is not driven by `s(τ̂)·G_i`
alone even conceptually: the actual selection step is a *weighted lottery*
over those scores (contradicting the module's stated design), and `G_i` for
never-yet-selected clients is itself a multiple of other clients' measured
norms rather than a property of the client being scored. Both 4a and 4b are
candidate (partial) explanations for the Stage 3 decay, independent of the
`G_i`-rises-with-neglect hypothesis Task 2 is set up to test — Task 2's design
should account for the weighted-sampling mechanism (4a) when interpreting its
counterfactual-ranking metric (Task 2.F), since a counterfactual "who would
S(τ̂) alone have picked" needs to be computed the same way selection actually
happens (weighted lottery), not as if it were top-K.

No behaviour was changed to produce this report.
