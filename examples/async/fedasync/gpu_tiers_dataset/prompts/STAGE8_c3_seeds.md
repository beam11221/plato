# Stage 8 — seeds for C3. 12 runs, no code changes.

Stage 7b produced the strongest result in the programme so far (§23b of
`consolidated-findings.md`), on a single seed:

| | open | closed | change |
|---|---|---|---|
| baseline | 7,238.0 s | 7,260.4 s | +0.3% |
| gpu | 6,768.6 s | 7,659.3 s | +13.2% |
| observed | 6,901.5 s | 7,874.3 s | +14.1% |

Polaris-style selectors beat random selection in the open population and lose to it in
the closed one, while random selection itself does not move. The mechanism is visible
in `G_i`: xavier : thor median at dispatch 1.4–1.6× open vs 2.9–3.1× closed.

This claim — **C3: closed-population evaluation flips the verdict on utility-based
selection** — is about to become a headline. This stage adds seeds 4127 and 4128 so it
can be quoted.

Use the Stage 7b code exactly as it stands. **No code changes.** Do not touch anything
under `plato/`.

---

## The one thing that must not vary: the physics seed

In 7b, `4126` plays two roles at `p_recur = 0`:

- **physics** — `vehicle_sleep_s(4126, veh_id, tier)` gives each trace vehicle its
  sleep; the trace itself is `e1_open_poisson_seed4126_range400`
- **training** — `identity_seed(4126, identity, round)` seeds torch, the lottery and
  dispatch-level randomness

**Only the training role changes in this stage.** The trace, every vehicle's
`sleep_s`, every vehicle's partition (`blake2b(veh_id) mod 20`) and the `p_recur = 1`
fleet (`clients_n20r400.csv`, `client_sleep_times_n20r400.csv`) must be byte-identical
to 7b in all 12 runs.

If the code derives both roles from one config value, split them — a `physics_seed`
pinned at 4126 and a `run_seed` that varies — and state where. If changing the seed
would silently redraw vehicle sleeps, the new seeds would be a different population, not
a replication, and every paired comparison in this stage would break without an error.
This is the same trap Stage 6 guarded against with the fleet MD5.

---

## Step 1 — pre-flight

Smoke-run `gpu_open` and `gpu_closed` for 100 rounds at `run_seed = 4127`. Required:

1. every Stage 7b pre-flight check at 100% (predictor tier == occupant tier, sleep ==
   occupant sleep, partition == occupant partition, `tau_hat` logged == used, state
   resets == recycles at `p_recur = 0`, one identity per slot at `p_recur = 1`)
2. **physics pinned:** for every `veh_id` dispatched, `sleep_s` equals its 7b (seed 4126)
   value exactly; for every slot at `p_recur = 1`, identity, tier, sleep and partition
   equal 7b's
3. **training seed actually changed:** `identity_seed` differs from 7b's for the same
   `(identity, round)`, and the first-round training loss differs from 7b's

Check 2 without check 3 means the seed was not applied. Check 3 without check 2 means
the population was redrawn. Both must hold. Report, then launch.

---

## Step 2 — the runs

| arm | `p_recur` | seeds |
|---|---|---|
| baseline | 0, 1 | 4127, 4128 |
| gpu | 0, 1 | 4127, 4128 |
| observed | 0, 1 | 4127, 4128 |

**12 runs.** Seed 4126 for all six cells is Stage 7b, reused as-is (same code, same
entry point — unlike Stage 6, there is no infrastructure asymmetry here).

`cpu` is excluded. It is worst in both regimes by margins (+6.7% open, +30.1% closed)
that seeds will not overturn, and it plays no part in C3.

Run **at most four at once**. Stage 6 hit CUDA OOM with more, and Stage 7b's checkpoint
cleaner settings (keep 50 rounds, 12 GB ceiling) stay on.

---

## Step 3 — what to report

Per run, the Stage 7b set: soundness checks, cumulative `sim_time_reconstructed` at
rounds 100 / 250 / 500 / 1,000 / 2,000 / 4,000, waste, dispatch share per tier per
window, median `G_i` at dispatch per tier for rounds 2,001–4,000 (dispatches with ≥ 1
observation), plateau, TTA at 70 / 75 / 78 / 80%.

**Occupancy invariant across all 18 runs** (12 new + 6 from 7b): the realised recycle
sequence must be identical. One digest.

### Across seeds

Per cell, over three seeds: mean and sample SD of T @ 4,000, waste, xavier share, and
the `G_i` xavier : thor ratio.

Then the claims that make up C3, each checked **in every seed individually**:

| # | comparison | claim |
|---|---|---|
| a | `gpu` open vs `gpu` closed | open is faster |
| b | `observed` open vs `observed` closed | open is faster |
| c | `baseline` open vs `baseline` closed | **no material difference** — report the gap against `baseline`'s own seed spread |
| d | `gpu`, `observed` vs `baseline`, **open**, at round 4,000 | Polaris arm ahead |
| e | `gpu`, `observed` vs `baseline`, **closed**, at round 4,000 | Polaris arm behind |
| f | `G_i` xavier : thor, closed vs open, both Polaris arms | closed ratio larger |
| g | round at which each closed Polaris arm crosses `baseline` | report per seed |

A comparison whose sign flips in any seed is reported as **undecided**, not as a mean
with a standard deviation wide enough to cover both signs.

Report (c) carefully: it is the control that makes C3 an identity effect rather than a
trace or fleet effect. The claim is not that `baseline` is exactly invariant but that
its open–closed gap is small relative to the Polaris arms' and within its own noise.

**Secondary, C2:** `gpu − observed` per seed in each regime. Stage 7b gave −1.93% open
and −2.73% closed. Report whether the sign is stable; no claim rests on it.

---

## Written down before the runs start

| # | prediction |
|---|---|
| 1 | Claims (a) and (b) hold in all three seeds, with open–closed gaps on the order of 10–15% |
| 2 | Claim (c): `baseline`'s open–closed gap stays within ~2% in every seed — an order of magnitude smaller than the Polaris arms' |
| 3 | Claims (d) and (e) hold in all three seeds: the sign of "Polaris vs random" flips between regimes |
| 4 | Claim (f) holds in every seed |
| 5 | The closed-regime crossing round varies by seed but stays in the ~1,500–2,500 range seen across Stages 3–7b |
| 6 | `gpu − observed` changes sign across seeds in at least one regime — C2 stays a null |

Prediction 3 is the one that matters most. (a) and (b) alone say open population helps
Polaris arms; (d) and (e) together say the **ranking against random selection
reverses**, which is what makes closed-population evaluation misleading rather than
merely pessimistic.

---

## Stop conditions

- Step 1 check 2 or check 3 fails
- the occupancy digest differs in any run
- any Stage 7b soundness check fails (conservation error > 3%, plateau outside
  82.1–83.7% by more than 2 points, predictor tier match < 100%)

---

## Do not

- change any code, including analysis code, except to split physics and training seeds
  if Step 1 shows they are coupled
- vary the trace, vehicle sleeps, partitions or the `p_recur = 1` fleet
- change sigma, tier ratio, `M`, startup `T_round`, the scoring function, the selection
  rule, the dataset, the model or the optimiser
- enable Stage 0
- run `cpu`
- re-run any Stage 7b seed-4126 cell
- report a mean across seeds without the per-seed sign check
- tune anything to make a prediction come true
