# Prompt L — Scenario 2: Polaris arms (seed 4126) and the full S2 comparison (RunPod)

Continues from Prompt I (Study 2, S2 seed 4126), Prompt J (Polaris, S1) and Prompt K (S1 close, may still be running). Same rules: do not edit `plato/` core; ground-truth / reported / estimated separation; locked values only; stop and report on any conflict. **Do not disturb Prompt K runs** (shared disk guard, cleaner, watchdog).

## Decisions locked on 23 Oct 2026

| # | Decision |
|---|---|
| S2-A | Scenario 2 = S1 with coverage [400, 1600] m (60 %, RSU radius 600 m) — **unchanged from every earlier S2 run**. `coverage_gated`. |
| S2-B | Arms. Main: random, Ours (gpu + Stage 0), Polaris, Wen et al. (`wen_fresh`). Plug-in: random + S0, wen_fresh + S0, wen_fresh + GPU, wen_fresh + GPU + S0, Polaris + S0. Ablation: gpu (Ours without Stage 0). |
| S2-C | Pipeline for every arm: eligibility (idle, inside coverage at dispatch) → Stage 0 (if on; **our** estimator, as Prompt I S2-4) → host scoring / constraints → selection. |
| S2-D | Energy to X % counts **every** dispatch finished by that time, **failed (lost-upload) dispatches included**. |
| S2-F | Headline participation metric = **successful training ratio (STR)** = updates delivered and merged / vehicles dispatched (Tu et al. 2024, |N^r| / |S^r|); it replaces "waste" in all new tables and plots. STR = 1 − waste fraction (dispatches). Earlier predictions phrased as "waste lower" are evaluated as "STR higher". Keep the failed GPU-s and kJ as secondary columns. Rename the column in the report tools, not in raw logs. |
| S2-E | Seed 4126 first for every arm; seeds 4127 / 4128 only on explicit go-ahead. |
| Polaris | Exactly the S1 host of Prompt J (PL1–PL8, paper formulation, A = 1, α = 10, initial G 0.5, initial τ 0.01, τ + 0.1, fleet-static p_i). In S2 the pool is the eligible set (inside coverage, idle). G_i and τ_i update only on **merged** updates; a lost upload changes neither. |

## Step 0 — Checks

1. **Energy definition.** Show (file:line) how "energy to X %" is computed in the report tools. If wasted dispatches are excluded anywhere, fix the tool (not the runs) and recompute energy for every existing S2 run; list old vs new values.
2. **Existing S2 runs at seed 4126**: list them with source hash and confirm they were run at coverage [400, 1600]: random, gpu, gpu + Stage 0, wen, wen_fresh, random_s0, wen_fresh_s0, wen_fresh_gpu, wen_fresh_gpu_s0. Missing or different coverage → report.
3. Prompt K status (running / done), free disk, cleaner and watchdog (30-min no-merge alert) active.

## Step 1 — Lock the predictions

Copy `predictions_v3_6_addendum.md` to `docs/`, commit, record the hash; extend the manifest guard to `polaris` and `polaris_s0` in S2.

## Step 2 — Implement (configuration only, expected)

1. `polaris` in S2: the S1 host with the S2 eligibility gate. Stage 0 off.
2. `polaris_s0`: the existing Stage 0 plug-in (Prompt I) in front of the Polaris host.
3. No change to Polaris itself. If anything beyond configuration is needed, list it before writing code.

## Step 3 — Tests

1. `polaris_s0` with Stage 0 switched off reproduces `polaris` S2 dispatch log for 100 rounds (seed 4126).
2. Stage 0 never receives a non-eligible vehicle; Polaris never receives a Stage 0-rejected vehicle.
3. A lost upload does not change G_i or τ_i (unit test).
4. Σ q = 1 over the post-filter pool on every logged call.
5. Determinism: two 100-round `polaris_s0` runs, identical logs.

## Step 4 — Smoke (200 rounds, seed 4126): `polaris` S2, `polaris_s0` S2

s/round, tier share, waste, Stage 0 rejection rate (reason, tier), effective pool size 1 / Σ q², distinct vehicles. Do not read accuracy.

## Step 5 — Runs (seed 4126)

| Arm | Scenario | Seed |
|---|---|---|
| `polaris` | S2 | 4126 |
| `polaris_s0` | S2 | 4126 |

4,000 rounds max, stop at 90 %, manifests with predictions v3.6 hash, source hash, seed.

## Step 6 — Report `evidence/s2_s4126.md` (mechanical only)

**6.1 S2 table, seed 4126, every arm in S2-B** (existing runs + the two new ones). Columns:
- TTA@80 / 85 / 90 %; rounds to 80 / 85 / 90 %; max accuracy;
- energy to 80 / 85 / 90 % (waste included), and of that the wasted part in kJ;
- **successful training ratio (STR)** = merged / dispatched (headline); failed dispatches' GPU-s and kJ (secondary);
- **uplink airtime to 80 / 85 / 90 %**: Σ T_up actually spent on the channel (including partial airtime of lost uploads, if any — state how the environment models it);
- mean staleness of merged updates;
- s/round; tier share (rounds 1–500, last 500); Gini; distinct vehicles; image coverage at r500 / r1000 / end;
- Stage 0 rejection rate (by reason and tier) for S0 arms; mean |x_upload − 1000|.

**6.2 Pairwise directions, seed 4126:** Ours vs each other arm at 80 / 85 / 90 % on TTA and energy, and on STR (sign only; single seed).

**6.3 Plug-in pairs:** host vs host + S0 for random, wen_fresh, wen_fresh + GPU, Polaris; and wen_fresh vs wen_fresh + GPU — differences on every 6.1 column.

**6.4 P-table:** P25–P28, P33, P38–P43 at seed 4126 — direction, status "single seed — not evaluated".

**6.5 Curves:** accuracy vs online clock and accuracy vs cumulative energy — main arms in one panel, plug-in pairs in another.

## Stop conditions

- Step 0.2 finds an S2 run at a different coverage → stop.
- Step 1 commit missing → no new run.
- Any Step 3 test fails → stop.
- Online ≠ replayed clock → stop.
- Free disk < 20 GB for > 15 minutes → stop launching and report.
- Seed 4126 done → write the report and stop.

## Deliverable message

Step 0 results (energy definition and any recomputed values), test results, smoke table, the S2 table, pairwise and plug-in tables, the P-table, and every deviation.
