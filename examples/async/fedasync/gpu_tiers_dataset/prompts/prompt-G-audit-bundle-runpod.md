# Prompt G — Code audit bundle (RunPod)

**Purpose:** package the implementation for an independent code audit. An external reviewer (the planning chat) will read the code and check every equation, with Wen-faithful as the top priority. **You are not the auditor.** Your job is to collect files completely and index them accurately — do not explain, justify, or defend the code.

**Rules**
- No new training runs. Do not edit any source file, config, log, or result. Write new files only, under `audit/`.
- Copy files **whole and verbatim**. No excerpts, no reformatting, no "simplified" versions. If a file is too large (> 5 MB), include it whole anyway unless it is a log; for logs see Step 4.
- If you cannot find something listed below, write `NOT FOUND — searched: <paths/greps>` in the README. Do not substitute a similar file silently.
- Where the README says "file:line", give the real line numbers from the copied file, checked after copying.

## Step 0 — Provenance

Record in `audit/PROVENANCE.md`:
- `git rev-parse HEAD`, `git status --porcelain` (uncommitted changes must be listed; also include `git diff` output as `audit/uncommitted.diff` if non-empty)
- PLATO upstream commit the repo is based on, and `git diff <upstream>..HEAD -- plato/` saved as `audit/plato_core.diff` (should be empty under our rules; if not, say so first in the README)
- Python / torch / PLATO versions, OS, GPU
- The commit hashes of predictions v3, v3.1, v3.2 and the run manifests' config hashes for every run in `evidence/main_runs_v1.md`, `cpu_cal_v1.md`, `wen_faithful_v1.md`

## Step 1 — Source files (full copies, original relative paths preserved under `audit/src/`)

Include every file on the call path of a run, not only the ones you think matter. At minimum:

**A. Wen-faithful arm (highest priority)**
1. Wen selector: candidate construction, eligibility, F1–F4 computation, constraint check (Eq. 10), fallback, exact enumeration (single pick and any start-up multi-pick path), Pareto filtering, min–max normalisation, final pick rule, tie-break
2. Wen's position prediction (Loc_x ± T̂_com·V_k, direction handling, road wrap at 2,000 m, Loc_y)
3. Wen's CPU time model (Eq. 4: b·|S|·ρ/f) and CPU energy model (Eq. 7, Eq. 8 upload energy)
4. F3 source: where the last-reported loss is stored, when it is updated, the never-trained prior (fleet mean), and what happens before any report
5. F4: label histogram per vehicle, reference distribution Q, KL direction and epsilon/smoothing for zero counts
6. T_max / E_max q = 90 computation script and its output
7. `wen_decisions.csv` writer and per-candidate side-file writer

**B. Shared environment (ground truth, every arm)**
8. True training time: FLOPs measurement, T = 3·FLOPs_fwd·|S|·E/(η·TFLOPS), per-vehicle lognormal jitter σ = 0.149, tier table
9. True upload: Wen Eq. 5–6 (distance at training finish, path loss φ = 3, B, p, noise, Rayleigh |χ|² draw and its RNG keying, Shannon rate), model size D
10. True energy: P_tier × T_train + p × T_up
11. Coverage predicate and `coverage_gated` / `always_accept` handling, waste accounting
12. SUMO trace loader, vehicle ID → tier → speed mapping, fleet file
13. Online clock and the post-hoc clock replay script

**C. Our method and other arms**
14. GPU estimator T̂_train, shared T̂_up estimator (fading expectation = 1), total T̂
15. Score V_i = G_i/√(1+τ̂_i): how τ̂ is derived from T̂ (this was an open question at Stage 3), G_i computation and prior α = 1
16. ε-greedy selection: RNG keying (seed, round, k), idle/eligible set, tie-break
17. Stage 0 filter: residence-time test with reported V_k, energy check, per-arm energy model family
18. cpu, cpu_cal, location_only, observed, gpu_oracle, random arms — wherever they differ
19. Disjoint Dirichlet sampler with Sinkhorn equalisation

**D. Aggregation and metrics**
20. FedAsync mixing override (`afl_algorithm.py` or equivalent), staleness function and its exponent
21. Metric scripts: TTA@80/85/90, s/round (whole run, before/after threshold), energy to 80%, dispatch share by tier and window, Gini, waste fractions

**E. Configs and tests**
22. Every config file used by the runs in Step 0's evidence files
23. All unit tests, plus their current output: run the test suite once and save the output as `audit/test_output.txt`

**F. Documentation**
24. All ADRs, `CLAUDE.md`, `STATE.md`, predictions v3 / v3.1 / v3.2, `evidence/*.md` reports

## Step 2 — Equation map (`audit/README.md`, main section)

A table, one row per equation or rule. Columns: `ID | What | Paper reference | file:line (start–end) | function name | constants used (name = value, defined at file:line)`.

Rows required (add more if the code has them):
- W-Eq4 CPU training time · W-Eq5 distance at upload · W-Eq6 upload rate/time · W-Eq7 CPU energy · W-Eq8 upload energy · W-F1 · W-F2 · W-F3 · W-F4 · W-Eq10 constraints · W-Pareto · W-Norm · W-Pick · W-Fallback · W-Limits (q = 90)
- GT-Train · GT-Jitter · GT-Upload · GT-Fading · GT-Energy · GT-Coverage · GT-Waste
- E-GPU · E-Up · E-Total · E-Tau (T̂ → τ̂) · S-Vi · S-Gi · S-Prior · S-Eps · Stage0-Time · Stage0-Energy · CPU-arm energy
- Agg-Mix · Agg-Staleness · Clock-Online · Clock-Replay · Sampler · each metric

Rules for this table: copy the formula **as it appears in the code** (a one-line transcription), not as it appears in the paper. If code and paper differ, do not comment — the auditor will compare.

## Step 3 — Call-path trace

For one dispatch decision of the `wen` arm and one of the `gpu` arm (S2, seed 4126, the first decision after round 100): list the functions called in order from "slot becomes free" to "vehicle dispatched", then from "update arrives" to "model mixed and clock advanced", each as `file:line function`. Plain list, no prose.

## Step 4 — Data samples for hand recomputation (`audit/samples/`)

- `wen_decisions.csv` and per-candidate side files: full files for S2 seed 4126; first 200 decisions only for the other 5 wen runs
- For the same S2 seed 4126 run: the 10 decisions at rounds 100–109 with, per candidate, every input the selector used (position, V_k, tier, f, ρ, |S|, b, D, last loss, label histogram, idle/eligible flags) and every output (F1–F4 raw, normalised, constraint flags, Pareto membership, pick). If any input is not logged, say which and do not reconstruct it
- Ground-truth record for the same dispatches: true T_train, true distance at upload, drawn |χ|², true T_up, true energy, upload accepted/rejected
- Dispatch and accuracy logs (first 500 rounds) for S2 seed 4126 `gpu` and `random`
- Fleet file, tier table, SUMO bundle sha256 and the first 2,000 trace lines
- Do **not** include checkpoints, datasets, or full multi-GB logs

## Step 5 — Self-reported deviations (separate section, clearly labelled as claims)

A list of every place where you know the implementation departs from Wen et al. or from a prompt (A–F), with file:line. Mark it "claim, not verified". Keep it factual; no justification.

## Step 6 — Package

- `audit/README.md` order: (1) anything alarming first (non-empty `plato_core.diff`, uncommitted changes, failed tests, NOT FOUND items); (2) how the bundle is laid out; (3) equation map; (4) call-path trace; (5) self-reported deviations
- Zip as `audit_bundle_<git-short-hash>_<YYYYMMDD>.zip`, include `audit/MANIFEST.txt` with every file's path, size and sha256
- Target < 100 MB. If over, drop sample logs first (keep the S2 seed 4126 decision files), never source or configs; record what was dropped
- Report: zip path and size, sha256 of the zip, test pass/fail counts, and the list of NOT FOUND items

## Stop conditions

- Test suite cannot run → still build the bundle, record the error in the README's first section.
- `plato_core.diff` non-empty → still build the bundle, put it first in the README.
