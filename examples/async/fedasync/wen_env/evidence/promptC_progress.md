# Prompt C: progress up to the pilot gate (16 Oct 2026)

> **Phase A is complete (16 Oct 2026, 14:24).** All 11 cells finished. Every one passes the clock check (online clock equals the replayed clock, maximum difference 0.0 s). The analysis is in `main_runs_seed4126.md`, with curves in `main_runs_seed4126_curves.png`. Phase B (seeds 4127 and 4128) has **not** been started and waits for an explicit go-ahead.
>
> **Phase B and location-only are complete (17 Oct 2026, 09:33).** All 36 runs pass the clock check. Results are in `main_runs_v1.md` and `main_runs_v1_curves.png`.
>
> **Update, 16 Oct 2026, about 15:00.**
> - The user approved Phase B. 22 runs are queued in `wen_env/logs/phaseB.out`, at most 5 in parallel.
> - The user also approved **location-only (P12) for all 3 seeds**, in S1 with the same config, following the Phase A analysis (`phaseA_analysis_seed4126.md` §4.2: the gpu advantage is mostly the position term). The configs are `wen_env/configs/runs/S1_location_only_s412{6,7,8}.toml`. They are queued behind Phase B by `wen_env/logs/queue_location_only.sh`, which waits until all 22 Phase B runs have started, so that two launchers never race for one slot.
> - The arm was checked before queueing: T̂_train is a constant 1.542 s (the fleet mean of the gpu estimates), so Xavier and Thor get the same T̂ at the same position.

> **Update, 16 Oct 2026, 04:35.** The blocking inputs below have been resolved.
>
> - Predictions v3 were supplied in `wen_env/evidence/predictions_v3.md`. They were copied verbatim to `docs/predictions_v3.md` and committed as **f2bb90a1** on branch `wen-env-main-runs`. Every run manifest records that hash.
> - **The user's instruction changed the gate from blocking to reporting.** Phase A starts without waiting for the pilot gate. The gate result will be written to `pilot_gate.md` when the two pilot runs finish, and the user will be told if it fails.
> - Pilot started 04:29 (S1 random, S1 gpu). The other 9 Phase A cells were queued at 04:35, at most 5 runs in parallel.

**Status: stopped before Step 4.** Two inputs the prompt depends on are not on this machine:

1. **Predictions v3.** The prompt points to `claude/roadmap-wen-environment.md`, which does not exist. The roadmap on disk (`gpu_tiers_dataset/prompts/roadmap-wen-environment.md`) holds predictions **v2** only.
   - Without v3 there is nothing to copy into `docs/predictions_v3.md` or to commit.
   - `wen_env/wen_manifest.py` refuses to start any run of 4,000 rounds or more until that file is committed.
   - The pilot gate (Step 4) and Phase A (Step 5) are therefore blocked.
2. **The SUMO ADR text** (`ADR-0001-wen-closed-environment.md` from the MacBook) is not on this machine, so ADR-0005 is still the placeholder.

Steps 1–3 and 4b are done.

## Step 1: D1, D2 and logging

**D1, ε-greedy.**
- Implemented in `wen_env/wen_egreedy.py` as one class, `WenSelector`, used by every arm. `random` is the same class with ε = 1 and no scorer.
- The coin and the uniform draw come from `default_rng([seed, round, k, salt])`.
- Exploit picks take the highest V_i; ties go to the lower `client_id`.
- The `proportional` ablation is delegated unchanged to the Polaris selector.
- To share the scoring code, `fedasync_polaris_vi_selection.py` gained a `score_candidates()` method. It was extracted without changing behaviour, and the `proportional` cell below reproduces the Prompt B log exactly.

**D2, per-family energy estimate.** `CpuEstimator.energy_hat` is Wen Eq. 7 + 8, with e = 1e-28. Its training part is 24.79 / 24.79 / 34.62 J.

| Tier | GPU-family Ê, worst case (d = 1,000 m) | CPU-family Ê, worst case |
|---|---|---|
| Xavier | ≈ 114 J | ≈ 27 J |
| Orin | ≈ 29 J | ≈ 27 J |
| Thor | ≈ 26 J | ≈ 37 J |

No arm is blocked at E_max = 600 J. Ground-truth energy is unchanged.

**New dispatch-log columns:** `selection_rule`, `epsilon`, `coin_u`, `explored`, `n_candidates`, `rank_of_pick`, `V_max`, `V_min`, `V_pick`, `E_hat`.

**Config keys:** `[wen_env] selection` and `epsilon`, validated in `wen_config.py`. Any ε other than 0.2, and `proportional`, are allowed only for the S1 gpu ablation.

**ADRs:** 0013 (ε-greedy), 0014 (per-family energy), 0015 (disjoint Dirichlet partition with Sinkhorn balancing).

## Step 2: tests

All six tests in the prompt pass.
1. ε = 1 (gpu arm) reproduces the random arm's dispatch log exactly (30-round runs `*_eps1chk`).
2. ε = 0 always picks rank 1, and ties go to the lower id.
3. Over 10,000 decisions with 27 candidates and ε = 0.2, rank 1 is picked with probability 0.8 + 0.2/27 (±1%), and every candidate is picked at least 0.2/27 − 0.3% of the time.
4. `proportional` reproduces the Prompt B gpu smoke log (seed 4126) over all 202 dispatches.
5. The cpu Ê matches D2, and no tier of either family is blocked at 600 J at any position.
6. Ground-truth energy for a cpu-arm dispatch equals the GPU formula, checked in the unit test and on every row of the cpu re-smoke run.

The `wen_env/tests` suite also covers the manifest refusal and partition rounding over 20 seeds.

## Step 3: re-smoke (200 rounds, seed 4126; accuracy not read)

| Cell | Rule | s / round | Share X / O / T | Explored | Mean rank (exploit) | V_max − V_min: q1 / median / q3 / max | Stage 0 rejected | Clock diff |
|---|---|---|---|---|---|---|---|---|
| S1 random | ε = 1 | 3.823 | 0.44 / 0.39 / 0.18 | 1.000 | — | — | — | 0 |
| S1 gpu | ε = 0.2 | 3.014 | 0.22 / 0.44 / 0.34 | 0.228 | 1.00 | 0.202 / 0.231 / 0.281 / 0.732 | — | 0 |
| S1 gpu | ε = 0 | 2.776 | 0.22 / 0.50 / 0.28 | 0.000 | 1.00 | 0.194 / 0.226 / 0.270 / 0.627 | — | 0 |
| S1 gpu | proportional | 3.721 | 0.51 / 0.31 / 0.18 | — | — | — | — | 0 |
| S1 cpu | ε = 0.2 | 3.716 | 0.38 / 0.28 / 0.34 | 0.228 | 1.00 | 0.107 / 0.125 / 0.165 / 0.550 | — | 0 |
| S2 gpu + Stage 0 | ε = 0.2 | 2.997 | 0.29 / 0.44 / 0.27 | 0.225 | 1.00 | 0.138 / 0.176 / 0.212 / 0.828 | 24.7% (0 fallbacks) | 0 |
| S2 cpu + Stage 0 | ε = 0.2 | 3.602 | 0.43 / 0.29 / 0.28 | 0.227 | 1.00 | 0.039 / 0.092 / 0.139 / 1.037 | 74.4% (2 fallback cycles of 201) | 0 |

The explored share is 0.225–0.228 against ε = 0.2. With about 200 picks per cell the binomial SD is about 0.028, so this is within 1 SD.

## Step 4b: checkpoint cleaner

**File.** `wen_env/tools/ckpt_cleaner.py`. It is new: `gpu_tiers/clean_checkpoints.sh` works on one run, keeps a number of rounds, and has none of Prompt C's rules.

**What it may delete.** Files directly inside `runs/wen_env/<run>/checkpoints/` that end in `.safetensors`, `.pth`, `.pt` or `.pkl`. The families actually present are:
- `checkpoint_resnet_18_#.safetensors` and its `.pkl` sidecar;
- `prng_state_#.pkl`, `numpy_prng_state_#.pkl`;
- `current_round.pkl`;
- `resnet_18_#.pkl`, `resnet_18_client_#.pkl` (per-client model payloads).

**What it keeps.**
- For a running run: the newest file of each family, plus anything younger than 10 minutes.
- A finished run loses all its checkpoints.

**Running process.** PID **49045** under `nohup`, looping every 5 minutes. Output goes to `wen_env/logs/ckpt_cleaner.out`, and each deletion is appended to `wen_env/logs/ckpt_cleaner.log`.

**Dry run.** It was run on the live re-smoke outputs before deletion was enabled, and the listing is in `ckpt_cleaner_dryrun.txt`.
- At the real 10-minute age it listed 0 files, which is correct: every run was live and every file was newer.
- At a 60 s age it listed 346 files. All were in `checkpoints/`, and the newest file of each family was kept.

**Freed so far:** 68.2 GiB across 5,097 files. Every path is a `checkpoints/` file, which was checked against the log.

**Disk guard.** It lives in `wen_env/run_cells.sh`. Because `df` reports the shared MooseFS, free space is computed as 200 GB − `du -sb /workspace`, capped at df's own figure. Below 20 GB the launcher waits; after 15 minutes it stops with exit code 3.

## Deviations and incidents

A. **Runs stop at 90% accuracy, not at 4,000 rounds. Kept at the user's decision on 16 Oct.**
   - Every config carries `[trainer] target_accuracy = 0.9`. This was copied from earlier stages, where it was "unreachable in practice". PLATO ends a run when that accuracy is reached ("Target accuracy reached. Training concluded."). In this environment it is reachable: S1 random stopped at round 1,770 (simulated t = 6,645 s), and S1 gpu and S1 cpu were at 89.75% and 89.01% at round 1,950.
   - The user chose to keep the early stop rather than rerun with an unreachable target.
   - **Consequences:**
     - Cells end at different round counts.
     - The Step 4 plateau gate is evaluated on the last 500 rounds each run actually has, not rounds 3,500–4,000.
     - The Step 6 dispatch-share window 3,500–4,000 is empty.
     - "Plateau" (mean of the last 200 rounds) is taken just below 90% for runs that stopped early.
     - TTA@80% is unaffected.

B. **16 Oct, about 07:14–07:46: the disk quota was exceeded because the cleaner had died.**
   - When `S1_gpu_oracle_s4126/checkpoints` was removed during its archiving, a `FileNotFoundError` killed the restarted cleaner (PID 103152).
   - With no cleaning, each run built up about 24 GB of per-round checkpoints. The 200 GB quota filled at about 07:40, and writes failed with `OSError: [Errno 122] Disk quota exceeded`.
   - Space was freed by hand at 07:46 under the cleaner's own rule: per-round checkpoint and PRNG files older than 10 minutes, only inside `checkpoints/`. That removed 6,832 files, about 71 GB.
   - A failed write had truncated `ckpt_cleaner.py` to 0 bytes. It was rewritten in full: vanished directories or files are now skipped, and a failed pass is retried instead of killing the process. It restarted as PID **108909**, and `tests/test_ckpt_cleaner.py` passes.
   - **Damage:**
     - The only quota errors are 4 logging-flush errors in `S1_gpu_eps0`'s client log.
     - Every run's dispatch, merge and cycle logs are intact: no NUL bytes, and the clock replay matches exactly for S1 random, gpu, cpu and gpu ε = 0.
     - The server logs of the live runs have NUL padding from the failed writes. The analysis does not read them, apart from the Stage 0 summary line in S2 runs, which had not started yet.
   - **Correction (found 08:46): all five live runs had hung at 07:40:53, the moment the quota filled.** The processes stayed alive, but every server stopped mid-checkpoint-save or while waiting on a client whose write had failed, and no round advanced afterwards:

     | Run | Round when it hung |
     |---|---|
     | S1 gpu | 1,957 |
     | S1 cpu | 1,929 |
     | S1 gpu ε = 0 | 1,876 |
     | S1 gpu_oracle | 194 |
     | S1 proportional | 191 |

     The watcher only matched error strings, so the hang went unseen for about an hour.
   - **Recovery.**
     - All five were stopped, and their directories (logs kept, checkpoints removed) archived as `runs/wen_env/_failed/<cell>_hung_diskfull_0740/`.
     - They were relaunched from round 0 with the same configs and seed, together with the 5 cells that had not started yet: one launcher (`wen_env/logs/phaseA2.out`), at most 5 runs at a time.
     - S1 random had finished at 07:20, before the quota filled. Its run is intact and is kept.
   - **The watcher now also reports a STALL** when any run's round has not advanced for 15 minutes.

C. **S1 gpu and S1 cpu are taken from the runs that hung. User decision, 16 Oct.**
   - Their reruns were stopped at round 133 (gpu) and round 122 (cpu), and archived as `_failed/*_rerun_stopped_by_user_*`. That freed two slots for the queued cells.
   - The results for these two cells come from `_failed/S1_gpu_s4126_hung_diskfull_0740` and `_failed/S1_cpu_s4126_hung_diskfull_0740`:
     - logs are intact up to the hang, and the replayed clock matches the online clock exactly;
     - gpu reached 89.75% at round 1,950 and cpu 89.68% at round 1,840. At the user's request, both count as having reached 90%.
   - **Consequences:**
     - These two cells end at about round 1,950.
     - Their server logs carry NUL padding from the disk-full write failures; their dispatch logs and accuracy CSVs are clean.
     - The pilot gate (Step 4) is evaluated on these runs and on S1 random.

0. **16 Oct, 07:06: the cleaner hung `S1_gpu_oracle_s4126` at round 1,501. The run was restarted from round 0.**
   - **What happened.** At 07:06:35 the server rewrote `checkpoints/resnet_18_19.pkl` to dispatch client 19. In the same second the cleaner deleted that file. The cleaner had stat-ed the file earlier, when its previous copy was over 10 minutes old, and unlinked it after the rewrite: a check-then-delete race. Client 19 could not load its model, the server waited for a report that would never come, and the run hung.
   - **Fix.** While a run is alive, `wen_env/tools/ckpt_cleaner.py` now deletes only the per-round families: the model checkpoint and its `.pkl` sidecar, and the two PRNG states. Per-client payloads (`resnet_18_<cid>.pkl`, `resnet_18_client_<cid>.pkl`) and `current_round.pkl` are never touched until the run finishes. These are bounded at about 2.6 GB per run. The file's age is also re-checked immediately before unlinking. Both behaviours are pinned by `tests/test_ckpt_cleaner.py`. The cleaner restarted as PID **103152**.
   - **The run.** The hung run is archived with its logs (checkpoints removed) at `runs/wen_env/_failed/S1_gpu_oracle_s4126_hung_r1501/`, and the cell was re-queued from round 0 with the same config and seed.
   - **Other runs.** No other run was affected: no `FileNotFoundError` appears in any other log.

1. **The container memory limit is 50 GB** (cgroup). That limit is not what `free` reports.
   - Nine parallel runs were OOM-killed: `S1_random_smoke2` and `S2_cpu_stage0_smoke2`, `oom_kill 2`. One run holds about 7 GB RSS.
   - Both runs were rerun from scratch, and `run_cells.sh` now caps live runs at `MAX_PARALLEL = 5`.
   - **Consequence for Phase A:** the 9 new runs cannot all execute in parallel. They will run 5 at a time.
2. **`run_cells.sh` was edited while one instance was running.** Bash reads scripts incrementally, so that instance stopped with a syntax error. Its runs had already been launched and finished normally.
3. **ADR-0015 says the rounding is tested over 20 seeds.** At first only one seed was tested; the 20-seed test was added so the ADR's statement is true.
4. **Within one selection call, the ε-greedy picks share one V snapshot.** This happens when several slots open at once, for example the first cycle. It is documented in ADR-0013.
