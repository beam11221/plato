# Prompt B, Step 0: read-only audit (16 Oct 2026)

This is the state of the code BEFORE the Wen-aligned environment model was added.
All paths are relative to `examples/async/fedasync/` unless they start with `plato/`.

## 1. Where sleep and training times come from, and where the anchor enters

**Where the times are made.** One formula is applied once, offline, to write a static table:

```
raw(tier)  = W_MODEL_FLOPS * partition_size * epochs / (gpu_tflops_fp16 * 1e12 * eta_nominal)
T_tier     = raw(tier) * anchor_scale        anchor_scale = anchor_sleep / raw(anchor_tier)
sleep_i    = T_tier(tier_i) * exp(eps_i) + C          eps_i ~ N(0, sigma), drawn once per client
```

- Current writer: `gpu_tiers_dataset/build_fleet.py`, which produces `fleets/<name>/sleep_times.csv`.
- Superseded writer: `gpu_tiers/generate_sleep_times.py`. It is still used by `separability*/`, `surgical/` and `sumo_close_loop/`.

**Where the anchor enters.**

Simulator side (what clients sleep):
- `build_fleet.anchor_scale()`.
- `gpu_tiers/generate_sleep_times.py --anchor-tier/--anchor-sleep`.

Predictor side (what selectors believe):
- `gpu_tiers/gpu_tier_sleep.predictor_scale()` reads `[gpu_tier] anchor_tier/anchor_sleep`.
- That result is applied in `_predicted_seconds()`. Every caller goes through it: `gpu_tops_predict_seconds`, `compute_pool_T_round_seconds`, `compute_pool_mean_T_seconds`.
- It also reaches the selector through those functions, via `fedasync_polaris_vi_server.Server.__init__` and `fedasync_polaris_vi_selection._busy_seconds`.

Configs and checks:
- Every config generator writes `anchor_tier = "xavier"` / `anchor_sleep = 15`. That covers `gpu_tiers_dataset/make_configs.py`, `sumo_close_loop/make_*_configs.py`, `sumo_open_loop/make_stage7_configs.py`, `separability*/make_configs.py`, `gpu_tiers/make_configs.py`, `gpu_tiers/make_comm_sweep.py` and `surgical/make_templates.py`.
- `gpu_tiers_dataset/preflight.py` checks `T_tier[anchor] == anchor_sleep`.

Constants derived from the anchored table:
- `sumo_close_loop/stage0_filter.FLEET_MEAN_T_HAT_S = 8.323077`.
- `sumo_open_loop/open_stage0` reuses the same constant.

Analysis and reporting:
- `gpu_tiers/summarize_sweep.py`, `gpu_tiers/comm_sweep.py`, `separability*/collect.py`, `separability_n100/health_report.py`, `sumo_close_loop/analyze*.py`.

## 2. PLATO consumes a static, per-client table

- `plato/config.py:452` `Config.simulate_client_speed()` reads `[clients.simulation_distribution] csv_path` once and caches it in `Config.client_sleep_times`, a numpy array indexed by `client_id - 1`.
- Under `sleep_simulation = true`, `plato/clients/strategies/defaults.py:347-372` replaces the client's measured training time with `(avg_training_time + sleep[client_id-1]) * epochs`. The client puts that into `report.training_time`.
- `plato/servers/base.py:1026` sets `finish_time = training_time + processing_time + comm_time + start_time`, and the simulated clock is that finish time.

**The value is per CLIENT, not per DISPATCH.** One row per `client_id`, the same for the whole run.

**Hook to override:** the server's `process_client_info(client_id, sid)` (`plato/servers/base.py:999`). It is overridden in the example folder, and rewrites `self.reports[sid].training_time` to this dispatch's `T_train_true + T_up_true` before calling `super()`. This is the hook `sumo_open_loop/sumo_open_server.py:568` already uses to rewrite the sleep per occupant identity.

`processing_time` is a host `perf_counter` delta (`defaults.py:247-251`, about 0.065 s per report per `gpu_tiers_dataset/stage2_evidence`). The same hook zeroes it, as `gpu_tiers_dataset/runtime_patches.py` does, so that only simulated quantities reach the clock.

## 3. Where T_round and tau-hat are computed, and what s(.) receives

- **Startup T_round:** `fedasync_polaris_vi_server.Server.__init__` calls `gpu_tier_sleep.compute_pool_T_round_seconds` over the whole fleet with `t_round_model = "queue"`, giving `T_round = mean(T_hat) / C`. It is stored in `context.state["T_round_seconds"]`.
- **Live T_round:** `fedasync_polaris_vi_selection.PolarisViClientSelectionStrategy.select_clients` recomputes T_round on every selection from the BUSY set's predicted full times (`compute_t_round_from_times`, same queue kernel), and computes busy residuals `max(0, T_hat_busy - elapsed)`.
- **tau-hat:** `gpu_tier_sleep.gpu_tops_predict_staleness_rounds(T_hat_i, busy_residuals, T_round)`. This is a two-phase count:
  - busy residuals shorter than `T_hat_i` count one round each;
  - then `remaining * (n_early / n_total) / T_round`, truncated with `int()`.
  - With no busy clients it falls back to `T_hat_i / T_round`.
- **s(.):** it receives tau-hat in ROUNDS. `V_i = G_i / sqrt(1 + max(tau_hat, 0))`, then the selector samples without replacement with probability proportional to V_i.
- For the `observed` arm, tau-hat is the client's last PLATO staleness (`current_round - starting_round`), stored in `context.state["polaris"]["local_stalenesses"]`.

## 4. Clock reconstruction

`gpu_tiers_dataset/clock_reconstruction.py` reads:
- the run's `config_used.toml`, to get C, `csv_path` and comm settings;
- the server log, for the `Starting round`, `Selecting client #` and `Adding client #` lines, in file order;
- the client logs' `[GpuTierSleep] ... sleep_time=` lines, to check the fleet;
- the static `sleep_times.csv`.

It replays C slots, filling each dispatch into `argmin(slots)` with `t_finish = t_start + sleep[client] + comm_time`.

**It assumes one fixed sleep per client.** That no longer holds, because upload time depends on position. It also assumes every freed slot is refilled at once, which does not hold under coverage gating.

## 5. Stage 0 and its speed term

- Closed loop: `sumo_close_loop/stage0_filter.py` (`Stage0FeasibilityGate`), installed by `sumo_server.SumoRsuMixin.install_sumo` between `RsuSelectionGate` and the arm's own strategy.
- Open loop: `sumo_open_loop/open_stage0.py`.
- The speed term is `[sumo] v_max_mps = 19.444444` (70 km/h, the E1 trace's lane limit) in every Stage 5, 6 and 10 config.
  - `vmax_elapsed`: `R_i = t_enter + 2*radius/v_max - t`.
  - `vmax_distance`: `R_i = |x_exit - x_i| / v_max`.
- Each candidate is tested against `R_i >= T_hat_train`. There is no upload term and no energy term.
- `T_hat` comes from `stage0_t_hat`: `fleet_mean` (anchored 8.323 s), `tier_spec` (via `gpu_tier_sleep`, anchored) or `oracle`.

## Consequences for this prompt

1. **Training time.** True training time is computed per client from the formula without an anchor, then multiplied by the fleet's persistent `exp(eps_i)`. It no longer comes from the static table's value.
2. **Per-dispatch duration.** Every dispatch's duration (`T_train_true + T_up_true`) is computed when the server dispatches. It is written into the report in `process_client_info`.
3. **Estimators.** The `wen_env` path does not go through `gpu_tier_sleep`'s predictor functions at all, so no anchor key can be read. The selector receives `T_hat_total` through `predict_staleness` / `busy_predict_seconds`. The only `[gpu_tier]` key the `wen_env` configs keep is `t_round_model = "queue"`, which the shared T_round kernel reads.
4. **Stage 0.** It is rewritten for one-way motion: `R_i = (1600 - x_i) / V_k`, where `V_k` is the vehicle's own reported speed. It is tested against `T_hat_train + T_hat_up` and against an energy cap.
