# `wen_env/`: the Wen-aligned closed-population environment

This folder rebuilds the environment model for Prompt B (`gpu_tiers_dataset/prompts/prompt-B-environment-model-runpod.md`): training time, upload time, energy, coverage, the clock, and the estimators the selectors use. Wen et al.'s own selector is a later prompt.

The decisions behind it are in `../docs/decisions/ADR-0005` … `ADR-0012`.

## Files

| file | role | who may read it |
|---|---|---|
| `wen_constants.py` | every locked value, in one place | everyone |
| `wen_formulas.py` | the physics as pure functions (training, Eq. 4, Eq. 5–6, energy) | everyone |
| `wen_trace.py` | bundle integrity, `x(veh, t)`, trace-end guard | ground truth |
| `wen_fleet.py` | tier per client, tier specs, persistent jitter | ground truth |
| `wen_truth.py` | the outcome of one dispatch | ground truth |
| `wen_reported.py` | `ReportedVehicle`: x, V_k, tier spec, \|S_i\| | server-visible |
| `wen_estimators.py` | T̂_train + T̂_up and Ê for gpu / cpu / gpu_oracle / location_only | selector side |
| `wen_selection.py` | coverage gate and pluggable Stage 0 | selector side |
| `wen_partition.py`, `wen_client.py` | disjoint Dirichlet split, client sampler | clients |
| `wen_server.py` | the clock, the per-dispatch override, waste, the logs | server |
| `wen_config.py`, `wen_fedasync.py` | config validation and the entry point | — |
| `wen_make_configs.py` | renders configs; cells may differ only in arm keys | — |
| `wen_clock_replay.py` | rebuilds the clock from the logs and compares it with the online clock | analysis |
| `wen_smoke_report.py`, `run_smoke.sh` | Step 9 | analysis |
| `measure_flops.py` | Step 1 measurement | — |

Selector-side modules never import ground-truth modules; `tests/test_wen_truth_and_selection.py` checks this.

## Running

```bash
cd examples/async/fedasync
uv sync                       # the repo venv; fvcore only for measure_flops.py
python wen_env/wen_make_configs.py --smoke
PYTHONPATH=$(cd ../../.. && pwd) /workspace/plato/.venv/bin/python \
    wen_env/wen_fedasync.py -c wen_env/configs/smoke/S1_gpu_s4126_smoke.toml
python wen_env/wen_clock_replay.py runs/wen_env/S1_gpu_s4126_smoke
```

Tests: `/workspace/plato/.venv/bin/python -m pytest wen_env/tests -q` (about 3 minutes).

## Landmines

- **The clock starts at 0.0.** `install_wen` sets PLATO's `initial_wall_time` and `wall_time` to 0. A clock at the epoch (about 1.76e9 s) resolves only about 2e-7 s, so the online clock and the replayed clock could not agree to 1e-6 s there.
- **One float addition per dispatch.** `t_finish = t_dispatch + (T_train + T_up)` is written in exactly the form PLATO uses, so the two clocks are bit-identical. Do not "simplify" it to `t_dispatch + T_train + T_up`.
- **The host's `processing_time` is zeroed** in `process_client_info`. The total that was removed is logged in the summary.
- **The static `csv_path` is a placeholder.** PLATO's client needs a sleep table, but the server overwrites every report's duration.
- **The fading key is `(seed, client, round, k)`.** In S2 a lost update holds the round counter, so the same client can be redispatched under the same round number; `k` prevents a replayed draw.
- **The merge-order residual** is logged (`order_violation` in `merge_log.csv`), not fixed. PLATO can only merge reports that have arrived in real time.
- **Never interpolate across a wrap, and never loop the trace.** See `wen_trace.py`.
