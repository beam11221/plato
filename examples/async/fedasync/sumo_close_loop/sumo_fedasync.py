"""
Entry point for the SUMO closed-loop FedAsync experiment.

One entry point serves all four arms; which one runs is decided by the TOML:

    [server] staleness_predictor absent    -> baseline (random selection)
    [server] staleness_predictor = "gpu_tops" | "cpu" | "observed"
                                          -> the matching Polaris V_i arm

That mirrors how the surgical experiment separates its arms, so a config can be
compared line-by-line against its non-SUMO counterpart.

Required config section:

    [sumo]
    coverage_csv = "/abs/path/to/input/coverage_n20.csv"
    # Optional, defaults to "coverage_gated" (this file's original, only
    # behaviour). "always_accept" is Stage 4 Task 4's counterfactual arm --
    # see sumo_server.resolve_upload_policy.
    upload_policy = "coverage_gated"
    # Optional, defaults to "none" (the gate disabled). One of none /
    # fleet_mean / tier_spec / oracle -- see stage0_filter.py. coverage_radius_m
    # and v_max_mps are required together whenever this is not "none"; both
    # come straight from the trace's own config.yaml.
    stage0_t_hat = "tier_spec"
    coverage_radius_m = 400.0
    v_max_mps = 19.444444
    # Optional, defaults to "vmax_elapsed" (Stage 5's original formula, kept
    # byte-identical). "vmax_distance" (Stage 5b's fix, prompts/STAGE5B_fix_
    # Ri.md) additionally needs positions_csv and rsu_x_m; "oracle_exit"
    # needs neither. See stage0_filter.py.
    stage0_r_i = "vmax_distance"
    positions_csv = "/abs/path/to/input/positions_n20r400.csv"
    rsu_x_m = 600.0

Run:
    cd examples/async/fedasync
    PYTHONPATH=gpu_tiers:.:sumo_close_loop \
        uv run sumo_close_loop/sumo_fedasync.py \
        -c sumo_close_loop/configs/gpu_n20.toml

Grep the server log for:
    [SumoRsu] report ...   one line per client report, with the upload verdict
    [SumoRsu] waste ...    cumulative waste per tier
    [PolarisVi] ...        V_i scores and staleness-prediction accuracy
"""

from __future__ import annotations

import logging

import fedasync_algorithm
import sumo_server
from fedasync_gpu_tier import GpuTierLogCallback
from stage0_filter import resolve_stage0_config
from sumo_mobility import CoverageOracle
from sumo_position import PositionOracle

from plato.clients import simple
from plato.config import Config


def main() -> None:
    sumo_config = getattr(Config().config, "sumo", None)
    if sumo_config is None or not hasattr(sumo_config, "coverage_csv"):
        raise ValueError(
            "This experiment needs a [sumo] section with coverage_csv pointing "
            "at the table written by remap_fleet.py."
        )

    oracle = CoverageOracle.from_csv(sumo_config.coverage_csv)
    upload_policy = sumo_server.resolve_upload_policy(
        getattr(sumo_config, "upload_policy", None)
    )
    stage0 = resolve_stage0_config(sumo_config)

    position_oracle = None
    if stage0.enabled and stage0.r_i_mode == "vmax_distance":
        positions_csv = getattr(sumo_config, "positions_csv", None)
        if positions_csv is None:
            raise ValueError(
                "[SumoRsu] stage0_r_i = 'vmax_distance' needs [sumo] "
                "positions_csv pointing at the table written by "
                "remap_positions.py."
            )
        position_oracle = PositionOracle.from_csv(positions_csv)

    algorithm = fedasync_algorithm.Algorithm
    client = simple.Client(
        algorithm=algorithm,
        trainer_callbacks=[GpuTierLogCallback],
    )

    if hasattr(Config().server, "staleness_predictor"):
        server_class = sumo_server.SumoPolarisServer
    else:
        server_class = sumo_server.SumoBaselineServer
    logging.info(
        "[SumoRsu] Server: %s  upload_policy=%s  stage0_t_hat=%s  stage0_r_i=%s",
        server_class.__name__, upload_policy, stage0.t_hat_source, stage0.r_i_mode,
    )

    server = server_class(
        oracle, upload_policy=upload_policy, stage0=stage0,
        position_oracle=position_oracle, algorithm=algorithm,
    )
    server.run(client)


if __name__ == "__main__":
    main()
