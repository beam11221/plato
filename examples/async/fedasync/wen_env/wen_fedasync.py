"""
Entry point for the Wen-aligned closed-population environment (Prompt B).

One script for every arm; the TOML decides which:

    [wen_env]
    scenario   = "S1" | "S2"
    arm        = "random" | "gpu" | "cpu" | "gpu_oracle" | "location_only" | "observed"
    stage0     = true | false                  # S2 only
    bundle_dir = ".../sumo_ver2/bundle_wen_closed_v1"
    bundle_zip = ".../bundle_wen_closed_v1.zip"   # optional; verified if present
    fleet_dir  = ".../gpu_tiers_dataset/fleets/n30_x15-o10-t5_phys_p1667_j0p149_s4126"
    tiers_csv  = ".../gpu_tiers_dataset/tiers/tiers_wen.csv"

Refused at start: any `[gpu_tier]` key other than `t_round_model` (in
particular anchor_tier / anchor_sleep / tiers_csv), and `[server]
staleness_predictor` (the arm comes from `[wen_env] arm` only, so there is one
switch, not two that could disagree).

Seeding happens at module level, before torch is imported, so every spawned
client process is seeded too (gpu_tiers_dataset/seeding.py).

Run (from examples/async/fedasync):
    PYTHONPATH=<repo root> python wen_env/wen_fedasync.py -c wen_env/configs/<cell>.toml
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FEDASYNC = _HERE.parent
for _path in (str(_FEDASYNC), str(_FEDASYNC / "gpu_tiers"),
              str(_FEDASYNC / "gpu_tiers_dataset"), str(_FEDASYNC / "sumo_close_loop"),
              str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import seeding  # noqa: E402

SEED_RECORD = seeding.seed_everything()

import client_seeded_trainer  # noqa: E402
import fedasync_algorithm  # noqa: E402
import wen_constants as K  # noqa: E402
import wen_estimators  # noqa: E402
import wen_faithful  # noqa: E402
import wen_fresh  # noqa: E402
import wen_manifest  # noqa: E402
from wen_config import validate_config  # noqa: E402
from wen_client import WenClient  # noqa: E402
from wen_fleet import Fleet  # noqa: E402
from wen_reported import ReportedView  # noqa: E402
from wen_server import WenBaselineServer, WenPolarisServer  # noqa: E402
from wen_trace import TraceIndex, load_speeds, veh_id, verify_bundle  # noqa: E402
from wen_truth import GroundTruth  # noqa: E402

from plato.config import Config  # noqa: E402

def model_bits() -> int:
    """D = parameter count of the configured model x 32 bits, read from the model."""
    from plato.models import registry as models_registry  # noqa: PLC0415

    model = models_registry.get()
    return sum(p.numel() for p in model.parameters()) * K.BITS_PER_PARAMETER


def build_environment(settings: dict, seed: int):
    verify_bundle(settings["bundle_dir"], settings["bundle_zip"])
    speeds = load_speeds(settings["bundle_dir"])
    trace = TraceIndex.from_bundle(settings["bundle_dir"], speeds)
    fleet = Fleet.load(settings["fleet_dir"], settings["tiers_csv"])
    bits = model_bits()
    truth = GroundTruth(fleet, trace, K.FLOPS_FWD_PER_IMAGE, bits, seed,
                        settings["scenario"])

    def spec(cid):
        s = fleet.spec(cid)
        return s.tier_id, s.tflops_fp16, s.cpu_ghz

    view = ReportedView(
        position=lambda cid, t: trace.x_at(veh_id(cid), t),
        speed=lambda cid: speeds[veh_id(cid)],
        spec=spec, n_samples=K.partition_size,
    )
    arm = settings["arm"]
    estimator = None
    reports = [view.report(c, 0.0) for c in fleet.client_ids]
    rho = None
    if arm == "cpu_cal":
        # rho_cal is computed from the fleet, never typed by hand: the config's
        # value must equal a fresh computation unless this is the rho = 1,250
        # reproduction check (cpu_rho_test_override, Prompt D Step 2 test 1).
        computed = wen_estimators.rho_cal(K.FLOPS_FWD_PER_IMAGE, reports)
        rho = settings["cpu_rho"]
        if not settings["cpu_rho_test_override"] and abs(rho - computed) > 1e-9 * computed:
            raise ValueError(f"[WenEnv] cpu_rho {rho!r} != rho_cal {computed!r} "
                             f"recomputed from the fleet.")
        logging.info("[WenEnv] cpu_cal: rho = %r (rho_cal from fleet = %r, override=%s)",
                     rho, computed, settings["cpu_rho_test_override"])
    if arm in wen_estimators.ESTIMATOR_ARMS:
        estimator = wen_estimators.build(
            arm, K.FLOPS_FWD_PER_IMAGE, bits, reports,
            # The single, explicit hand-over of ground truth: gpu_oracle only.
            true_train_seconds=truth.t_train_true if arm == "gpu_oracle" else None,
            rho=rho,
        )
    return truth, view, estimator, bits


def main() -> None:
    seeding.log_seeding()
    cfg = Config()
    settings = validate_config(cfg.config)
    seed = int(cfg.clients.random_seed)
    out_dir = Path(cfg.params["base_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seeding.config_path_from_argv(), out_dir / "config_used.toml")
    manifest = wen_manifest.write(out_dir, Path(seeding.config_path_from_argv()),
                                  settings, int(cfg.trainer.rounds), seed)
    logging.info("[WenEnv] manifest: %s", manifest)

    truth, view, estimator, bits = build_environment(settings, seed)
    logging.info("[WenEnv] D = %d bits, FLOPs_fwd = %d, scenario=%s arm=%s stage0=%s "
                 "seed=%d", bits, K.FLOPS_FWD_PER_IMAGE, settings["scenario"],
                 settings["arm"], settings["stage0"], seed)

    wen = {
        "truth": truth, "view": view, "estimator": estimator, "arm": settings["arm"],
        "scenario": settings["scenario"], "stage0": settings["stage0"],
        "run_id": out_dir.name, "seed": seed, "out_dir": out_dir,
        "selection": settings["selection"], "epsilon": settings["epsilon"],
    }
    if settings["arm"] in ("wen", "wen_fresh"):
        selector_args = dict(
            estimator=wen_estimators.CpuEstimator(K.FLOPS_FWD_PER_IMAGE, bits),
            t_max=settings["wen_t_max"], e_max=settings["wen_e_max"],
            kl=wen_faithful.label_kl(
                wen_faithful.cifar10_train_targets(Path(cfg.data.data_path)),
                int(cfg.data.random_seed)),
            out_dir=out_dir,
        )
        if settings["arm"] == "wen":
            wen["selector"] = wen_faithful.WenFaithfulSelection(**selector_args)
        else:
            wen["selector"] = wen_fresh.WenFreshSelection(
                fresh=wen_fresh.FreshLoss(Path(cfg.data.data_path),
                                          int(cfg.data.random_seed)),
                stale_f3_test=settings["wen_fresh_stale_f3_test"], **selector_args)
    algorithm = fedasync_algorithm.Algorithm
    server_class = (WenBaselineServer if settings["arm"] in ("random", "wen", "wen_fresh")
                    else WenPolarisServer)
    server = server_class(wen=wen, algorithm=algorithm)
    client = WenClient(algorithm=algorithm,
                       trainer=client_seeded_trainer.ClientSeededTrainer)
    server.run(client)


if __name__ == "__main__":
    main()
