"""
`[wen_env]` config validation, kept apart from the entry point so it can be
tested without triggering the entry point's module-level seeding.
"""

from __future__ import annotations

from pathlib import Path

import wen_constants as K
import wen_estimators
from wen_egreedy import DEFAULT_EPSILON, SELECTION_RULES

# Prompt I (Study 2): plug-in arms = (host, Wen F1/F2 family). The `_s0` arms
# run Stage 0 with the gpu estimator in front of an unchanged host.
STUDY2_ARMS = {"random_s0": ("random", None), "wen_fresh_s0": ("wen_fresh", "cpu"),
               "wen_fresh_gpu": ("wen_fresh", "gpu"),
               "wen_fresh_gpu_s0": ("wen_fresh", "gpu")}
ARMS = (("random", "observed", "wen", "wen_fresh", "polaris")
        + wen_estimators.ESTIMATOR_ARMS + tuple(STUDY2_ARMS))
WEN_ARMS = ("wen", "wen_fresh")


def host_of(arm: str) -> str:
    return STUDY2_ARMS[arm][0] if arm in STUDY2_ARMS else arm
ALLOWED_GPU_TIER_KEYS = {"t_round_model"}


def validate_config(cfg) -> dict:
    """Return the [wen_env] settings, refusing anything ambiguous."""
    wen = getattr(cfg, "wen_env", None)
    if wen is None:
        raise ValueError("[WenEnv] the config has no [wen_env] section.")
    gpu_tier = getattr(cfg, "gpu_tier", None)
    if gpu_tier is None:
        keys = set()
    elif hasattr(gpu_tier, "keys"):          # plato ConfigNode is a Munch (a dict)
        keys = set(gpu_tier.keys())
    else:
        keys = set(vars(gpu_tier))
    extra = keys - ALLOWED_GPU_TIER_KEYS
    if extra:
        raise ValueError(f"[WenEnv] [gpu_tier] may only hold {sorted(ALLOWED_GPU_TIER_KEYS)};"
                         f" found {sorted(extra)}. The Wen environment has no anchor "
                         f"and reads tier specs from [wen_env] tiers_csv.")
    if hasattr(cfg.server, "staleness_predictor"):
        raise ValueError("[WenEnv] remove [server] staleness_predictor; the arm is "
                         "[wen_env] arm.")
    arm = str(wen.arm)
    scenario = str(wen.scenario)
    stage0 = bool(getattr(wen, "stage0", False))
    if arm not in ARMS:
        raise ValueError(f"[WenEnv] arm {arm!r} not in {ARMS}")
    if scenario not in K.SCENARIOS:
        raise ValueError(f"[WenEnv] scenario {scenario!r} not in {K.SCENARIOS}")
    if int(cfg.clients.total_clients) != K.N_CLIENTS or \
            int(cfg.clients.per_round) != K.CONCURRENCY:
        raise ValueError("[WenEnv] total_clients / per_round must be 30 / 3.")
    if int(cfg.trainer.epochs) != K.LOCAL_EPOCHS:
        raise ValueError("[WenEnv] trainer.epochs must be 1.")
    host = host_of(arm)
    study2 = {"host": host, "wen_f12": STUDY2_ARMS[arm][1] if arm in STUDY2_ARMS else
              ("cpu" if host in WEN_ARMS else None),
              "stage0_estimator": "gpu" if arm in STUDY2_ARMS and stage0 else None}
    if arm in STUDY2_ARMS and stage0 != arm.endswith("_s0"):
        raise ValueError(f"[WenEnv] arm {arm} needs stage0 = {arm.endswith('_s0')}.")
    # Prompt I Step 3 test 1: the plug-in arm's code path with its plug-in off
    # (Stage 0 off; Wen F1/F2 back to the CPU family, with the CPU limits).
    plugin_off = bool(getattr(wen, "study2_plugin_off_test", False))
    if plugin_off:
        if arm not in STUDY2_ARMS:
            raise ValueError("[WenEnv] study2_plugin_off_test applies to Study 2 arms only.")
        stage0 = False
        study2.update(stage0_estimator=None,
                      wen_f12="cpu" if study2["wen_f12"] else None)
    study2["study2_plugin_off_test"] = plugin_off
    if host == "polaris":
        # Prompt J: Polaris as proposed, S1 only; alpha = 10 (Plato code, PL4)
        # is the one exception to the alpha = 1 lock.
        if scenario != "S1" or stage0:
            raise ValueError("[WenEnv] polaris runs in S1 without Stage 0 (Prompt J PL6).")
        if str(getattr(wen, "selection", "polaris")) != "polaris" or hasattr(wen, "epsilon"):
            raise ValueError("[WenEnv] arm polaris uses selection = 'polaris' and no epsilon.")
        if float(cfg.server.polaris_alpha) != K.POLARIS_ALPHA:
            raise ValueError(f"[WenEnv] polaris_alpha is locked at {K.POLARIS_ALPHA} for polaris.")
        return {"arm": arm, "scenario": scenario, "stage0": False, "selection": "polaris",
                "epsilon": None, "cpu_rho": None, "cpu_rho_test_override": False, **study2,
                "bundle_dir": Path(wen.bundle_dir),
                "bundle_zip": Path(wen.bundle_zip) if hasattr(wen, "bundle_zip") else None,
                "fleet_dir": Path(wen.fleet_dir), "tiers_csv": Path(wen.tiers_csv)}
    if host != "random" and float(cfg.server.polaris_alpha) != 1.0:
        raise ValueError("[WenEnv] polaris_alpha (G_i prior alpha) is locked at 1.")
    stale_test = bool(getattr(wen, "wen_fresh_stale_f3_test", False))
    if stale_test and arm != "wen_fresh":
        raise ValueError("[WenEnv] wen_fresh_stale_f3_test applies to wen_fresh only.")
    if host in WEN_ARMS:
        # Prompt E: Wen's own rule (no epsilon), no Stage 0, rescaled limits.
        # Prompt H: wen_fresh is the same rule with F3 = the current global
        # model's loss on each vehicle's partition.
        if stage0 and arm not in STUDY2_ARMS:
            raise ValueError(f"[WenEnv] the {arm} arm runs without Stage 0 (Prompt E W7).")
        if str(getattr(wen, "selection", "wen")) != "wen" or hasattr(wen, "epsilon"):
            raise ValueError(f"[WenEnv] arm {arm} uses selection = 'wen' and no epsilon.")
        if not (hasattr(wen, "wen_t_max") and hasattr(wen, "wen_e_max")):
            raise ValueError(f"[WenEnv] arm {arm} needs [wen_env] wen_t_max and wen_e_max "
                             "(wen_limits.py).")
        return {"arm": arm, "scenario": scenario, "stage0": stage0, "selection": "wen",
                "epsilon": None, "cpu_rho": None, "cpu_rho_test_override": False,
                "wen_fresh_stale_f3_test": stale_test, **study2,
                "wen_t_max": float(wen.wen_t_max), "wen_e_max": float(wen.wen_e_max),
                "bundle_dir": Path(wen.bundle_dir),
                "bundle_zip": Path(wen.bundle_zip) if hasattr(wen, "bundle_zip") else None,
                "fleet_dir": Path(wen.fleet_dir), "tiers_csv": Path(wen.tiers_csv)}
    selection = str(getattr(wen, "selection", "egreedy"))
    epsilon = getattr(wen, "epsilon", None)
    if selection not in SELECTION_RULES:
        raise ValueError(f"[WenEnv] selection {selection!r} not in {SELECTION_RULES}")
    if selection == "egreedy":
        if epsilon is None:
            raise ValueError("[WenEnv] selection = 'egreedy' needs [wen_env] epsilon.")
        epsilon = float(epsilon)
        if host == "random" and epsilon != 1.0:
            raise ValueError("[WenEnv] the random arm is egreedy with epsilon = 1.")
        # Any epsilon other than 0.2 is an S1 gpu ablation (0 = top-1; 1 = the
        # check that the gpu arm at epsilon 1 reproduces the random arm).
        if host != "random" and epsilon != DEFAULT_EPSILON and \
                not (arm == "gpu" and scenario == "S1"):
            raise ValueError("[WenEnv] epsilon other than 0.2 is an S1 gpu ablation only.")
    else:
        if epsilon is not None:
            raise ValueError("[WenEnv] proportional takes no epsilon.")
        if not (arm == "gpu" and scenario == "S1"):
            raise ValueError("[WenEnv] proportional is an S1 gpu ablation only.")
    cpu_rho = getattr(wen, "cpu_rho", None)
    rho_override = bool(getattr(wen, "cpu_rho_test_override", False))
    if arm == "cpu_cal":
        if scenario != "S1":
            raise ValueError("[WenEnv] cpu_cal is an S1-only ablation (Prompt D).")
        if cpu_rho is None:
            raise ValueError("[WenEnv] cpu_cal needs [wen_env] cpu_rho (rho_cal, "
                             "written by wen_make_configs.py).")
        cpu_rho = float(cpu_rho)
    elif cpu_rho is not None or rho_override:
        raise ValueError("[WenEnv] cpu_rho / cpu_rho_test_override apply to cpu_cal only.")
    return {"arm": arm, "scenario": scenario, "stage0": stage0, **study2,
            "selection": selection, "epsilon": epsilon,
            "cpu_rho": cpu_rho, "cpu_rho_test_override": rho_override,
            "bundle_dir": Path(wen.bundle_dir),
            "bundle_zip": Path(wen.bundle_zip) if hasattr(wen, "bundle_zip") else None,
            "fleet_dir": Path(wen.fleet_dir), "tiers_csv": Path(wen.tiers_csv)}
