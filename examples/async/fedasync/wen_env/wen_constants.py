"""
Locked values for the Wen-aligned closed-population environment.

Every number here is a design decision taken 14-15 Oct 2026
(`gpu_tiers_dataset/prompts/prompt-B-environment-model-runpod.md`, "Locked
decisions"). They live in code, not in the TOML, so that two configs of one
experiment cannot disagree about them; a config only chooses the scenario, the
arm, the seed and the paths. `wen_make_configs.py` writes these values into
each config's header as a comment for the record, and nothing reads them back.

Sources are given per block; the ADRs in `../docs/decisions/` (ADR-0006 and
later) carry the reasoning.
"""

from __future__ import annotations

import math

# --- Fleet and concurrency (Wen Table 2: K = 30; Pisces / Polaris: C/N = 10%) --
N_CLIENTS = 30
CONCURRENCY = 3
TIER_COUNTS = {"xavier": 15, "orin": 10, "thor": 5}

# --- Tier-mix sweep (Prompt M Part B, predictions v3.7) ----------------------
# Xavier / Orin / Thor counts; tiers are contiguous ID blocks in that order.
# M0 is the locked fleet above; only the counts change between mixes.
FLEET_MIXES = {
    "M0": (15, 10, 5), "M1": (10, 10, 10), "M2": (5, 10, 15), "M3": (2, 8, 20),
    "T1": (15, 0, 15), "T2": (5, 0, 25), "T3": (15, 15, 0),
}

# --- Data: 50,000 CIFAR-10 training images split with no image reused --------
# 20 clients x 1,667 + 10 x 1,666 = 50,000. Clients 21-30 carry the 1,666.
TRAIN_IMAGES = 50_000
PARTITION_LARGE = 1_667
PARTITION_SMALL = 1_666
N_SMALL_PARTITIONS = 10
DIRICHLET_ALPHA = 0.8
LOCAL_EPOCHS = 1


def partition_size(client_id: int) -> int:
    """|S_i| for a 1-based client id."""
    if not 1 <= client_id <= N_CLIENTS:
        raise ValueError(f"client_id {client_id} outside 1..{N_CLIENTS}")
    if client_id > N_CLIENTS - N_SMALL_PARTITIONS:
        return PARTITION_SMALL
    return PARTITION_LARGE


# --- Training time (decided 15 Oct: anchor removed) --------------------------
# T_train = 3 * FLOPs_fwd * |S| * E / (eta * TFLOPS * 1e12).
# FLOPs_fwd is MEASURED from the model (measure_flops.py) and recorded in
# evidence/envmodel_step1.md; this constant is that measurement, pinned so the
# simulator and every estimator use one number. Tests check it against a fresh
# measurement when torch and fvcore are available.
FLOPS_FWD_PER_IMAGE = 556_651_520 * 2        # fvcore MACs x 2 -> FLOPs
TRAIN_FLOPS_MULTIPLIER = 3                   # forward + backward + optimizer
GPU_EFFICIENCY_ETA = 0.2                     # every tier, until Jetson profiling
JITTER_SIGMA = math.sqrt(math.log(1.0 + 0.15 ** 2))   # 0.149166, CV 15%

# Datasheet values, read from gpu_tiers_dataset/tiers/tiers_wen.csv at run time;
# repeated here only so the loader can refuse a tier file that drifted.
TIER_TFLOPS_FP16 = {"xavier": 10.0, "orin": 78.0, "thor": 150.0}
TIER_CPU_GHZ = {"xavier": 2.2, "orin": 2.2, "thor": 2.6}

# --- Upload: Wen et al. 2024, Eq. 5-6 ----------------------------------------
# Tr = B * log2(1 + h * p / (eps + N)),  h = d^(-phi) * |chi|^2,  T_up = D / Tr
BANDWIDTH_HZ = 2.0e6
TX_POWER_DBM = 23.0
NOISE_DBM = -114.0
PATH_LOSS_EXPONENT = 3.0
INTERFERENCE_W = 0.0            # epsilon: orthogonal sub-channels
BITS_PER_PARAMETER = 32         # D = parameter count x 32 bits, count read from the model


def dbm_to_watts(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0) / 1000.0


TX_POWER_W = dbm_to_watts(TX_POWER_DBM)      # 0.1995 W
NOISE_W = dbm_to_watts(NOISE_DBM)

# --- Geometry (Wen Fig. 2) ---------------------------------------------------
ROAD_LENGTH_M = 2_000.0
RSU_X_M = 1_000.0
RSU_LATERAL_M = 10.0
S2_COVERAGE_M = (400.0, 1_600.0)

# --- CPU-cycle estimator (Wen Eq. 4; rho from Xiao et al., T-ITS 2022) --------
BITS_PER_IMAGE = 32 * 32 * 3 * 8            # b = 24,576
CYCLES_PER_BIT_RHO = 1_250.0

# --- Energy (NVIDIA datasheets, top power mode; Wen Eq. 8) -------------------
TIER_POWER_W = {"xavier": 40.0, "orin": 75.0, "thor": 130.0}
# CPU-family ESTIMATE only (Prompt C, D2; Wen Eq. 7): E_cpu = e * rho * b * |S| * f^2.
# Ground-truth energy stays on the GPU model for every arm.
CPU_EFFECTIVE_CAPACITANCE = 1e-28
UPLOAD_POWER_W = 0.2                         # p = 23 dBm
E_MAX_J = 600.0                              # non-binding by default

# --- Scenarios ---------------------------------------------------------------
SCENARIOS = ("S1", "S2")
UPLOAD_POLICY = {"S1": "always_accept", "S2": "coverage_gated"}

# --- Polaris (Prompt J PL4): the authors' Plato-code defaults --------------
# alpha: the unseen-client G prior scale (paper's gamma, Eq. 26); A: P4's
# staleness constant; stored tau = observed staleness + 0.1, initial 0.01.
POLARIS_ALPHA = 10.0
POLARIS_A = 1.0
POLARIS_TAU_OFFSET = 0.1
POLARIS_INITIAL_TAU = 0.01
