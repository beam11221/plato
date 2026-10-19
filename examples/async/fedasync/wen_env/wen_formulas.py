"""
The physical formulas of the environment, as pure functions.

Shared by the ground truth (`wen_truth.py`) and by the estimators
(`wen_estimators.py`): both sides evaluate the SAME equations, and differ only in
what they feed them -- the truth uses jittered training time, the finish
position from the trace and a fading draw; an estimator uses the datasheet, a
position extrapolated from what the vehicle reported, and |chi|^2 = 1. Keeping
the equations here means a belief can only be wrong because of its inputs,
never because it carries a second copy of the physics.

Nothing in this module holds state or reads ground truth.
"""

from __future__ import annotations

import math

import wen_constants as K


def train_seconds(
    flops_fwd_per_image: float,
    n_images: int,
    epochs: int,
    tflops_fp16: float,
    eta: float = K.GPU_EFFICIENCY_ETA,
) -> float:
    """T_train = 3 * FLOPs_fwd * |S| * E / (eta * TFLOPS * 1e12)."""
    work = K.TRAIN_FLOPS_MULTIPLIER * flops_fwd_per_image * n_images * epochs
    return work / (eta * tflops_fp16 * 1e12)


def cpu_train_seconds(n_images: int, cpu_ghz: float,
                      rho: float = K.CYCLES_PER_BIT_RHO) -> float:
    """Wen Eq. 4: T_cpu = b * |S| * rho / f, with b = 24,576 bits per image."""
    return K.BITS_PER_IMAGE * n_images * rho / (cpu_ghz * 1e9)


def wrap_x(x: float) -> float:
    """Fold a position onto the closed 2,000 m loop."""
    return x % K.ROAD_LENGTH_M


def distance_to_rsu(x: float) -> float:
    """d = sqrt((x - 1000)^2 + 10^2)."""
    return math.hypot(x - K.RSU_X_M, K.RSU_LATERAL_M)


def upload_rate_bps(d: float, chi2: float) -> float:
    """Wen Eq. 5-6: Tr = B log2(1 + d^-phi |chi|^2 p / (eps + N))."""
    gain = d ** (-K.PATH_LOSS_EXPONENT) * chi2
    snr = gain * K.TX_POWER_W / (K.INTERFERENCE_W + K.NOISE_W)
    return K.BANDWIDTH_HZ * math.log2(1.0 + snr)


def upload_seconds(d: float, chi2: float, model_bits: float) -> float:
    """T_up = D / Tr."""
    rate = upload_rate_bps(d, chi2)
    if rate <= 0.0:
        return math.inf
    return model_bits / rate


def energy_joules(tier: str, t_train: float, t_up: float) -> tuple[float, float]:
    """(E_train, E_up) = (P_tier * T_train, p * T_up)."""
    return K.TIER_POWER_W[tier] * t_train, K.UPLOAD_POWER_W * t_up


def cpu_train_energy_joules(n_images: int, cpu_ghz: float,
                            rho: float = K.CYCLES_PER_BIT_RHO) -> float:
    """Wen Eq. 7: E = e * rho * b * |S| * f^2 (CPU-family estimate only)."""
    f = cpu_ghz * 1e9
    return K.CPU_EFFECTIVE_CAPACITANCE * rho * K.BITS_PER_IMAGE * n_images * f * f


def in_s2_coverage(x: float) -> bool:
    lo, hi = K.S2_COVERAGE_M
    return lo <= x <= hi
