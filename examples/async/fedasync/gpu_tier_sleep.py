"""
GPU-tier-aware sleep-time computation for FedAsync IoV simulation.

Loads tiers.csv and clients.csv once (lazy, at first call) and provides a
per-call sleep_time formula so that noise_sigma can inject per-round variance
without any code changes — just edit the CSV.

Public API
----------
sleep_time_seconds(client_id, dataset_size, epochs=1) -> float
    Compute simulated training time for one client on its hardware tier.

tier_for(client_id) -> TierSpec
    Return the TierSpec for inspection / logging.

Configuration (read from TOML [gpu_tier] section via Config().config.gpu_tier)
    tiers_csv   – path to tiers.csv   (default: same directory as this file)
    clients_csv – path to clients.csv (default: same directory as this file)
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# FLOPs per sample per epoch for ResNet-18 on 32×32 input (forward + backward).
# Calibrate by replacing this constant; no code changes elsewhere needed.
# ---------------------------------------------------------------------------
W_MODEL_FLOPS: float = 1.1e9

_MODULE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Dataclasses — one per CSV row, immutable after load
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierSpec:
    """One row from tiers.csv."""
    tier_id: str
    gpu_tflops_fp16: float   # Peak FP16 throughput (TFLOPS)
    eta_nominal: float        # Utilisation efficiency (0, 1]; 1.0 = ideal
    noise_sigma: float        # Std-dev of multiplicative noise; 0 = no noise


@dataclass(frozen=True)
class ClientSpec:
    """One row from clients.csv."""
    client_id: int
    tier_id: str


# ---------------------------------------------------------------------------
# Loader + formula
# ---------------------------------------------------------------------------

class GpuTierSleepComputer:
    """Loads tier and client CSVs; computes per-call sleep times."""

    def __init__(self, tiers_csv: str | Path, clients_csv: str | Path) -> None:
        self._tiers: dict[str, TierSpec] = {}
        self._clients: dict[int, ClientSpec] = {}
        self._load(Path(tiers_csv), Path(clients_csv))

    def _load(self, tiers_path: Path, clients_path: Path) -> None:
        with tiers_path.open(newline="") as f:
            for row in csv.DictReader(f):
                spec = TierSpec(
                    tier_id=row["tier_id"],
                    gpu_tflops_fp16=float(row["gpu_tflops_fp16"]),
                    eta_nominal=float(row["eta_nominal"]),
                    noise_sigma=float(row["noise_sigma"]),
                )
                self._tiers[spec.tier_id] = spec

        with clients_path.open(newline="") as f:
            for row in csv.DictReader(f):
                spec = ClientSpec(
                    client_id=int(row["client_id"]),
                    tier_id=row["tier_id"],
                )
                self._clients[spec.client_id] = spec

        logging.info(
            "[GpuTierSleep] Loaded %d tiers (%s) and %d clients.",
            len(self._tiers),
            ", ".join(self._tiers),
            len(self._clients),
        )

    def tier_for(self, client_id: int) -> TierSpec:
        """Return the TierSpec for a given client_id."""
        client = self._clients.get(client_id)
        if client is None:
            raise KeyError(
                f"[GpuTierSleep] client_id={client_id} not found in clients CSV."
            )
        tier = self._tiers.get(client.tier_id)
        if tier is None:
            raise KeyError(
                f"[GpuTierSleep] tier_id='{client.tier_id}' "
                f"(client {client_id}) not found in tiers CSV."
            )
        return tier

    def sleep_time_seconds(
        self,
        client_id: int,
        dataset_size: int,
        epochs: int = 1,
    ) -> float:
        """
        Compute simulated training wall-time (seconds) for one client.

        Formula:
            work_flops = W_MODEL_FLOPS × dataset_size × epochs
            sleep_time = work_flops / (gpu_tflops_fp16 × 1e12 × eta_nominal)
            if noise_sigma > 0:
                sleep_time *= (1 + N(0, noise_sigma))

        Parameters
        ----------
        client_id    : PLATO client ID (1-based as PLATO assigns them)
        dataset_size : number of samples in this client's local partition
        epochs       : local training epochs for this call (typically 1)

        Returns
        -------
        Non-negative float — seconds to sleep (or account for in wall-time).
        """
        tier = self.tier_for(client_id)
        work_flops = W_MODEL_FLOPS * dataset_size * epochs
        sleep = work_flops / (tier.gpu_tflops_fp16 * 1e12 * tier.eta_nominal)
        if tier.noise_sigma > 0.0:
            # Multiplicative noise — enables per-round variance when sigma > 0.
            # Keep noise_sigma = 0 in the CSV to disable (current default).
            noise = float(np.random.normal(0.0, tier.noise_sigma))
            sleep *= 1.0 + noise
            # Multiplicative Gaussian noise: sleep *= (1 + N(0, sigma)).
            # For sigma <= 0.15 the max(0.0, ...) clip almost never fires.
            # For sigma > 0.30 the clip introduces asymmetric bias — consider switching
            # to log-normal noise (sleep *= exp(N(0, sigma))) which guarantees positivity
            # without clipping. Current default sigma=0.0 means this path is inactive.
        return max(0.0, sleep)


# ---------------------------------------------------------------------------
# Module-level singleton — lazy-initialised on first call
# ---------------------------------------------------------------------------

_computer: Optional[GpuTierSleepComputer] = None
_t_round_logged: bool = False  # log compute_pool_T_round_seconds once per process


def _resolve_paths() -> tuple[Path, Path]:
    """Read CSV paths from TOML [gpu_tier] section, fall back to module dir."""
    try:
        from plato.config import Config  # noqa: PLC0415 — lazy import
        gpu_tier_cfg = getattr(Config().config, "gpu_tier", None)
        if gpu_tier_cfg is not None:
            tiers = Path(os.path.expanduser(str(gpu_tier_cfg.tiers_csv)))
            clients = Path(os.path.expanduser(str(gpu_tier_cfg.clients_csv)))
            return tiers, clients
    except Exception:  # noqa: BLE001 — Config not yet initialised in tests
        pass
    return _MODULE_DIR / "tiers.csv", _MODULE_DIR / "clients.csv"


def get_computer() -> GpuTierSleepComputer:
    """Return the module-level singleton, initialising it on first call."""
    global _computer
    if _computer is None:
        tiers_path, clients_path = _resolve_paths()
        _computer = GpuTierSleepComputer(tiers_path, clients_path)
    return _computer


def generate_client_sleep_csv(
    output_path: str | Path,
    total_clients: int,
    dataset_size: int,
    epochs: int = 1,
    tiers_csv: str | Path | None = None,
    clients_csv: str | Path | None = None,
) -> None:
    """
    Write a PLATO-compatible sleep-time CSV (client_id, sleep_time) to
    *output_path* using the GPU-tier formula.

    Call this once whenever tiers.csv, partition_size, or epochs change.
    The resulting file is read by PLATO's 'custom' simulation_distribution
    in every client subprocess via Config.simulate_client_speed().

    Parameters
    ----------
    output_path   : destination CSV path (created/overwritten)
    total_clients : number of clients (CSV will have rows 1..total_clients)
    dataset_size  : samples per client (PLATO's partition_size)
    epochs        : local training epochs per round
    tiers_csv     : override path for tiers.csv (default: module directory)
    clients_csv   : override path for clients.csv (default: module directory)
    """
    import csv as _csv  # noqa: PLC0415

    t_path = Path(tiers_csv) if tiers_csv else _MODULE_DIR / "tiers.csv"
    c_path = Path(clients_csv) if clients_csv else _MODULE_DIR / "clients.csv"
    computer = GpuTierSleepComputer(t_path, c_path)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["client_id", "sleep_time"])
        for cid in range(1, total_clients + 1):
            sleep = computer.sleep_time_seconds(cid, dataset_size, epochs)
            writer.writerow([cid, f"{sleep:.9f}"])

    logging.info(
        "[GpuTierSleep] Generated %s (%d clients, dataset_size=%d, epochs=%d).",
        out,
        total_clients,
        dataset_size,
        epochs,
    )


def tier_for(client_id: int) -> TierSpec:
    """Convenience wrapper — returns the TierSpec for *client_id*."""
    return get_computer().tier_for(client_id)


def sleep_time_seconds(
    client_id: int,
    dataset_size: int,
    epochs: int = 1,
) -> float:
    """
    Convenience wrapper around ``GpuTierSleepComputer.sleep_time_seconds``.

    Computes on every call — no caching — so noise_sigma produces fresh
    variance each round once turned on in tiers.csv.
    """
    return get_computer().sleep_time_seconds(client_id, dataset_size, epochs)


# ---------------------------------------------------------------------------
# Pool-level T_round estimation and conversion utilities
# ---------------------------------------------------------------------------

def compute_pool_T_round_seconds(
    client_ids: list,
    partition_sizes: dict,
    local_epochs: int,
    clients_per_round: int = 1,
) -> float:
    """Return the expected inter-report interval for the async round clock (seconds).

    In async FL, one "round" = one aggregation event = one client finishes.
    With ``clients_per_round`` concurrent clients, the server clock advances as
    soon as the FASTEST of those clients finishes — not when the median one does.
    T_round must therefore be the expected minimum training time across a batch
    of ``clients_per_round`` randomly-drawn clients, not the pool median.

    Why this matters for staleness prediction
    ------------------------------------------
    The selector computes τ̂_i = T_C / T_round.  If T_round is the pool median
    and clients_per_round = 10, the denominator is ~7× too large, so every
    predicted staleness is ~7× too small (e.g. Xavier predicts 1.8 rounds but
    observes 5–12).

    Formula — E[min of P i.i.d. samples from the pool distribution]:
        Sort the distinct training-time values v_1 < v_2 < ... < v_K
        with pool probabilities p_1, p_2, ..., p_K (CDF = F_k = sum_{j<=k} p_j).
        P(min = v_k) = (1 - F_{k-1})^P - (1 - F_k)^P
        E[T_round] = sum_k v_k * P(min = v_k)

    When clients_per_round = 1 the formula reduces to the pool mean (not
    median), which is very close to the median for near-symmetric distributions.

    Parameters
    ----------
    client_ids        : iterable of PLATO client IDs (1-based)
    partition_sizes   : mapping from client_id -> number of local training samples
    local_epochs      : local training epochs per round
    clients_per_round : number of clients training concurrently (P in the formula)

    Returns
    -------
    E[min of clients_per_round samples from the pool training-time distribution]

    Notes
    -----
    Do NOT substitute gpu_tier_sleep.sleep_time_seconds() here.  That path
    applies multiplicative Gaussian noise on every call (when noise_sigma > 0),
    making T_round non-reproducible across calls.  This function always uses
    the deterministic, noise-free expected value so T_round is stable across
    the entire experiment run.
    """
    global _t_round_logged
    computer = get_computer()

    per_client_T: list = []
    tier_times: dict = {}

    for cid in client_ids:
        tier = computer.tier_for(cid)
        dataset_size = partition_sizes[cid]
        # Noise-free expected value — do NOT call sleep_time_seconds() here.
        # That path applies multiplicative Gaussian noise per call, making
        # T_round non-reproducible when noise_sigma > 0 in the CSV.
        work_flops = W_MODEL_FLOPS * dataset_size * local_epochs
        t = work_flops / (tier.gpu_tflops_fp16 * 1e12 * tier.eta_nominal)
        per_client_T.append(t)
        tier_times.setdefault(tier.tier_id, []).append((t, tier))

    # Compute E[min of clients_per_round i.i.d. samples] using order statistics.
    # This is the expected time until the first report when P clients train in
    # parallel — the correct denominator for staleness normalisation in async FL.
    P = max(1, clients_per_round)
    total = len(per_client_T)
    distinct_sorted = sorted(set(per_client_T))
    cdf = 0.0
    t_round = 0.0
    for v in distinct_sorted:
        p_v = per_client_T.count(v) / total
        p_min_equals_v = (1.0 - cdf) ** P - (1.0 - cdf - p_v) ** P
        t_round += v * p_min_equals_v
        cdf += p_v

    t_round_median = float(np.median(per_client_T))  # kept for logging comparison

    if not _t_round_logged:
        _t_round_logged = True
        logging.info(
            "[PolarisVi] T_round = %.6f s  (E[min of %d concurrent clients],"
            " pool median = %.6f s, speedup = %.2fx)",
            t_round,
            P,
            t_round_median,
            t_round_median / max(t_round, 1e-12),
        )
        for tid, entries in sorted(tier_times.items()):
            times = [e[0] for e in entries]
            spec = entries[0][1]
            logging.info(
                "[PolarisVi] T_round per-tier: %-8s  median=%.6f s  "
                "n=%d  η=%.4f  σ=%.4f",
                tid,
                float(np.median(times)),
                len(times),
                spec.eta_nominal,
                spec.noise_sigma,
            )

    return t_round


def seconds_to_rounds(
    T_seconds: float,
    T_round_seconds: float,
) -> float:
    """Convert predicted training time (seconds) to predicted staleness (rounds).

    τ̂_rounds = T_seconds / T_round_seconds

    Returns a non-negative float. Raises ValueError if T_round_seconds <= 0.
    """
    if T_round_seconds <= 0.0:
        raise ValueError(
            f"T_round_seconds must be > 0, got {T_round_seconds}"
        )
    return max(0.0, T_seconds / T_round_seconds)


def compute_pool_mean_T_seconds(
    client_ids: list,
    partition_sizes: dict,
    local_epochs: int,
) -> float:
    """Return the noise-free mean single-client training time across the pool.

    Used as the replacement-client time in gpu_tops_predict_staleness_rounds:
    when a busy slot frees during the simulation, a new client with this
    expected training time joins the queue.
    """
    computer = get_computer()
    times = []
    for cid in client_ids:
        tier = computer.tier_for(cid)
        work_flops = W_MODEL_FLOPS * partition_sizes[cid] * local_epochs
        times.append(work_flops / (tier.gpu_tflops_fp16 * 1e12 * tier.eta_nominal))
    return float(np.mean(times)) if times else 0.0


def compute_t_round_from_times(
    per_client_T: list[float],
    clients_per_round: int = 1,
) -> float:
    """Compute E[min of P i.i.d. samples] from pre-computed per-client times.

    Same order-statistic formula as compute_pool_T_round_seconds, but accepts
    training times directly so callers can supply EMA-updated estimates rather
    than re-deriving from the GPU-tier CSV.

    Returns 0.0 if per_client_T is empty.
    """
    if not per_client_T:
        return 0.0
    P = max(1, clients_per_round)
    total = len(per_client_T)
    distinct_sorted = sorted(set(per_client_T))
    cdf = 0.0
    t_round = 0.0
    for v in distinct_sorted:
        p_v = per_client_T.count(v) / total
        p_min_equals_v = (1.0 - cdf) ** P - (1.0 - cdf - p_v) ** P
        t_round += v * p_min_equals_v
        cdf += p_v
    return t_round


def gpu_tops_predict_staleness_rounds(
    t_client: float,
    busy_residuals: list,
    T_round: float,
) -> float:
    """Predict staleness using residual-aware two-phase simulation.

    Phase 1 — drain the known residuals:
        Count how many of the currently-busy clients finish before t_client.
        Each client finishing = one round increment.  Multiple clients with
        equal residuals (same tier, started together) count individually since
        PLATO increments current_round once per SequentialFedAsync step.
        Clients whose residual ≥ t_client do not finish before our client and
        are not counted.

    Phase 2 — steady-state tail:
        After the initial residuals are exhausted, replacement clients join at
        rate 1/T_round.  The remaining time is divided by T_round to get the
        expected additional events.

    The two-phase design avoids the replacement-cascade artefact that arises
    when a fixed pool_mean replacement time is used: identical-tier clients
    expire simultaneously, adding many same-valued entries that then all expire
    together in the next step.

    Parameters
    ----------
    t_client       : full training time of the newly selected client (s)
    busy_residuals : residual training times of currently-busy clients (s)
    T_round        : expected inter-report interval for the steady-state tail (s);
                     use the global compute_pool_T_round_seconds value
    """
    # Phase 1: count initial busy-client finishes that precede t_client.
    # Clients with residual >= t_client finish after us — skip them.
    early_finishes = sorted(r for r in busy_residuals if 0.0 < r < t_client)
    events = len(early_finishes)

    # Phase 2: steady-state tail after the last Phase-1 event.
    # T_round is calibrated for P concurrent clients, but after Phase 1 only
    # n_early replacement slots are "active contributors" — the slow survivors
    # (residual >= t_client) are still running but finish after us and add 0
    # events before t_client.  Scale Phase 2 by n_early / n_total so the
    # effective rate reflects only the replacement slots.
    t_last_early = early_finishes[-1] if early_finishes else 0.0
    remaining = t_client - t_last_early
    if remaining > 0.0 and T_round > 0.0:
        n_total = len(busy_residuals) or 1
        phase2_scale = max(len(early_finishes), 1) / n_total
        events += int(remaining * phase2_scale / T_round)

    return float(events)


def gpu_tops_predict_seconds(
    client_id: int,
    dataset_size: int,
    local_epochs: int,
) -> float:
    """GPU-TOPS-based prediction of training time in seconds.

    Uses the same formula as compute_pool_T_round_seconds (so the resulting
    τ̂_rounds = T_pred / T_round is dimensionally consistent), but for a
    single client.

    This is the deterministic, noise-free prediction the SELECTOR uses.
    The SIMULATOR applies noise on top of this via gpu_tier_sleep — those
    are separate code paths intentionally (selector ignores noise, simulator
    applies it). Selection bias from noise = prediction error, which is the
    quantity Exp 3 is designed to measure.
    """
    tier = tier_for(client_id)
    work_flops = W_MODEL_FLOPS * dataset_size * local_epochs
    return work_flops / (tier.gpu_tflops_fp16 * 1e12 * tier.eta_nominal)
