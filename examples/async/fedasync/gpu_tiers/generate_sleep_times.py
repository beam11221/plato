"""
Generate per-client training sleep times from GPU tier specs.

Reads two CSVs and writes one:

    input/tiers.csv    tier_id -> GPU spec (TFLOPS, efficiency, noise)
    input/clients.csv  client_id -> tier_id
        |
        v
    output/client_sleep_times.csv    client_id, sleep_time

Formula (per client, per round):

    work_flops = W_MODEL_FLOPS * dataset_size * epochs
    sleep_time = work_flops / (gpu_tflops_fp16 * 1e12 * eta_nominal)
    if noise_sigma > 0:
        sleep_time *= (1 + N(0, noise_sigma))

Change hardware behaviour by editing tiers.csv only:
    eta_nominal < 1  -> lower effective TFLOPS -> longer sleep
    noise_sigma > 0  -> multiplicative Gaussian noise per call

Usage:
    python generate_sleep_times.py --dataset-size 500 --epochs 1
    python generate_sleep_times.py --dataset-size 500 --seed 42 -o mytimes.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

# FLOPs per sample per epoch for ResNet-18 on 32x32 input (forward + backward).
# Recalibrate here if the model changes; nothing else needs to be touched.
W_MODEL_FLOPS: float = 1.1e9

_MODULE_DIR = Path(__file__).parent
DEFAULT_TIERS_CSV = _MODULE_DIR / "input" / "tiers.csv"
DEFAULT_CLIENTS_CSV = _MODULE_DIR / "input" / "clients.csv"
DEFAULT_OUTPUT_CSV = _MODULE_DIR / "output" / "client_sleep_times.csv"


@dataclass(frozen=True)
class TierSpec:
    """One row from tiers.csv (only the columns the formula uses)."""

    tier_id: str
    gpu_tflops_fp16: float  # Peak FP16 throughput (TFLOPS)
    eta_nominal: float      # Utilisation efficiency in (0, 1]; 1.0 = ideal
    noise_sigma: float      # Std-dev of multiplicative noise; 0 = deterministic


class SleepTimeGenerator:
    """Maps each client to its tier and turns the tier spec into a sleep time."""

    def __init__(
        self,
        tiers_csv: str | Path = DEFAULT_TIERS_CSV,
        clients_csv: str | Path = DEFAULT_CLIENTS_CSV,
    ) -> None:
        self.tiers: dict[str, TierSpec] = self._load_tiers(Path(tiers_csv))
        self.client_tiers: dict[int, str] = self._load_clients(Path(clients_csv))

    @staticmethod
    def _load_tiers(path: Path) -> dict[str, TierSpec]:
        with path.open(newline="") as f:
            tiers = {
                row["tier_id"]: TierSpec(
                    tier_id=row["tier_id"],
                    gpu_tflops_fp16=float(row["gpu_tflops_fp16"]),
                    eta_nominal=float(row["eta_nominal"]),
                    noise_sigma=float(row["noise_sigma"]),
                )
                for row in csv.DictReader(f)
            }
        if not tiers:
            raise ValueError(f"No tiers found in {path}.")
        return tiers

    @staticmethod
    def _load_clients(path: Path) -> dict[int, str]:
        with path.open(newline="") as f:
            clients = {int(row["client_id"]): row["tier_id"] for row in csv.DictReader(f)}
        if not clients:
            raise ValueError(f"No clients found in {path}.")
        return clients

    @property
    def client_ids(self) -> list[int]:
        """All client IDs from clients.csv, in ascending order."""
        return sorted(self.client_tiers)

    def tier_for(self, client_id: int) -> TierSpec:
        """Return the TierSpec assigned to *client_id*."""
        try:
            tier_id = self.client_tiers[client_id]
        except KeyError:
            raise KeyError(f"client_id={client_id} is not listed in clients.csv.") from None
        try:
            return self.tiers[tier_id]
        except KeyError:
            raise KeyError(
                f"tier_id='{tier_id}' (client {client_id}) is not listed in tiers.csv."
            ) from None

    def sleep_time_seconds(
        self,
        client_id: int,
        dataset_size: int,
        epochs: int = 1,
    ) -> float:
        """
        Simulated training wall-time in seconds for one client, one round.

        Recomputed on every call, so a tier with noise_sigma > 0 yields a fresh
        draw each time. Returns a non-negative float.
        """
        tier = self.tier_for(client_id)
        work_flops = W_MODEL_FLOPS * dataset_size * epochs
        sleep = work_flops / (tier.gpu_tflops_fp16 * 1e12 * tier.eta_nominal)

        if tier.noise_sigma > 0.0:
            sleep *= 1.0 + random.gauss(0.0, tier.noise_sigma)

        return max(0.0, sleep)

    def anchor_scale(
        self,
        tier_id: str,
        target_sleep: float,
        dataset_size: int,
        epochs: int = 1,
    ) -> float:
        """
        Factor that rescales all sleep times so *tier_id* sleeps *target_sleep*.

        The raw formula produces times on the order of milliseconds, which sit
        far below the framework's own per-client overhead (~2 s of payload
        processing and communication, measured).  Since that overhead is
        additive, the observed fast/slow ratio is
        (overhead + s_slow) / (overhead + s_fast) — so anchoring the FASTEST
        tier well above the overhead is what keeps the ratio close to the
        nominal TFLOPS gap.  Anchoring the slowest tier instead lets s_fast
        fall toward zero, and the ratio saturates at overhead / s_fast.

        Scaling is uniform, so every tier's relative speed is untouched.
        """
        tier = self.tiers[tier_id]
        raw = W_MODEL_FLOPS * dataset_size * epochs
        raw /= tier.gpu_tflops_fp16 * 1e12 * tier.eta_nominal
        return target_sleep / raw

    def sleep_times(
        self,
        dataset_size: int | dict[int, int],
        epochs: int = 1,
        scale: float = 1.0,
        comm_latency: float = 0.0,
    ) -> dict[int, float]:
        """
        Sleep time for every client in clients.csv.

        *dataset_size* is either one sample count shared by all clients, or a
        {client_id: sample_count} mapping for uneven partitions.
        *scale* multiplies every result uniformly (see anchor_scale).
        *comm_latency* (C = D/B, seconds) is ADDED after scaling — identical
        for every client, since bandwidth is not yet modelled per-client
        (BRIEF_comm_sweep.md § 2). C is already in the simulator's real-seconds
        units, same as the anchored sleep values, so it must not itself be
        multiplied by *scale*.
        """
        sizes = dataset_size if isinstance(dataset_size, dict) else None
        return {
            cid: scale
            * self.sleep_time_seconds(
                cid,
                sizes[cid] if sizes is not None else dataset_size,
                epochs,
            )
            + comm_latency
            for cid in self.client_ids
        }


def write_sleep_time_csv(
    output_path: str | Path,
    sleep_times: dict[int, float],
) -> Path:
    """Write a `client_id,sleep_time` CSV, creating parent directories."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "sleep_time"])
        for client_id, sleep in sorted(sleep_times.items()):
            writer.writerow([client_id, f"{sleep:.9f}"])
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--dataset-size",
        type=int,
        required=True,
        help="Number of local training samples per client (PLATO's partition_size).",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round.")
    parser.add_argument("--tiers-csv", type=Path, default=DEFAULT_TIERS_CSV)
    parser.add_argument("--clients-csv", type=Path, default=DEFAULT_CLIENTS_CSV)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible noise (only matters when noise_sigma > 0).",
    )
    parser.add_argument(
        "--anchor-tier",
        default=None,
        help="Rescale all sleep times so this tier sleeps exactly --anchor-sleep "
        "seconds. Anchor the FASTEST tier to keep the observed gap close to the "
        "nominal TFLOPS gap.",
    )
    parser.add_argument(
        "--anchor-sleep",
        type=float,
        default=None,
        help="Target sleep time in seconds for --anchor-tier.",
    )
    parser.add_argument(
        "--comm-bandwidth-mbps",
        type=float,
        default=None,
        help="Uplink bandwidth B (Mbps). Adds C = payload_mbit / B seconds to every "
        "client's sleep time, identical across clients (BRIEF_comm_sweep.md). "
        "Omit for C = 0 (default; byte-identical to no comm modelling).",
    )
    parser.add_argument(
        "--payload-mbit",
        type=float,
        default=341.52,
        help="Uplink payload D (Mbit). Default 341.52 Mbit = 42.69 MB, the "
        "measured ResNet-18 payload (HANDOFF § 4.8).",
    )
    args = parser.parse_args()
    if (args.anchor_tier is None) != (args.anchor_sleep is None):
        parser.error("--anchor-tier and --anchor-sleep must be given together.")
    if args.comm_bandwidth_mbps is not None and args.comm_bandwidth_mbps <= 0:
        parser.error("--comm-bandwidth-mbps must be > 0.")
    return args


def main() -> None:
    args = _parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(8888)

    generator = SleepTimeGenerator(args.tiers_csv, args.clients_csv)

    scale = 1.0
    if args.anchor_tier is not None:
        if args.anchor_tier not in generator.tiers:
            raise KeyError(f"tier_id='{args.anchor_tier}' is not in {args.tiers_csv}.")
        scale = generator.anchor_scale(
            args.anchor_tier, args.anchor_sleep, args.dataset_size, args.epochs
        )

    comm_latency = 0.0
    if args.comm_bandwidth_mbps is not None:
        comm_latency = args.payload_mbit / args.comm_bandwidth_mbps

    sleep_times = generator.sleep_times(args.dataset_size, args.epochs, scale, comm_latency)
    out = write_sleep_time_csv(args.output, sleep_times)

    anchor_note = (
        f", {args.anchor_tier}={args.anchor_sleep}s -> scale={scale:.4g}x"
        if args.anchor_tier
        else ""
    )
    comm_note = (
        f", C={comm_latency:.4f}s (B={args.comm_bandwidth_mbps:g} Mbps, "
        f"D={args.payload_mbit:g} Mbit)"
        if args.comm_bandwidth_mbps is not None
        else ""
    )
    print(
        f"Wrote {len(sleep_times)} clients to {out} "
        f"(dataset_size={args.dataset_size}, epochs={args.epochs}{anchor_note}{comm_note})."
    )
    for tier_id in generator.tiers:
        times = [
            sleep_times[cid]
            for cid, tid in generator.client_tiers.items()
            if tid == tier_id
        ]
        if times:
            print(
                f"  {tier_id:<10} n={len(times):<4} "
                f"sleep_time={min(times):.6f}..{max(times):.6f} s"
            )


if __name__ == "__main__":
    main()
