"""
Summarise a generated heterogeneity sweep — nominal vs. EFFECTIVE gap.

The tier formula controls only the injected sleep.  PLATO charges the server's
simulated wall clock with the whole client turnaround
(``servers/base.py``: finish_time = training_time + processing_time + comm_time),
so every client carries an additive floor c on top of its sleep: real training,
payload processing, and communication.  The heterogeneity the server actually
observes is therefore

    effective_gap = (c + sleep_slowest) / (c + sleep_fastest)

not ``sleep_slowest / sleep_fastest``.  Report the effective gap — it is the
real x-axis of the experiment.

Measure c for your setup from a run whose sleep times are ~0: take the median
of "Sending the current model to client #N" -> "Received ... from client #N"
in server_*.log.  The 2.0 s default came from
runs/..._gpu_tops_ver3_log (ResNet-18, 42.69 MB payload, comm_simulation=false).

With --predictor-tiers, two more columns describe the staleness predictor's own
view of the pool.  The predictor (``gpu_tops_predict_seconds``) reads the tiers
CSV named in ``[gpu_tier] tiers_csv``, which is NOT required to be the CSV the
clients actually sleep by — the ``staleness_predictor = "cpu"`` arm points it at
tiers_cpu.csv on purpose, so the predictor sees CPU heterogeneity (cores x GHz,
1.95x) while training really happens at GPU speed.  The predictor has no
additive floor, so its perceived spread is the raw throughput ratio.

    pred_gap  slowest/fastest throughput ratio the predictor believes in
    mismatch  effective / pred_gap — >1 means the predictor under-estimates how
              spread out the pool really is (it will under-predict staleness for
              stragglers); <1 means it over-estimates.

Usage:
    python summarize_sweep.py --anchor-sleep 20
    python summarize_sweep.py --anchor-sleep 20 --predictor-tiers input/tiers_cpu.csv
    python summarize_sweep.py --pattern 'output/client_sleep_times_gap*.csv' --floor 2.0
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
DEFAULT_CLIENTS_CSV = _MODULE_DIR / "input" / "clients.csv"
DEFAULT_FLOOR_SECONDS = 2.0

# Slowest tier first — the order the summary table prints.
TIER_ORDER = ["xavier", "orin", "thor"]


def load_client_tiers(path: Path) -> dict[int, str]:
    """client_id -> tier_id, from clients.csv."""
    with path.open(newline="") as f:
        return {int(row["client_id"]): row["tier_id"] for row in csv.DictReader(f)}


def per_tier_sleep(csv_path: Path, client_tiers: dict[int, str]) -> dict[str, float]:
    """Median sleep time per tier for one generated sweep file."""
    times: dict[str, list[float]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            tier = client_tiers[int(row["client_id"])]
            times.setdefault(tier, []).append(float(row["sleep_time"]))
    return {tier: statistics.median(values) for tier, values in times.items()}


def pool_mean_sleep(csv_path: Path) -> float:
    """Mean sleep time across every client — the per-round compute budget."""
    with csv_path.open(newline="") as f:
        return statistics.mean(float(row["sleep_time"]) for row in csv.DictReader(f))


def predictor_gap(tiers_csv: Path) -> float:
    """Slowest/fastest training-time ratio implied by a tiers CSV.

    Mirrors gpu_tops_predict_seconds: time is inversely proportional to
    gpu_tflops_fp16 * eta_nominal, so the ratio is the inverse throughput ratio.
    (For CPU tier files that column holds cores * GHz / 1000 — see make_cpu_tiers.py.)
    """
    with tiers_csv.open(newline="") as f:
        throughput = {
            row["tier_id"]: float(row["gpu_tflops_fp16"]) * float(row["eta_nominal"])
            for row in csv.DictReader(f)
        }
    return throughput[TIER_ORDER[-1]] / throughput[TIER_ORDER[0]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--anchor-sleep",
        default=None,
        help="Shorthand for --pattern 'output/client_sleep_times_xa<VALUE>_gap*.csv'.",
    )
    parser.add_argument("--pattern", default=None, help="Glob of sweep CSVs to summarise.")
    parser.add_argument("--clients-csv", type=Path, default=DEFAULT_CLIENTS_CSV)
    parser.add_argument(
        "--floor",
        type=float,
        default=DEFAULT_FLOOR_SECONDS,
        help="Additive per-client floor c in seconds (comm + real training + processing).",
    )
    parser.add_argument(
        "--predictor-tiers",
        type=Path,
        default=None,
        help="Tiers CSV the staleness predictor reads ([gpu_tier] tiers_csv); "
        "adds pred_gap and mismatch columns.",
    )
    args = parser.parse_args()
    if args.pattern is None:
        anchor = args.anchor_sleep if args.anchor_sleep is not None else "*"
        args.pattern = f"output/client_sleep_times_xa{anchor}_gap*.csv"
    return args


def main() -> None:
    args = _parse_args()
    client_tiers = load_client_tiers(args.clients_csv)
    paths = sorted(Path(p) for p in glob.glob(str(_MODULE_DIR / args.pattern)))
    if not paths:
        raise SystemExit(f"No files matched {args.pattern}.")

    c = args.floor
    print(f"Additive floor c = {c:.2f} s per client")

    pred_gap = None
    if args.predictor_tiers is not None:
        pred_gap = predictor_gap(args.predictor_tiers)
        print(f"Predictor tiers  = {args.predictor_tiers} (perceived gap {pred_gap:.2f}x)")

    header = ["file"] + TIER_ORDER + ["pool_mean", "nominal", "effective"]
    if pred_gap is not None:
        header += ["pred_gap", "mismatch"]
    print("\n" + f"{header[0]:<44}" + "".join(f"{h:>10}" for h in header[1:]))

    for path in paths:
        sleeps = per_tier_sleep(path, client_tiers)
        slow = sleeps[TIER_ORDER[0]]
        fast = sleeps[TIER_ORDER[-1]]
        nominal = slow / fast if fast > 0 else float("inf")
        effective = (c + slow) / (c + fast)
        cells = [f"{sleeps.get(t, 0.0):>10.3f}" for t in TIER_ORDER]
        row = (
            f"{path.name:<44}"
            + "".join(cells)
            + f"{pool_mean_sleep(path):>10.3f}"
            + f"{nominal:>10.1f}"
            + f"{effective:>10.1f}"
        )
        if pred_gap is not None:
            row += f"{pred_gap:>10.2f}" + f"{effective / pred_gap:>10.2f}"
        print(row)


if __name__ == "__main__":
    main()
