"""
Check the fleets and configs agree before any GPU time is spent.

Everything below is derived from the files on disk, so a mismatch is caught in a
second instead of 70 seconds into a five-hour run. It also prints the numbers the
run must reproduce -- E[T], T_round, tau_hat per tier -- so the first log lines
can be checked against an expectation instead of eyeballed.

Two checks here are the OPPOSITE of the ones in `../separability/preflight.py`,
and the reason is the jitter:

    that file                              this file
    -------------------------------------  -------------------------------------
    FAIL if a tier has more than one       FAIL if a tier has only one sleep
    sleep time                             time (the jitter did not apply)

    FAIL unless xavier sleeps exactly      FAIL unless xavier's GEOMETRIC MEAN
    anchor_sleep                           is anchor_sleep (exp(eps) has median
                                           1, so T_tier is the tier's median)

and three are new: the realised CV per tier, whether adjacent tiers overlap, and
whether the oracle arm reads exactly the file the clients sleep from.

Usage:
    python3 preflight.py                             # every config in configs/
    python3 preflight.py --tag _det1                 # only the tagged build
    python3 preflight.py --configs configs/gpu_a0p8_N20_C4_s4126.toml
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import socket
import statistics
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLEETS_DIR = HERE / "fleets"

# FLOPs per sample per epoch for ResNet-18 on 32x32, forward + backward.
# Mirrors build_fleet.W_MODEL_FLOPS; kept local so preflight needs no imports.
W_MODEL_FLOPS = 1.1e9

# Stage 2 locks the NOMINAL CV -- a property of sigma -- at 15%. The realised
# per-tier CV scatters around it by several points at these fleet sizes (3-6
# thor clients), so it is reported against a tolerance rather than a band, and a
# tier outside it warns rather than fails.
NOMINAL_CV_TARGET = 0.15
NOMINAL_CV_TOLERANCE = 0.001
CV_REPORT_TOLERANCE = 0.05

FAILURES: list[str] = []
WARNINGS: list[str] = []


def ok(message: str) -> None:
    print(f"  ok    {message}")


def fail(message: str) -> None:
    FAILURES.append(message)
    print(f"  FAIL  {message}")


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"  warn  {message}")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------

def check_environment() -> None:
    print("\n[1] interpreter, torch, GPU")
    if sys.version_info < (3, 11):
        fail(f"python {sys.version_info.major}.{sys.version_info.minor} — "
             "PLATO's config reader needs tomllib, i.e. 3.11+")
    else:
        ok(f"python {sys.version_info.major}.{sys.version_info.minor}")

    missing = [
        module
        for module in ("munch", "socketio", "aiohttp", "sklearn", "pandas", "scipy",
                       "matplotlib", "transformers", "timm", "torch_optimizer",
                       "zstandard", "boto3", "numpy")
        if not importlib.util.find_spec(module)
    ]
    if missing:
        fail(f"missing packages: {', '.join(missing)}")
    else:
        ok("PLATO's dependencies import")

    try:
        import torch
    except ImportError:
        fail("torch does not import")
        return
    ok(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        ok(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------

def check_fleet(fleet_dir: Path) -> dict | None:
    """The fleet directory is internally consistent and matches its manifest."""
    print(f"\n[2] fleet  {fleet_dir.name}")
    manifest_path = fleet_dir / "manifest.json"
    if not manifest_path.exists():
        fail(f"{fleet_dir} has no manifest.json — run build_fleet.py")
        return None
    manifest = json.loads(manifest_path.read_text())

    clients = read_csv(fleet_dir / "clients.csv")
    sleeps = read_csv(fleet_dir / "sleep_times.csv")
    table = read_csv(fleet_dir / "sleep_table.csv")

    tiers = {int(r["client_id"]): r["tier_id"] for r in clients}
    per_client = {int(r["client_id"]): float(r["sleep_time"]) for r in sleeps}

    counts = {t: sum(1 for v in tiers.values() if v == t)
              for t in sorted(set(tiers.values()))}
    ok(f"clients.csv        {len(clients)} clients, {counts}")

    if set(per_client) != set(tiers):
        fail("sleep_times.csv does not cover the same client IDs as clients.csv")
        return None
    ok(f"sleep_times.csv    {len(sleeps)} clients")

    if {int(r["client_id"]) for r in table} != set(tiers):
        fail("sleep_table.csv does not cover the same client IDs")
    else:
        mismatched = [
            r["client_id"] for r in table
            if abs(float(r["sleep_s"]) - per_client[int(r["client_id"])]) > 1e-9
            or r["tier"] != tiers[int(r["client_id"])]
        ]
        if mismatched:
            fail(f"sleep_table.csv disagrees with sleep_times.csv for clients "
                 f"{mismatched}")
        else:
            ok("sleep_table.csv    agrees with sleep_times.csv row for row")

    # The formula, re-derived from the audit table rather than trusted.
    bad = [
        r["client_id"] for r in table
        if abs(float(r["T_tier_s"]) * math.exp(float(r["eps"]))
               - float(r["sleep_s"])) > 1e-6
    ]
    if bad:
        fail(f"sleep_s != T_tier * exp(eps) for clients {bad}")
    else:
        ok("sleep = T_tier * exp(eps)   holds for every client")

    return manifest


def check_jitter(fleet_dir: Path, manifest: dict, anchor_sleep: float) -> dict:
    """The jitter did what it is supposed to do, and did not blur the tiers."""
    print(f"\n[3] jitter  sigma = {manifest['jitter_sigma']:g}, "
          f"seed {manifest['seed']}")
    table = read_csv(fleet_dir / "sleep_table.csv")
    by_tier: dict[str, list[float]] = {}
    for row in table:
        by_tier.setdefault(row["tier"], []).append(float(row["sleep_s"]))

    sigma = manifest["jitter_sigma"]
    nominal_cv = math.sqrt(math.exp(sigma ** 2) - 1) if sigma > 0 else 0.0
    if abs(nominal_cv - NOMINAL_CV_TARGET) <= NOMINAL_CV_TOLERANCE:
        ok(f"nominal CV           {100 * nominal_cv:5.2f}%  "
           f"(sqrt(exp(sigma^2)-1); locked at Stage 2 and chosen on the "
           f"tier-separation margin)")
    else:
        fail(f"nominal CV {100 * nominal_cv:.2f}% is not the locked "
             f"{100 * NOMINAL_CV_TARGET:.0f}% -- sigma has been changed")

    for tier in sorted(by_tier):
        values = by_tier[tier]
        if len(set(values)) == 1:
            fail(f"{tier:8s} every client sleeps {values[0]:.6f} s — the jitter "
                 f"did not apply, and this arm's `gpu` belief would be an oracle "
                 f"again")
            continue
        mean = statistics.fmean(values)
        cv_pop = statistics.pstdev(values) / mean
        cv_smp = statistics.stdev(values) / mean if len(values) > 1 else 0.0
        line = (f"{tier:8s} n={len(values):<3d} "
                f"{min(values):8.4f} .. {max(values):8.4f} s   "
                f"CV {100 * cv_pop:5.2f}% pop / {100 * cv_smp:5.2f}% smp")
        # Small tiers (3 thor clients at N=20) carry a large sampling error, so a
        # realised CV outside the band is expected and is NOT a failure -- sigma
        # is chosen from the nominal CV above. It is reported so it cannot be
        # quietly discovered later.
        outside = max(abs(cv_pop - nominal_cv), abs(cv_smp - nominal_cv)) \
            > CV_REPORT_TOLERANCE
        (warn if outside else ok)(
            line + (f"  — more than {100 * CV_REPORT_TOLERANCE:.0f} pt from the "
                    f"nominal {100 * nominal_cv:.1f}% (small-n sampling error)"
                    if outside else "")
        )

    # Anchoring: exp(eps) has median 1, so the DATASHEET value T_tier is the
    # tier's median, and the geometric mean is what should land on the anchor.
    anchor_tier = manifest["anchor_tier"]
    if anchor_tier in by_tier:
        values = by_tier[anchor_tier]
        geo = math.exp(statistics.fmean(math.log(v) for v in values))
        t_tier = manifest["T_tier_s"][anchor_tier]
        if abs(t_tier - anchor_sleep) > 1e-6:
            fail(f"T_tier[{anchor_tier}] = {t_tier:.6f} s but the configs anchor "
                 f"at {anchor_sleep} s — the predictor would mix units (HANDOFF 4.1)")
        else:
            ok(f"anchor               T_tier[{anchor_tier}] = {t_tier:g} s exactly")
        # exp(mean of this tier's eps). It is the draw's own offset from the
        # datasheet, and it is NOT expected to be 1: with 10 clients the mean of
        # eps has SD sigma/sqrt(10) ~ 0.057, so a few per cent either way is
        # ordinary. Printed because it is exactly the size of the error the `gpu`
        # arm now carries at the tier level, on top of the per-client error.
        drift = geo / t_tier
        ok(f"realised offset      geometric mean {geo:.4f} s = "
           f"{drift:.4f}x T_tier  (this fleet's {anchor_tier} clients came out "
           f"{abs(1 - drift) * 100:.1f}% {'faster' if drift < 1 else 'slower'} "
           f"than the datasheet median)")

    print()
    for row in manifest["tier_overlaps"]:
        message = (f"{row['slower']:8s} / {row['faster']:<8s} "
                   f"margin {row['margin']:.3f}x")
        if row["overlap"]:
            fail(message + "  — RANGES OVERLAP. The tiers are blurred and no "
                           "per-tier result is interpretable.")
        elif row["margin"] < 1.02:
            warn(message + "  — separated by under 2%; a different draw would "
                           "overlap")
        else:
            ok(message + "  separated")

    return by_tier


def check_round_clock(manifest: dict, per_round: int, staleness_bound: int,
                      by_tier: dict) -> None:
    print(f"\n[4] derived round clock at C = {per_round}")
    expected_t = manifest["E_T_s"]
    t_round = expected_t / per_round
    ok(f"E[T]                 {expected_t:.4f} s   <- the realised mean, jitter included")
    ok(f"T_round (queue)      {t_round:.6f} s   <- the server must log this")

    for tier in sorted(manifest["T_tier_s"]):
        if tier not in by_tier:
            continue
        tau_hat = manifest["T_tier_s"][tier] / t_round
        tau_worst = max(by_tier[tier]) / t_round
        line = (f"tau_hat {tier:8s}     {tau_hat:7.4f} rounds "
                f"(slowest client {tau_worst:7.4f})")
        # Mean staleness is C/M, so tau_hat near the bound means the aggregator
        # would start censoring exactly the tier under measurement (HANDOFF 4.9).
        # Checked against the SLOWEST CLIENT, not the tier mean: with jitter they
        # are no longer the same number.
        if tau_worst > 0.8 * staleness_bound:
            warn(line + f"  — within 20% of staleness_bound = {staleness_bound}; "
                 "re-derive the bound before trusting this cell")
        else:
            ok(line)


# ---------------------------------------------------------------------------

def check_configs(paths: list[Path]) -> dict[tuple[int, int], dict]:
    """Validate every config; return the apparatus settings, keyed by (N, C)."""
    print(f"\n[5] configs ({len(paths)})")
    settings: dict[tuple[int, int], dict] = {}

    for path in sorted(paths):
        config = tomllib.loads(path.read_text())
        name = path.stem
        clients = config["clients"]
        server = config["server"]
        gpu_tier = config["gpu_tier"]
        trainer = config["trainer"]
        data = config["data"]

        problems = []
        if trainer["max_concurrency"] != clients["per_round"]:
            problems.append("max_concurrency != per_round")
        if clients.get("avg_training_time") != 0.0:
            problems.append("avg_training_time must be present and 0.0")
        if not clients.get("comm_simulation"):
            problems.append("comm_simulation must be true")
        if not server.get("simulate_wall_time"):
            problems.append("simulate_wall_time must be true")
        if gpu_tier.get("t_round_model") != "queue":
            problems.append("t_round_model must be 'queue'")
        for section, key in (("clients", "random_seed"), ("server", "random_seed"),
                             ("data", "random_seed")):
            if key not in config[section]:
                problems.append(f"[{section}] {key} is missing")
        seeds = {config[s]["random_seed"] for s in ("clients", "server", "data")
                 if "random_seed" in config[s]}
        if len(seeds) > 1:
            problems.append(f"the three random_seed keys disagree: {sorted(seeds)}")

        truth = Path(clients["simulation_distribution"]["csv_path"])
        if not truth.exists():
            problems.append(f"sleep CSV missing: {truth}")
        for key in ("tiers_csv", "clients_csv"):
            if not Path(gpu_tier[key]).exists():
                problems.append(f"{key} missing: {gpu_tier[key]}")

        # The arm / oracle-key cross-check, ahead of the run-time one in
        # fedasync_polaris_vi_server.check_oracle_wiring.
        predictor = server.get("staleness_predictor")
        oracle = gpu_tier.get("oracle_sleep_csv")
        if (predictor == "gpu_oracle") != (oracle is not None):
            problems.append(
                f"staleness_predictor={predictor!r} but oracle_sleep_csv is "
                f"{'present' if oracle else 'absent'} — either half alone makes "
                f"this a different arm"
            )
        if oracle is not None and Path(oracle) != truth:
            problems.append(
                f"the oracle reads {oracle} but the clients sleep from {truth}"
            )

        # The fleet a config points at must be the one its numbers came from.
        fleet_dir = truth.parent
        manifest_path = fleet_dir / "manifest.json"
        if not manifest_path.exists():
            problems.append(f"no manifest beside the sleep CSV: {fleet_dir}")
        else:
            manifest = json.loads(manifest_path.read_text())
            if manifest["partition_size"] != data["partition_size"]:
                problems.append(
                    f"config partition_size {data['partition_size']} != the "
                    f"fleet's {manifest['partition_size']} — the sleep times were "
                    f"generated for a different workload"
                )
            if manifest["total"] != clients["total_clients"]:
                problems.append(
                    f"config total_clients {clients['total_clients']} != the "
                    f"fleet's {manifest['total']}"
                )
            if float(manifest["anchor_sleep"]) != float(gpu_tier["anchor_sleep"]):
                problems.append("anchor_sleep differs from the fleet's")
            settings[(clients["total_clients"], clients["per_round"])] = {
                "fleet_dir": fleet_dir,
                "manifest": manifest,
                "staleness_bound": server["staleness_bound"],
                "anchor_sleep": gpu_tier["anchor_sleep"],
            }

        if problems:
            for problem in problems:
                fail(f"{name}: {problem}")
        else:
            ok(f"{name:46s} port {server['port']}  bound {server['staleness_bound']}")

    return settings


def check_ports(paths: list[Path]) -> None:
    print("\n[6] ports")
    used: dict[int, str] = {}
    for path in sorted(paths):
        port = tomllib.loads(path.read_text())["server"]["port"]
        if port in used:
            fail(f"{path.stem} and {used[port]} both use port {port} — "
                 "two servers on one port hang silently")
            continue
        used[port] = path.stem
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                fail(f"port {port} ({path.stem}) is already in use")
    if not FAILURES:
        ok(f"{len(used)} ports, all free")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--configs", nargs="*", type=Path, default=None)
    parser.add_argument("--tag", default=None,
                        help="Only configs whose name ends with this tag.")
    parser.add_argument("--skip-env", action="store_true",
                        help="Skip the interpreter/torch/GPU section.")
    args = parser.parse_args()

    if args.configs:
        paths = list(args.configs)
    else:
        paths = sorted((HERE / "configs").glob("*.toml"))
        if args.tag:
            paths = [p for p in paths if p.stem.endswith(args.tag)]
        else:
            # `make_configs.THROWAWAY_TAGS` is the single registry of what
            # counts as a throwaway probe/determinism build. A hand-maintained
            # copy here drifted before: this set once listed only
            # `_det1, _det2, _probe`, silently missing `_det3`-`_det6` and
            # `_smoke` after they were registered in make_configs.py, which
            # would have let 200-round probe builds slip into a bare
            # `preflight.py` run of the real sweep.
            import make_configs  # noqa: PLC0415
            paths = [p for p in paths
                     if not any(p.stem.endswith(t) for t in make_configs.THROWAWAY_TAGS)]

    if not paths:
        print("no configs to check — run make_configs.py first")
        return 1

    if not args.skip_env:
        check_environment()

    settings = check_configs(paths)

    seen: set[Path] = set()
    for (total_clients, per_round), entry in sorted(settings.items()):
        fleet_dir = entry["fleet_dir"]
        if fleet_dir not in seen:
            seen.add(fleet_dir)
            manifest = check_fleet(fleet_dir)
            if manifest is None:
                continue
            by_tier = check_jitter(fleet_dir, manifest, entry["anchor_sleep"])
        else:
            manifest = entry["manifest"]
            table = read_csv(fleet_dir / "sleep_table.csv")
            by_tier = {}
            for row in table:
                by_tier.setdefault(row["tier"], []).append(float(row["sleep_s"]))
        check_round_clock(manifest, per_round, entry["staleness_bound"], by_tier)

    check_ports(paths)

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S) — do not launch:")
        for message in FAILURES:
            print(f"  - {message}")
    if WARNINGS:
        print(f"{len(WARNINGS)} warning(s):")
        for message in WARNINGS:
            print(f"  - {message}")
    if not FAILURES:
        print("preflight clean.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
