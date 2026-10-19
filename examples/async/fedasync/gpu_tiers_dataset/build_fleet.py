"""
Build one complete, immutable fleet: composition, sleep times, audit table, manifest.

This replaces the two-step `make_fleet.py` + `gpu_tiers/generate_sleep_times.py`
dance that `../separability/build_inputs.sh` and `../sumo_close_loop/build_inputs.sh`
perform.  Those two steps produced two files in two directories that had to agree,
and `../separability/README.md` lists the consequence as a landmine:

    The fleet file and the sleep CSV are one fleet seen from two sides.
    build_inputs.sh regenerates both; crossing files from different builds is
    silent, and preflight.py is what catches it.

Here one command writes one directory, and a config either points at that
directory or it does not.  There is nothing left to cross.

    fleets/n20_x10-o7-t3_xa15_p2500_j0p180_s4126/
        clients.csv       client_id,tier_id,schema_version   (the fleet)
        sleep_times.csv   client_id,sleep_time               (what PLATO reads)
        sleep_table.csv   client_id,tier,T_tier_s,eps,sleep_s (the audit trail)
        manifest.json     every parameter, plus sha256 of the tier file used

The formula
-----------
Unchanged from `gpu_tiers/generate_sleep_times.py` up to the anchor:

    work_flops = W_MODEL_FLOPS * partition_size * epochs
    raw(tier)  = work_flops / (gpu_tflops_fp16 * 1e12 * eta_nominal)
    scale      = anchor_sleep / raw(anchor_tier)
    T_tier     = raw(tier) * scale

and then the one thing that is new:

    sleep_i    = T_tier(tier_i) * exp(eps_i) + comm_latency
    eps_i      ~ Normal(0, jitter_sigma), drawn ONCE per client

With `--jitter-sigma 0` this reproduces the old generator's output byte for byte
(asserted by `tests/test_build_fleet.py`), so the fleets the oracle-branch runs
used can be rebuilt here and checked against what is on disk.

Why the jitter exists
---------------------
In reality T = W / (P . eta): the throughput P is on the datasheet and genuinely
knowable, but the achieved efficiency eta is not -- it depends on memory
bandwidth, thermals, batch size and background load, and two units of the same
model do not train at identical speed.

Without it, the sleep the simulator hands a client is EXACTLY the number the
`gpu` arm's predictor derives from the same tier file, so that arm is an oracle
by construction rather than an estimator, and the claim it exists to test -- that
a spec-derived estimate is good enough -- was never actually tested.

`eps_i` is drawn once per client and held for the whole run: it models a
persistent per-unit offset, not transient load.  Redrawing it per round would
model something else entirely (and would also make T_round non-reproducible,
which is the bug the old `noise_sigma` column carried -- see
`gpu_tier_sleep.compute_pool_T_round_seconds`, "Do NOT substitute
sleep_time_seconds() here").

Lognormal, not additive Gaussian, for two reasons: sleep time cannot go
negative, and the multiplicative form is what "efficiency is off by x %" means.
Note the consequence -- `T_tier` becomes the MEDIAN of its tier, not the mean;
the tier mean lands at `T_tier * exp(sigma^2/2)`, about +1.6 % at sigma = 0.18.
No correction is applied: the spec says `sleep_i = T_tier * exp(eps_i)`, and
introducing a correction would quietly move the belief/reality gap that the
`gpu` arm is there to measure.

Usage:
    python3 build_fleet.py --total 20 --jitter-sigma 0.18 --seed 4126
    python3 build_fleet.py --total 30 --jitter-sigma 0.18 --seed 4126 \
        --partition-size 1667
    python3 build_fleet.py --total 20 --jitter-sigma 0 --seed 4126 --dry-run

    # Unanchored, physical seconds (Wen-aligned environment, roadmap 15 Oct 2026)
    python3 build_fleet.py --total 30 --tiers-csv tiers/tiers_wen.csv \
        --no-anchor --w-model-flops 3.3e9 --partition-size 1667 \
        --jitter-sigma 0.149 --seed 4126
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import make_fleet

HERE = Path(__file__).resolve().parent
DEFAULT_TIERS_CSV = HERE / "tiers" / "tiers_estimated_15x.csv"
FLEETS_DIR = HERE / "fleets"

# FLOPs per sample per epoch for ResNet-18 on 32x32 input.  The historical
# default; 1.1 GFLOP is the FORWARD pass only, which made no difference while
# every fleet was anchored (the anchor scale cancels it).  Unanchored fleets
# pass --w-model-flops 3.3e9 (3 x forward: forward + backward + optimizer).
# Must stay equal to gpu_tier_sleep.W_MODEL_FLOPS -- the simulator and the
# predictor divide the same budget, and a mismatch would show up as a constant
# belief error that looks exactly like a real estimator bias.
# `tests/test_build_fleet.py::test_w_model_flops_matches_runtime` pins them.
W_MODEL_FLOPS: float = 1.1e9

# Directory-name tag for an unanchored fleet: sleep times straight from the
# physical formula, in seconds, with no rescaling.
PHYSICAL_TAG = "phys"

# Uplink payload D (Mbit): 42.69 MB, the measured ResNet-18 payload
# (gpu_tiers/HANDOFF.md section 4.8).
DEFAULT_PAYLOAD_MBIT = 341.52

# Bumped whenever a change here would alter the bytes this script writes for
# inputs that are otherwise identical. It rides in the manifest digest, so a
# fleet built by an older version cannot silently pass as one built by this one.
GENERATOR_VERSION = 1

SLEEP_FORMAT = "{:.9f}"   # matches gpu_tiers/generate_sleep_times.py exactly


@dataclass(frozen=True)
class TierSpec:
    """The columns of tiers.csv this formula uses."""

    tier_id: str
    gpu_tflops_fp16: float
    eta_nominal: float


def load_tiers(path: Path) -> dict[str, TierSpec]:
    with path.open(newline="") as fh:
        tiers = {
            row["tier_id"]: TierSpec(
                tier_id=row["tier_id"],
                gpu_tflops_fp16=float(row["gpu_tflops_fp16"]),
                eta_nominal=float(row["eta_nominal"]),
            )
            for row in csv.DictReader(fh)
        }
    if not tiers:
        raise SystemExit(f"no tiers found in {path}")
    return tiers


def raw_seconds(
    tier: TierSpec,
    partition_size: int,
    epochs: int,
    w_model_flops: float = W_MODEL_FLOPS,
) -> float:
    """Noise-free training time from the tier formula, before anchoring."""
    work_flops = w_model_flops * partition_size * epochs
    denominator = tier.gpu_tflops_fp16 * 1e12 * tier.eta_nominal
    if denominator <= 0.0:
        raise SystemExit(
            f"tier {tier.tier_id!r} has non-positive throughput x efficiency "
            f"({tier.gpu_tflops_fp16} x {tier.eta_nominal})"
        )
    return work_flops / denominator


def anchor_scale(
    tiers: dict[str, TierSpec],
    anchor_tier: str,
    anchor_sleep: float,
    partition_size: int,
    epochs: int,
) -> float:
    """Factor that rescales every tier so *anchor_tier* sleeps *anchor_sleep* s.

    The raw formula produces milliseconds, far below PLATO's own ~2 s per-client
    overhead.  That overhead is ADDITIVE, so the observed fast/slow ratio is
    (overhead + s_slow) / (overhead + s_fast): anchoring the FASTEST tier lets
    s_fast fall toward zero and the ratio saturates, which is why the anchor tier
    is the SLOWEST one.  Scaling is uniform, so relative tier speeds -- and
    therefore tau_hat = T_i / T_round -- are untouched by the anchor.
    """
    if anchor_tier not in tiers:
        raise SystemExit(
            f"anchor tier {anchor_tier!r} is not in the tier file "
            f"(have: {', '.join(sorted(tiers))})"
        )
    raw = raw_seconds(tiers[anchor_tier], partition_size, epochs)
    if raw <= 0.0:
        raise SystemExit(f"anchor tier {anchor_tier!r} has non-positive raw time")
    return anchor_sleep / raw


def tier_times(
    tiers: dict[str, TierSpec],
    partition_size: int,
    epochs: int,
    scale: float,
    w_model_flops: float = W_MODEL_FLOPS,
) -> dict[str, float]:
    """T_tier in seconds, per tier, after anchoring. This is the DATASHEET value.

    It is what the `gpu` arm believes and what the jitter is applied on top of;
    it is deliberately NOT the realised mean of any tier's clients.
    """
    return {
        tier_id: raw_seconds(spec, partition_size, epochs, w_model_flops) * scale
        for tier_id, spec in tiers.items()
    }


def draw_epsilons(
    client_ids: list[int],
    jitter_sigma: float,
    seed: int,
) -> dict[int, float]:
    """One persistent per-unit offset per client, in ascending client_id order.

    A dedicated `random.Random` instance, not the `random` module singleton, so
    the draw cannot move because some other part of the process consumed the
    global stream first.  Ascending client_id order so the draw does not depend
    on dict iteration order or on how the fleet happens to be grouped.

    `jitter_sigma == 0` short-circuits to all-zero WITHOUT touching the RNG, so
    a no-jitter build is byte-identical to the old generator's output.
    """
    if jitter_sigma < 0.0:
        raise SystemExit(f"--jitter-sigma must be >= 0, got {jitter_sigma}")
    if jitter_sigma == 0.0:
        return {cid: 0.0 for cid in client_ids}

    rng = random.Random(seed)
    return {cid: rng.gauss(0.0, jitter_sigma) for cid in sorted(client_ids)}


def sleep_times(
    client_tiers: dict[int, str],
    t_tier: dict[str, float],
    epsilons: dict[int, float],
    comm_latency: float = 0.0,
) -> dict[int, float]:
    """sleep_i = T_tier(tier_i) * exp(eps_i) + comm_latency.

    `comm_latency` (C = D/B) is added AFTER the jitter and is identical for every
    client: bandwidth is not modelled per client, and C is already in the
    simulator's real-seconds units, so it is neither scaled nor jittered.
    """
    out: dict[int, float] = {}
    for cid in sorted(client_tiers):
        tier_id = client_tiers[cid]
        if tier_id not in t_tier:
            raise SystemExit(
                f"client {cid} is tier {tier_id!r}, which is not in the tier file"
            )
        out[cid] = t_tier[tier_id] * math.exp(epsilons[cid]) + comm_latency
    return out


# ---------------------------------------------------------------------------
# Realised statistics -- what actually landed on disk, per tier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierStats:
    tier_id: str
    n: int
    t_tier_s: float          # the datasheet value, = the median of the tier
    minimum: float
    mean: float
    maximum: float
    cv_population: float     # SD/mean with ddof=0
    cv_sample: float         # SD/mean with ddof=1 -- reported too, see below

    def as_dict(self) -> dict:
        return {
            "tier_id": self.tier_id,
            "n": self.n,
            "T_tier_s": self.t_tier_s,
            "min_s": self.minimum,
            "mean_s": self.mean,
            "max_s": self.maximum,
            "cv_population": self.cv_population,
            "cv_sample": self.cv_sample,
        }


def tier_stats(
    client_tiers: dict[int, str],
    sleeps: dict[int, float],
    t_tier: dict[str, float],
) -> dict[str, TierStats]:
    """Per-tier spread of the sleep times actually written.

    Both CV conventions are reported because the tiers are small -- 3 thor
    clients at N = 20 -- and the two differ by sqrt(n/(n-1)), which is 22 % at
    n = 3. Quoting one number without saying which estimator produced it would
    make a 15-20 % target unfalsifiable.
    """
    stats: dict[str, TierStats] = {}
    for tier_id in sorted(set(client_tiers.values())):
        values = [sleeps[cid] for cid in sorted(client_tiers) if client_tiers[cid] == tier_id]
        mean = statistics.fmean(values)
        pop_sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        smp_sd = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[tier_id] = TierStats(
            tier_id=tier_id,
            n=len(values),
            t_tier_s=t_tier[tier_id],
            minimum=min(values),
            mean=mean,
            maximum=max(values),
            cv_population=pop_sd / mean if mean else 0.0,
            cv_sample=smp_sd / mean if mean else 0.0,
        )
    return stats


def tier_overlaps(
    stats: dict[str, TierStats],
    order: tuple[str, ...] = make_fleet.TIERS,
) -> list[tuple[str, str, bool, float]]:
    """Adjacent tiers, slowest first: (slower, faster, overlap, margin).

    Jitter must not blur the tiers: if the slowest client of a fast tier is
    slower than the fastest client of the tier above it, "tier" has stopped
    being a speed class and any per-tier result becomes uninterpretable.

    `margin` = slower.min / faster.max, so > 1 means separated and the value
    says by how much. The boolean alone hides the thing worth watching -- a gap
    of 1.05 and a gap of 1.27 are both "separated", but only one of them would
    survive a slightly different draw, and the fleet-size sweep changes the draw
    at every N.
    """
    present = [t for t in order if t in stats]
    out: list[tuple[str, str, bool, float]] = []
    for slower, faster in zip(present, present[1:]):
        # `slower` is the slower tier, so its minimum is the low end of its range.
        low = stats[slower].minimum
        high = stats[faster].maximum
        out.append((slower, faster, low <= high, low / high if high else math.inf))
    return out


# ---------------------------------------------------------------------------
# Writing the fleet directory
# ---------------------------------------------------------------------------

def sigma_tag(jitter_sigma: float) -> str:
    """`0.18` -> `j0p180`. Three decimals, no dot (dots read as extensions)."""
    return "j" + f"{jitter_sigma:.3f}".replace(".", "p")


def fleet_dir_name(
    counts: tuple[int, int, int],
    anchor_tier: str | None,
    anchor_sleep: float | None,
    partition_size: int,
    jitter_sigma: float,
    seed: int,
) -> str:
    """The directory a fleet is stored under.

    Human-readable on the axes that are swept, and the manifest's `inputs_digest`
    covers everything else (tier file contents, epochs, comm latency, FLOPs,
    generator version).  `build()` refuses to overwrite a directory whose
    manifest digest disagrees, so a parameter that is not in the name still
    cannot be silently swapped underneath one.  An unanchored fleet carries
    PHYSICAL_TAG where an anchored one carries e.g. `xa15`.
    """
    anchor = (
        PHYSICAL_TAG if anchor_tier is None
        else f"{anchor_tier[:2]}{anchor_sleep:g}"
    )
    return "_".join((
        make_fleet.fleet_tag(counts),
        anchor,
        f"p{partition_size}",
        sigma_tag(jitter_sigma),
        f"s{seed}",
    ))


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs_digest(payload: dict) -> str:
    """Stable hash over every input that can change the bytes written.

    Sorted keys and a compact separator so the digest does not move with dict
    ordering or json formatting.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def write_sleep_times_csv(path: Path, sleeps: dict[int, float]) -> Path:
    """The file PLATO reads: exactly `client_id,sleep_time`, nothing else.

    `plato/config.py` reads it with a DictReader and would tolerate extra
    columns, but the audit fields live in `sleep_table.csv` instead so that the
    file handed to the simulator stays the same shape it has always been.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["client_id", "sleep_time"])
        for cid in sorted(sleeps):
            writer.writerow([cid, SLEEP_FORMAT.format(sleeps[cid])])
    return path


def write_sleep_table_csv(
    path: Path,
    client_tiers: dict[int, str],
    t_tier: dict[str, float],
    epsilons: dict[int, float],
    sleeps: dict[int, float],
) -> Path:
    """The audit trail: where each client's sleep came from, term by term."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["client_id", "tier", "T_tier_s", "eps", "sleep_s"])
        for cid in sorted(client_tiers):
            writer.writerow([
                cid,
                client_tiers[cid],
                SLEEP_FORMAT.format(t_tier[client_tiers[cid]]),
                f"{epsilons[cid]:.9f}",
                SLEEP_FORMAT.format(sleeps[cid]),
            ])
    return path


def build(
    total: int,
    mix: tuple[int, int, int],
    tiers_csv: Path,
    anchor_tier: str | None,
    anchor_sleep: float | None,
    partition_size: int,
    epochs: int,
    jitter_sigma: float,
    seed: int,
    comm_bandwidth_mbps: float | None = None,
    payload_mbit: float = DEFAULT_PAYLOAD_MBIT,
    fleets_dir: Path = FLEETS_DIR,
    force: bool = False,
    dry_run: bool = False,
    w_model_flops: float = W_MODEL_FLOPS,
) -> tuple[Path, dict]:
    """Build one fleet directory. Returns (directory, manifest).

    Refuses to write over an existing fleet whose manifest digest differs, since
    that is the case where a config still pointing at the directory would pick up
    a different fleet than the one its results were produced with. Rebuilding the
    SAME fleet is allowed and is how a fleet gets verified against what is on
    disk.

    `anchor_tier=None` builds an UNANCHORED fleet: scale = 1.0, so T_tier is the
    physical formula itself, and the tier file's `eta_nominal` carries the
    achieved efficiency the anchor used to absorb.  `anchor_sleep` must then be
    None too -- one without the other is a half-specified build.
    """
    if (anchor_tier is None) != (anchor_sleep is None):
        raise SystemExit(
            "anchor_tier and anchor_sleep must be given together, or both "
            "omitted (--no-anchor) for an unanchored fleet."
        )
    if w_model_flops <= 0.0:
        raise SystemExit(f"--w-model-flops must be > 0, got {w_model_flops}")
    if anchor_tier is not None and w_model_flops != W_MODEL_FLOPS:
        # The anchor scale cancels W exactly, so on an anchored fleet the flag
        # would change nothing but the digest -- a second name for one fleet.
        raise SystemExit(
            "--w-model-flops only applies to an unanchored fleet (--no-anchor); "
            "the anchor scale cancels it."
        )

    tiers = load_tiers(tiers_csv)
    counts = make_fleet.counts_for(total, mix)
    fleet_rows = make_fleet.rows_for(counts)
    client_tiers = {int(r["client_id"]): r["tier_id"] for r in fleet_rows}

    if anchor_tier is None:
        scale = 1.0
    else:
        scale = anchor_scale(tiers, anchor_tier, anchor_sleep, partition_size, epochs)
    t_tier = tier_times(tiers, partition_size, epochs, scale, w_model_flops)
    epsilons = draw_epsilons(sorted(client_tiers), jitter_sigma, seed)

    comm_latency = 0.0
    if comm_bandwidth_mbps is not None:
        if comm_bandwidth_mbps <= 0.0:
            raise SystemExit(
                f"--comm-bandwidth-mbps must be > 0, got {comm_bandwidth_mbps}"
            )
        comm_latency = payload_mbit / comm_bandwidth_mbps

    sleeps = sleep_times(client_tiers, t_tier, epsilons, comm_latency)
    stats = tier_stats(client_tiers, sleeps, t_tier)

    digest_payload = {
        "generator_version": GENERATOR_VERSION,
        "w_model_flops": w_model_flops,
        "total": total,
        "mix": list(mix),
        "counts": list(counts),
        "tiers_sha256": sha256_of(tiers_csv),
        "anchor_tier": anchor_tier,
        "anchor_sleep": anchor_sleep,
        "partition_size": partition_size,
        "epochs": epochs,
        "jitter_sigma": jitter_sigma,
        "seed": seed,
        "comm_latency_s": comm_latency,
    }
    digest = inputs_digest(digest_payload)

    name = fleet_dir_name(counts, anchor_tier, anchor_sleep, partition_size,
                          jitter_sigma, seed)
    directory = fleets_dir / name

    manifest = dict(digest_payload)
    manifest.update({
        "fleet": name,
        "inputs_digest": digest,
        "tiers_csv": str(tiers_csv),
        "anchor_scale": scale,
        "T_tier_s": {tier: t_tier[tier] for tier in sorted(t_tier)},
        "E_T_s": statistics.fmean(sleeps.values()),
        "tier_stats": {tier: stats[tier].as_dict() for tier in sorted(stats)},
        "tier_overlaps": [
            {"slower": a, "faster": b, "overlap": bool(o), "margin": m}
            for a, b, o, m in tier_overlaps(stats)
        ],
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    if dry_run:
        return directory, manifest

    existing = directory / "manifest.json"
    if existing.exists() and not force:
        previous = json.loads(existing.read_text())
        if previous.get("inputs_digest") != digest:
            raise SystemExit(
                f"{directory} already holds a fleet built from DIFFERENT inputs "
                f"(digest {previous.get('inputs_digest', '?')[:12]} on disk, "
                f"{digest[:12]} requested). Configs already point at this "
                f"directory; overwriting it would change their fleet without "
                f"changing their name. Pick different parameters, or pass "
                f"--force if you are certain nothing depends on it."
            )

    directory.mkdir(parents=True, exist_ok=True)
    make_fleet.write_fleet_csv(directory / "clients.csv", fleet_rows)
    write_sleep_times_csv(directory / "sleep_times.csv", sleeps)
    write_sleep_table_csv(directory / "sleep_table.csv", client_tiers, t_tier,
                          epsilons, sleeps)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory, manifest


def format_summary(manifest: dict) -> str:
    """The lines a human should read before spending GPU time on a fleet."""
    lines = [
        f"fleet          {manifest['fleet']}",
        f"tier file      {Path(manifest['tiers_csv']).name} "
        f"(sha256 {manifest['tiers_sha256'][:12]})",
        (
            f"anchor         none -- physical formula, W = "
            f"{manifest['w_model_flops']:.4g} FLOP/sample"
            if manifest["anchor_tier"] is None
            else f"anchor         {manifest['anchor_tier']} = "
            f"{manifest['anchor_sleep']:g} s (scale {manifest['anchor_scale']:.6g}x)"
        ),
        f"jitter sigma   {manifest['jitter_sigma']:g}   seed {manifest['seed']}",
        f"E[T]           {manifest['E_T_s']:.4f} s",
        "",
        f"{'tier':8s} {'n':>3s} {'T_tier':>9s} {'min':>9s} {'mean':>9s} "
        f"{'max':>9s} {'CV(pop)':>8s} {'CV(smp)':>8s}",
    ]
    for tier, row in manifest["tier_stats"].items():
        lines.append(
            f"{tier:8s} {row['n']:3d} {row['T_tier_s']:9.4f} {row['min_s']:9.4f} "
            f"{row['mean_s']:9.4f} {row['max_s']:9.4f} "
            f"{100 * row['cv_population']:7.2f}% {100 * row['cv_sample']:7.2f}%"
        )
    lines.append("")
    for row in manifest["tier_overlaps"]:
        verdict = "OVERLAP" if row["overlap"] else "separated"
        lines.append(
            f"{row['slower']:8s} / {row['faster']:<8s} {verdict:9s} "
            f"margin {row['margin']:.3f}x  "
            f"(slowest {row['slower']} / fastest {row['faster']})"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--total", type=int, required=True, help="Fleet size N.")
    parser.add_argument(
        "--mix", type=int, nargs=3, metavar=("XAVIER", "ORIN", "THOR"),
        default=[50, 35, 15],
        help="Percent per tier, slowest first (default: 50 35 15). A fleet that "
             "does not divide evenly gives the remainder to the fastest tier; "
             "the directory name carries the resulting COUNTS.",
    )
    parser.add_argument("--tiers-csv", type=Path, default=DEFAULT_TIERS_CSV)
    parser.add_argument("--anchor-tier", default="xavier",
                        help="Tier pinned to --anchor-sleep. Anchor the SLOWEST "
                             "tier; see anchor_scale().")
    parser.add_argument("--anchor-sleep", type=float, default=15.0)
    parser.add_argument(
        "--no-anchor", action="store_true",
        help="Build an unanchored fleet: sleep times straight from the physical "
             "formula (scale 1.0). --anchor-tier/--anchor-sleep are ignored; "
             "efficiency comes from the tier file's eta_nominal.",
    )
    parser.add_argument(
        "--w-model-flops", type=float, default=W_MODEL_FLOPS,
        help="Training FLOPs per sample per epoch (default %(default)g, the "
             "historical forward-only figure). Unanchored fleets only.",
    )
    parser.add_argument("--partition-size", type=int, required=True,
                        help="PLATO's [data] partition_size -- local samples per client.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--jitter-sigma", type=float, required=True,
        help="SD of eps_i in sleep_i = T_tier * exp(eps_i). 0 reproduces the "
             "pre-jitter generator byte for byte.",
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed for the per-client eps draw.")
    parser.add_argument("--comm-bandwidth-mbps", type=float, default=None,
                        help="Uplink B (Mbps); adds C = D/B seconds to every "
                             "client. Omit for C = 0.")
    parser.add_argument("--payload-mbit", type=float, default=DEFAULT_PAYLOAD_MBIT)
    parser.add_argument("--fleets-dir", type=Path, default=FLEETS_DIR)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite a fleet directory built from different inputs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the summary; write nothing.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    directory, manifest = build(
        total=args.total,
        mix=tuple(args.mix),
        tiers_csv=args.tiers_csv,
        anchor_tier=None if args.no_anchor else args.anchor_tier,
        anchor_sleep=None if args.no_anchor else args.anchor_sleep,
        partition_size=args.partition_size,
        epochs=args.epochs,
        jitter_sigma=args.jitter_sigma,
        seed=args.seed,
        comm_bandwidth_mbps=args.comm_bandwidth_mbps,
        payload_mbit=args.payload_mbit,
        fleets_dir=args.fleets_dir,
        force=args.force,
        dry_run=args.dry_run,
        w_model_flops=args.w_model_flops,
    )
    print(format_summary(manifest))
    print()
    print(f"{'would write' if args.dry_run else 'wrote'}  {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
