"""
Invariants of the fleet-mix sweep (HANDOFF § 4d).

The sweep's claim is "same GPU speeds, same data, same seed — only the ratio
moves".  Each check below pins one half of that sentence, because every one of
them is silent when it breaks: a crossed sleep file, a fleet that lost a client,
or a mix whose tier speeds drifted all produce runs that look completely normal
and answer a different question.

Run:
    cd plato/examples/async/fedasync/gpu_tiers
    python3 test_mix_sweep.py
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import make_configs  # noqa: E402
import make_mixes  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures: list[str] = []

ANCHOR = 20
GAP = "30"
MIXES = ["m50_35_15", "m70_20_10", "m34_33_33", "m15_35_50"]
TIER_ORDER = make_mixes.TIER_ORDER


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{PASS if ok else FAIL}] {name}{f': {detail}' if detail else ''}")
    if not ok:
        failures.append(name)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
print("\n1. Every fleet covers all 100 clients, in contiguous slow->fast blocks")

fleets: dict[str, list[str]] = {}
for mix in MIXES:
    rows = read_csv(_HERE / "input" / f"clients_{mix}.csv")
    tiers = [r["tier_id"] for r in rows]
    fleets[mix] = tiers
    ids = [int(r["client_id"]) for r in rows]
    check(f"{mix}: 100 clients, ids 1..100", ids == list(range(1, 101)), f"n={len(ids)}")
    # Contiguous means the tier sequence has exactly one run per tier, in order.
    runs = [t for i, t in enumerate(tiers) if i == 0 or tiers[i - 1] != t]
    check(f"{mix}: blocks in slow->fast order", runs == [t for t in TIER_ORDER if t in tiers],
          " ".join(runs))
    counts = tuple(tiers.count(t) for t in TIER_ORDER)
    check(f"{mix}: counts match its name", "m" + "_".join(map(str, counts)) == mix, str(counts))

# ---------------------------------------------------------------------------
print("\n2. The fleets are nested — a client only ever moves to a FASTER tier")
# Sorting by fast share puts them in promotion order; then no client may move
# backwards.  This is what makes two cells differ by a known set of clients
# rather than by an arbitrary re-draw (§ 4d).
rank = {t: i for i, t in enumerate(TIER_ORDER)}
by_fast = sorted(MIXES, key=lambda m: fleets[m].count(TIER_ORDER[0]), reverse=True)
for slower, faster in zip(by_fast, by_fast[1:]):
    moved_back = [
        cid for cid, (a, b) in enumerate(zip(fleets[slower], fleets[faster]), start=1)
        if rank[b] < rank[a]
    ]
    check(f"{slower} -> {faster}: no client slows down", not moved_back,
          f"{len(moved_back)} moved back" if moved_back else "")

# ---------------------------------------------------------------------------
print("\n3. Same GPU speeds in every fleet — only the assignment moves")
per_mix_speeds = {}
for mix in MIXES:
    sleeps = {
        int(r["client_id"]): float(r["sleep_time"])
        for r in read_csv(_HERE / "output" / f"client_sleep_times_xa{ANCHOR}_gap{GAP}_{mix}.csv")
    }
    by_tier: dict[str, set] = {}
    for cid, tier in enumerate(fleets[mix], start=1):
        by_tier.setdefault(tier, set()).add(round(sleeps[cid], 9))
    check(f"{mix}: one sleep value per tier", all(len(v) == 1 for v in by_tier.values()),
          str({t: len(v) for t, v in by_tier.items()}))
    per_mix_speeds[mix] = {t: next(iter(v)) for t, v in by_tier.items()}

reference = per_mix_speeds[MIXES[0]]
for mix in MIXES[1:]:
    same = all(abs(per_mix_speeds[mix][t] - reference[t]) < 1e-9 for t in reference)
    check(f"{mix}: tier speeds identical to {MIXES[0]}", same,
          " ".join(f"{t}={per_mix_speeds[mix][t]:.4f}" for t in TIER_ORDER))

# ---------------------------------------------------------------------------
print("\n4. The rendered configs describe one fleet, not two")
cfg_dir = _HERE / "configs_mix"
configs = sorted(cfg_dir.glob("*.toml")) if cfg_dir.is_dir() else []
check("configs_mix/ holds 12 configs", len(configs) == 12, f"n={len(configs)}")
ports: dict[int, str] = {}
for path in configs:
    cfg = tomllib.loads(path.read_text())
    from_clients, from_sleep = make_configs.mix_labels(cfg)
    check(f"{path.name}: clients and sleep name the same fleet",
          from_clients == from_sleep and from_clients in MIXES,
          f"{from_clients} / {from_sleep}")
    port = cfg["server"]["port"]
    check(f"{path.name}: port {port} unused elsewhere", port not in ports, ports.get(port, ""))
    ports[port] = path.name
    if "staleness_predictor" in cfg.get("server", {}):
        check(f"{path.name}: t_round_model = queue",
              cfg["gpu_tier"].get("t_round_model") == "queue",
              str(cfg["gpu_tier"].get("t_round_model")))
    check(f"{path.name}: early stop disabled",
          cfg["trainer"]["target_accuracy"] > 1.0, str(cfg["trainer"]["target_accuracy"]))

# ---------------------------------------------------------------------------
print("\n5. The gap sweep's configs are untouched by all of this")
# The mix work added three placeholders to the shared templates.  Their defaults
# have to leave the gap sweep's rendered VALUES exactly as they were, or every
# run already on disk is describing a config that no longer regenerates.
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(
        [sys.executable, str(_HERE / "make_configs.py"), "--outdir", tmp],
        check=True, capture_output=True, cwd=_HERE,
    )
    live = sorted((_HERE / "configs").glob("*.toml"))
    check("configs/ holds 15 configs", len(live) == 15, f"n={len(live)}")
    for path in live:
        regenerated = Path(tmp) / path.name
        if not regenerated.exists():
            check(f"{path.name}: regenerates", False, "missing")
            continue
        check(f"{path.name}: values unchanged",
              tomllib.loads(path.read_text()) == tomllib.loads(regenerated.read_text()))

# ---------------------------------------------------------------------------
print("\n6. A crossed mix is refused, not run")
crossed = tomllib.loads((cfg_dir / f"gpu_xa{ANCHOR}_gap{GAP}_m70_20_10.toml").read_text())
crossed["clients"]["simulation_distribution"]["csv_path"] = (
    f"/x/client_sleep_times_xa{ANCHOR}_gap{GAP}_m50_35_15.csv"
)
# Keep the real filename: validate() reads the mix label out of it, so a
# synthetic name would test a different code path than the one that runs.
name = f"gpu_xa{ANCHOR}_gap{GAP}_m70_20_10.toml"
problems = make_configs.validate({name: ("", crossed, 1)}, check_paths=False)
check("validate() catches a sleep file from another fleet", bool(problems),
      problems[0] if problems else "nothing reported")

# ---------------------------------------------------------------------------
print("\n" + "-" * 62)
if failures:
    print(f"{FAIL} {len(failures)} check(s) failed:")
    for name in failures:
        print(f"  - {name}")
    raise SystemExit(1)
print(f"{PASS} all checks passed")
