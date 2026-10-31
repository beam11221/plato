"""
Prompt M Part B, B0.1: one fleet directory per tier mix, derived from M0.

Only the tier label of each client id changes (contiguous blocks Xavier ->
Orin -> Thor, wen_constants.FLEET_MIXES). The persistent jitter eps_i of each id
is copied from M0, so every id keeps its own eps_i; vehicle ids, speeds and
partitions do not live in the fleet and are untouched. `sleep_table.csv` /
`sleep_times.csv` are rewritten with the new tier's T_tier for the record --
the server never reads that column (wen_fleet.py).

Writes `evidence/tier_sweep_fleets.md`: per-id tier, per-tier count and mean
speed, and each id's partition fingerprint against M0.

Usage (from examples/async/fedasync):
    python wen_env/wen_fleet_mixes.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parent
sys.path[:0] = [str(HERE), str(FEDASYNC / "sumo_close_loop")]

import wen_constants as K  # noqa: E402
import wen_partition  # noqa: E402
from wen_faithful import cifar10_train_targets  # noqa: E402
from wen_fleet import (  # noqa: E402
    TIER_ORDER,
    Fleet,
    contiguous_tiers,
    mix_counts,
    mix_fleet_dir,
)
from wen_trace import load_speeds, veh_id  # noqa: E402

FLEETS = FEDASYNC / "gpu_tiers_dataset" / "fleets"
TIERS = FEDASYNC / "gpu_tiers_dataset" / "tiers" / "tiers_wen.csv"
BUNDLE = FEDASYNC / "sumo_ver2" / "bundle_wen_closed_v1"
DATA = FEDASYNC / "data"
DATA_SEED = 4126


def fleet_files_sha256(fleet_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ("clients.csv", "sleep_table.csv", "manifest.json"):
        h.update(name.encode() + b"\0" + (Path(fleet_dir) / name).read_bytes())
    return h.hexdigest()


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def build(mix: str) -> Path:
    m0 = mix_fleet_dir(FLEETS, "M0")
    out = mix_fleet_dir(FLEETS, mix)
    if mix == "M0":
        return out
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((m0 / "manifest.json").read_text())
    t_tier = manifest["T_tier_s"]
    tiers = contiguous_tiers(mix)
    eps = {int(r["client_id"]): r["eps"] for r in _read(m0 / "sleep_table.csv")}

    with (out / "clients.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["client_id", "tier_id", "schema_version"])
        for cid, tier in enumerate(tiers, start=1):
            w.writerow([cid, tier, 1])
    sleeps = {cid: t_tier[tier] * math.exp(float(eps[cid]))
              for cid, tier in enumerate(tiers, start=1)}
    with (out / "sleep_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["client_id", "tier", "T_tier_s", "eps", "sleep_s"])
        for cid, tier in enumerate(tiers, start=1):
            w.writerow([cid, tier, f"{t_tier[tier]:.9f}", eps[cid], f"{sleeps[cid]:.9f}"])
    with (out / "sleep_times.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["client_id", "sleep_time"])
        for cid in sorted(sleeps):
            w.writerow([cid, f"{sleeps[cid]:.9f}"])
    counts = mix_counts(mix)
    manifest.update(
        mix_id=mix, derived_from=m0.name, fleet=out.name,
        counts=[counts[t] for t in TIER_ORDER],
        mix=[round(100 * counts[t] / K.N_CLIENTS, 4) for t in TIER_ORDER],
        E_T_s=sum(sleeps.values()) / len(sleeps),
        note="Prompt M tier-mix sweep: M0 with the tier labels re-blocked; eps_i per id "
             "copied from M0.")
    manifest.pop("tier_stats", None)
    manifest.pop("tier_overlaps", None)
    manifest.pop("built_at", None)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> int:
    speeds = load_speeds(BUNDLE)
    part = wen_partition.build(cifar10_train_targets(DATA), DATA_SEED)
    m0_eps = {int(r["client_id"]): r["eps"]
              for r in _read(mix_fleet_dir(FLEETS, "M0") / "sleep_table.csv")}
    m0_hash = {cid: part.fingerprint(cid) for cid in part.indices}
    lines = ["# Tier-mix fleets (Prompt M B0.1)", "",
             "Each mix is M0 with only the tier labels re-blocked (contiguous Xavier → "
             "Orin → Thor). Speeds come from the SUMO bundle by id; partitions from "
             "`wen_partition.build(targets, 4126)`, which does not depend on the fleet.", "",
             "| mix | X / O / T | fleet dir | fleet sha256 (12) | mean speed X / O / T (m/s) "
             "| eps_i = M0 (30 ids) | partition = M0 (30 ids) |", "|---|---|---|---|---|---|---|"]
    per_id = []
    problems = []
    for mix in K.FLEET_MIXES:
        d = build(mix)
        fleet = Fleet.load(d, TIERS, mix=mix)
        eps = {int(r["client_id"]): r["eps"] for r in _read(d / "sleep_table.csv")}
        eps_ok = eps == m0_eps
        # The partition is rebuilt from the same seed for every mix; the check is
        # that nothing about the mix enters it.
        part_mix = wen_partition.build(cifar10_train_targets(DATA), DATA_SEED)
        part_ok = all(part_mix.fingerprint(c) == m0_hash[c] for c in m0_hash)
        if not (eps_ok and part_ok):
            problems.append(mix)
        speed = {}
        for tier in TIER_ORDER:
            ids = [c for c in fleet.client_ids if fleet.tier(c) == tier]
            speed[tier] = (f"{sum(speeds[veh_id(c)] for c in ids) / len(ids):.2f}"
                           if ids else "—")
        x, o, t = K.FLEET_MIXES[mix]
        lines.append(f"| {mix} | {x} / {o} / {t} | `{d.name}` | "
                     f"{fleet_files_sha256(d)[:12]} | {speed['xavier']} / {speed['orin']} / "
                     f"{speed['thor']} | {'yes' if eps_ok else 'NO'} | "
                     f"{'yes' if part_ok else 'NO'} |")
        per_id.append((mix, [fleet.tier(c)[0].upper() for c in fleet.client_ids]))
    lines += ["", "## Tier per client id (X = Xavier, O = Orin, T = Thor)", "",
              "| mix | " + " | ".join(str(c) for c in range(1, K.N_CLIENTS + 1)) + " |",
              "|---|" + "---|" * K.N_CLIENTS]
    lines += [f"| {mix} | " + " | ".join(row) + " |" for mix, row in per_id]
    lines += ["", "## Partition fingerprint per id (sha256 of the sorted indices, 12 hex)", "",
              "| id | size | fingerprint |", "|---|---|---|"]
    lines += [f"| {c} | {len(part.indices[c])} | {m0_hash[c]} |" for c in sorted(m0_hash)]
    lines += ["", f"Check: {'PASS' if not problems else 'FAIL ' + ', '.join(problems)}.", ""]
    out = HERE / "evidence" / "tier_sweep_fleets.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines[:14]))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
