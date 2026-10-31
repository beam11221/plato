"""
The fleet: client id -> veh_id, tier, partition size, tier spec, and jitter.

Two sources, both checked against the locked values on load:

  * `gpu_tiers_dataset/tiers/tiers_wen.csv` -- tier specs (FP16 TFLOPS, CPU GHz).
  * a fleet directory from `gpu_tiers_dataset/build_fleet.py` -- `clients.csv`
    (client_id -> tier) and `sleep_table.csv` (the persistent eps_i per
    client). Only the tier and eps_i are used from it: the training time itself
    is recomputed from the measured FLOPs and each client's own |S_i|
    (`wen_truth.py`), so the fleet's sleep column is not read.

Jitter (eps_i) is ground truth. `Fleet.jitter()` exists for `wen_truth.py` and
for the `gpu_oracle` arm's explicitly-injected callable; nothing reachable from
a selector calls it (tests/test_wen_separation.py).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import wen_constants as K
from wen_trace import veh_id

TIER_ORDER = ("xavier", "orin", "thor")


def mix_counts(mix: str) -> dict[str, int]:
    """{tier: count} of a Prompt M tier mix (M0 = the locked fleet)."""
    if mix not in K.FLEET_MIXES:
        raise ValueError(f"[WenFleet] fleet_mix {mix!r} not in {sorted(K.FLEET_MIXES)}")
    return dict(zip(TIER_ORDER, K.FLEET_MIXES[mix]))


def contiguous_tiers(mix: str) -> list[str]:
    """Tier of client ids 1..30 in order: contiguous blocks Xavier -> Orin -> Thor."""
    return [t for t, n in mix_counts(mix).items() for _ in range(n)]


def mix_fleet_dir(fleets_root: Path, mix: str) -> Path:
    """M0 is the locked fleet directory; every other mix is derived from it."""
    if mix == "M0":
        return Path(fleets_root) / M0_FLEET
    x, o, t = K.FLEET_MIXES[mix]
    return Path(fleets_root) / f"n30_mix{mix}_x{x}-o{o}-t{t}_phys_p1667_j0p149_s4126"


M0_FLEET = "n30_x15-o10-t5_phys_p1667_j0p149_s4126"


@dataclass(frozen=True)
class TierSpec:
    tier_id: str
    tflops_fp16: float
    cpu_ghz: float


def load_tier_specs(path: Path) -> dict[str, TierSpec]:
    with Path(path).open(newline="") as fh:
        specs = {
            r["tier_id"]: TierSpec(r["tier_id"], float(r["gpu_tflops_fp16"]),
                                   float(r["cpu_freq_ghz"]))
            for r in csv.DictReader(fh)
        }
    problems = []
    for tier in TIER_ORDER:
        spec = specs.get(tier)
        if spec is None:
            problems.append(f"{tier} missing")
            continue
        if spec.tflops_fp16 != K.TIER_TFLOPS_FP16[tier]:
            problems.append(f"{tier} TFLOPS {spec.tflops_fp16} != locked "
                            f"{K.TIER_TFLOPS_FP16[tier]}")
        if spec.cpu_ghz != K.TIER_CPU_GHZ[tier]:
            problems.append(f"{tier} CPU {spec.cpu_ghz} GHz != locked "
                            f"{K.TIER_CPU_GHZ[tier]}")
    if problems:
        raise ValueError(f"[WenFleet] tier file {path} disagrees with the locked "
                         f"values: " + "; ".join(problems))
    return specs


class Fleet:
    """N = 30 clients, tiers in contiguous ID blocks, one eps_i each."""

    def __init__(self, tiers: dict[int, str], eps: dict[int, float],
                 specs: dict[str, TierSpec], manifest: dict) -> None:
        self._tiers = tiers
        self._eps = eps
        self.specs = specs
        self.manifest = manifest

    @classmethod
    def load(cls, fleet_dir: Path, tiers_csv: Path, mix: str = "M0") -> "Fleet":
        fleet_dir = Path(fleet_dir)
        manifest = json.loads((fleet_dir / "manifest.json").read_text())
        with (fleet_dir / "clients.csv").open(newline="") as fh:
            tiers = {int(r["client_id"]): r["tier_id"] for r in csv.DictReader(fh)}
        with (fleet_dir / "sleep_table.csv").open(newline="") as fh:
            eps = {int(r["client_id"]): float(r["eps"]) for r in csv.DictReader(fh)}

        problems = []
        if sorted(tiers) != list(range(1, K.N_CLIENTS + 1)):
            problems.append(f"client ids are not 1..{K.N_CLIENTS}")
        counts = {t: sum(1 for v in tiers.values() if v == t) for t in TIER_ORDER}
        expected = mix_counts(mix)
        if counts != expected:
            problems.append(f"tier counts {counts} != mix {mix} {expected}")
        if [tiers[c] for c in sorted(tiers)] != contiguous_tiers(mix):
            problems.append(f"tiers are not contiguous Xavier -> Orin -> Thor blocks ({mix})")
        if manifest.get("mix_id", "M0") != mix:
            problems.append(f"fleet manifest mix_id {manifest.get('mix_id', 'M0')!r} != {mix!r}")
        if not math.isclose(float(manifest["jitter_sigma"]), K.JITTER_SIGMA,
                            abs_tol=5e-7):
            problems.append(f"jitter_sigma {manifest['jitter_sigma']} != locked "
                            f"{K.JITTER_SIGMA:.6f}")
        if manifest.get("anchor_tier") is not None:
            problems.append("fleet is anchored; the Wen environment has no anchor")
        if sorted(eps) != sorted(tiers):
            problems.append("sleep_table.csv and clients.csv list different clients")
        if problems:
            raise ValueError(f"[WenFleet] {fleet_dir} is not the locked fleet: "
                             + "; ".join(problems))
        return cls(tiers, eps, load_tier_specs(tiers_csv), manifest)

    @property
    def client_ids(self) -> list[int]:
        return sorted(self._tiers)

    def tier(self, client_id: int) -> str:
        return self._tiers[client_id]

    def spec(self, client_id: int) -> TierSpec:
        return self.specs[self._tiers[client_id]]

    def veh_id(self, client_id: int) -> str:
        return veh_id(client_id)

    def jitter(self, client_id: int) -> float:
        """exp(eps_i). GROUND TRUTH -- simulator and the oracle arm only."""
        return math.exp(self._eps[client_id])
