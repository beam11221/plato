"""
The SUMO bundle: integrity check, per-vehicle speeds, and position lookup.

Bundle: `bundle_wen_closed_v1` (SUMO 1.21.0, kinematic motion, 30 vehicles on a
2,000 m one-way loop, 1 s samples for t in [1, 30000] s after a 300 s warm-up).

Integrity
---------
The prompt pins the sha256 of `bundle_wen_closed_v1.zip`. The bundle reached
this machine as an extracted directory, without the zip, so two checks stand in
for it:

  * `manifest.yaml` must hash to MANIFEST_SHA256 (pinned from the copy uploaded
    on 15 Oct 2026); it lists every other file's sha256,
  * every file listed there must match.

If a zip is supplied (`[wen_env] bundle_zip`), its own sha256 must equal
BUNDLE_ZIP_SHA256 as well. Any mismatch raises; there is no "warn and go on".

Position
--------
`x(veh, t)` interpolates linearly between the two 1 s samples around t, and
never across a wrap: within one lap the motion is exactly x0 + V_k*t (checked
to 1 mm), but the SUMO side wraps by moving the vehicle to x + V_k*dt - 2000 on
the step it would pass 2,000 m, which costs one 0.1 s step per lap
(`wen_closed.py`, docstring). So the GLOBAL formula (x0 + V_k*t) mod 2000 drifts
by about 0.15% of distance travelled -- up to ~1 km by t = 30,000 s -- and is
not used anywhere. Inside a bracket that contains the wrap, the position is
extrapolated kinematically from whichever sample is on the same lap as t; the
error there is at most one 0.1 s step (V_k * 0.1 <= 3.3 m), always at the far
end of the road (x < 4 m or x > 1,996 m), far from both the RSU and the S2
boundaries.

`lane` is inert under kinematic motion and is never read.

Times beyond the trace end raise TraceEndError: the SUMO ADR forbids looping
the trace, so a run that needs more time needs a longer trace, not a wrap.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

import wen_constants as K

BUNDLE_ZIP_SHA256 = "18532ec2348ae53279b0cbad4d916712c23772c0a62ed7b70bcc7cd8cb5d8a24"
MANIFEST_SHA256 = "f07a188028ab8c1775041ee30b6153449ecc3c14188c9fc9a5b2a1a59fff43d2"
BUNDLE_NAME = "bundle_wen_closed_v1"
N_VEHICLES = 30
SAMPLE_DT_S = 1.0
# SUMO wraps once x + V*dt reaches 2000 - POSITION_EPS (wen_closed.py).
WRAP_THRESHOLD_M = K.ROAD_LENGTH_M - 0.1


class BundleIntegrityError(RuntimeError):
    pass


class TraceEndError(RuntimeError):
    """A position was requested past the last trace sample. Stop condition."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_bundle(bundle_dir: Path, zip_path: Path | None = None) -> dict:
    """Refuse to continue unless every checksum matches. Returns the manifest."""
    bundle_dir = Path(bundle_dir)
    if zip_path is not None:
        got = sha256_file(zip_path)
        if got != BUNDLE_ZIP_SHA256:
            raise BundleIntegrityError(
                f"{zip_path} sha256 {got} != pinned {BUNDLE_ZIP_SHA256}"
            )

    manifest_path = bundle_dir / "manifest.yaml"
    got = sha256_file(manifest_path)
    if got != MANIFEST_SHA256:
        raise BundleIntegrityError(
            f"{manifest_path} sha256 {got} != pinned {MANIFEST_SHA256}"
        )
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest.get("bundle") != BUNDLE_NAME:
        raise BundleIntegrityError(f"manifest names bundle {manifest.get('bundle')!r}")

    bad = []
    for name, expected in manifest["sha256"].items():
        path = bundle_dir / name
        if not path.exists():
            bad.append(f"{name}: missing")
        elif sha256_file(path) != expected:
            bad.append(f"{name}: sha256 mismatch")
    if bad:
        raise BundleIntegrityError("bundle files fail verification: " + "; ".join(bad))
    logging.info("[WenTrace] %s verified: manifest + %d files", BUNDLE_NAME,
                 len(manifest["sha256"]))
    return manifest


def veh_id(client_id: int) -> str:
    """client i <-> veh{i-1:02d}: client 1 is veh00, client 30 is veh29."""
    if not 1 <= client_id <= N_VEHICLES:
        raise ValueError(f"client_id {client_id} outside 1..{N_VEHICLES}")
    return f"veh{client_id - 1:02d}"


def load_speeds(bundle_dir: Path) -> dict[str, float]:
    """V_k per veh_id, from fleet_speeds.csv (written before SUMO started)."""
    with (Path(bundle_dir) / "fleet_speeds.csv").open(newline="") as fh:
        return {r["veh_id"]: float(r["desired_speed_mps"]) for r in csv.DictReader(fh)}


@dataclass(frozen=True)
class _Track:
    t0: float               # time of sample 0
    x: np.ndarray           # positions at t0, t0 + 1, ...
    lap: np.ndarray
    speed: float            # V_k


class TraceIndex:
    """x(veh, t) and lap(veh, t) for t in [0, t_end]."""

    def __init__(self, tracks: dict[str, _Track]) -> None:
        self._tracks = tracks
        self.t_end = min(tr.t0 + (len(tr.x) - 1) * SAMPLE_DT_S for tr in tracks.values())

    @classmethod
    def from_bundle(cls, bundle_dir: Path, speeds: dict[str, float]) -> "TraceIndex":
        rows: dict[str, list[tuple[float, float, int]]] = {}
        with gzip.open(Path(bundle_dir) / "trace.csv.gz", "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.setdefault(r["veh_id"], []).append(
                    (float(r["t"]), float(r["x_m"]), int(r["lap"]))
                )
        tracks = {}
        for vid, samples in rows.items():
            samples.sort()
            t = np.array([s[0] for s in samples])
            if not np.allclose(np.diff(t), SAMPLE_DT_S):
                raise BundleIntegrityError(f"{vid}: trace is not on a 1 s grid")
            tracks[vid] = _Track(
                t0=float(t[0]),
                x=np.array([s[1] for s in samples]),
                lap=np.array([s[2] for s in samples], dtype=int),
                speed=speeds[vid],
            )
        if len(tracks) != N_VEHICLES:
            raise BundleIntegrityError(f"trace has {len(tracks)} vehicles, expected 30")
        index = cls(tracks)
        logging.info("[WenTrace] trace loaded: %d vehicles, t in [0, %.1f] s",
                     len(tracks), index.t_end)
        return index

    def _locate(self, vid: str, t: float) -> tuple[float, int]:
        """(x, lap) at time t; see the module docstring for the wrap rule."""
        if t > self.t_end + 1e-9:
            raise TraceEndError(
                f"position of {vid} needed at t = {t:.3f} s, past the trace end "
                f"({self.t_end:.0f} s). The trace must not be looped; request a "
                f"longer trace."
            )
        if t < 0.0:
            raise ValueError(f"negative time {t}")
        tr = self._tracks[vid]
        offset = (t - tr.t0) / SAMPLE_DT_S
        k = math.floor(offset)

        if k >= 0 and offset == k:                      # exactly on a sample
            return float(tr.x[k]), int(tr.lap[k])

        lo = k if k >= 0 else None                      # t < t0: no left sample
        hi = k + 1
        if lo is not None and tr.lap[lo] == tr.lap[hi]:
            frac = offset - k
            return float(tr.x[lo] + (tr.x[hi] - tr.x[lo]) * frac), int(tr.lap[lo])

        # The bracket contains a wrap, or there is no left sample: extrapolate
        # from the side that is on t's own lap.
        if lo is not None:
            t_lo = tr.t0 + lo * SAMPLE_DT_S
            forward = tr.x[lo] + tr.speed * (t - t_lo)
            if forward < WRAP_THRESHOLD_M:
                return float(forward), int(tr.lap[lo])
        t_hi = tr.t0 + hi * SAMPLE_DT_S
        backward = tr.x[hi] - tr.speed * (t_hi - t)
        if backward >= 0.0:
            return float(backward), int(tr.lap[hi])
        # Only reachable for t < t0 on a vehicle that wrapped inside [t, t0].
        return float(backward + K.ROAD_LENGTH_M), int(tr.lap[hi]) - 1

    def x_at(self, vid: str, t: float) -> float:
        return self._locate(vid, t)[0]

    def lap_at(self, vid: str, t: float) -> int:
        return self._locate(vid, t)[1]

    def stays_inside(self, vid: str, t_a: float, t_b: float,
                     lo: float, hi: float) -> bool:
        """Whether the vehicle is inside [lo, hi] for the whole interval.

        Motion is monotone within a lap, so the end points decide it, provided
        no wrap happens in between (a wrap would take it through x = 2000 and
        x = 0, both outside any coverage zone here).
        """
        xa, lap_a = self._locate(vid, t_a)
        xb, lap_b = self._locate(vid, t_b)
        return lap_a == lap_b and lo <= xa <= hi and lo <= xb <= hi

    def samples(self, vid: str) -> tuple[float, np.ndarray, np.ndarray]:
        tr = self._tracks[vid]
        return tr.t0, tr.x, tr.lap
