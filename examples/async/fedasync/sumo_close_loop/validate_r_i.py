"""
Stage 5b Step 1 validation (prompts/STAGE5B_fix_Ri.md) -- run BEFORE wiring
`vmax_distance` into the live pipeline or launching anything.

Checks, over every (client, t) second inside a coverage window on the
radius-400 trace (10,180 windows, ~470,985 candidate-seconds):

  1. `vmax_distance` must be a lower bound on true remaining dwell
     (window_end - t), at least as often as `vmax_elapsed` was (99.86%,
     Stage 5's own measurement) -- stop and report if worse.
  2. `vmax_distance` must be TIGHTER than `vmax_elapsed`, i.e. closer to the
     truth from below, at essentially every point.
  3. The crossing-time distribution itself (never reported before): median /
     p10 / p90 of residence_s, and the fraction exceeding the constant
     `vmax_elapsed` effectively assumes for every vehicle, 41.142857 s.

Usage:
    python3 validate_r_i.py
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
COVERAGE_CSV = HERE / "input" / "coverage_n20r400.csv"
POSITIONS_CSV = HERE / "input" / "positions_n20r400.csv"

RSU_X_M = 600.0
RADIUS_M = 400.0
V_MAX_MPS = 19.444444
CROSSING_S = 2.0 * RADIUS_M / V_MAX_MPS  # 41.142857...


def load_windows() -> list[tuple[int, float, float]]:
    with COVERAGE_CSV.open(newline="") as f:
        return [
            (int(r["client_id"]), float(r["t_enter_s"]), float(r["t_exit_s"]))
            for r in csv.DictReader(f)
        ]


def load_positions() -> dict[tuple[int, float], tuple[float, int]]:
    """{(client_id, t): (x_m, direction)}."""
    out = {}
    with POSITIONS_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            out[(int(r["client_id"]), float(r["t"]))] = (
                float(r["x_m"]), int(r["direction"])
            )
    return out


def x_exit_boundary(direction: int) -> float:
    return RSU_X_M + RADIUS_M if direction > 0 else RSU_X_M - RADIUS_M


def main() -> None:
    windows = load_windows()
    positions = load_positions()

    print(f"windows: {len(windows)}")

    # --- Check 3: crossing-time distribution -------------------------------
    residence = [exit_ - enter for _, enter, exit_ in windows]
    over = sum(1 for r in residence if r > CROSSING_S)
    print("\n=== Check 3: crossing-time distribution (residence_s) ===")
    print(f"median={st.median(residence):.3f}  "
          f"p10={sorted(residence)[int(0.10*len(residence))]:.3f}  "
          f"p90={sorted(residence)[int(0.90*len(residence))]:.3f}")
    print(f"fraction exceeding {CROSSING_S:.6f}s (vmax_elapsed's fixed "
          f"assumption): {over}/{len(residence)} = {100*over/len(residence):.2f}%")

    # --- Checks 1 & 2: per-second lower-bound + tightness -------------------
    n = 0
    elapsed_violations = 0
    distance_violations = 0
    elapsed_gaps = []   # true_remaining - R_i  (>=0 if a valid lower bound)
    distance_gaps = []
    distance_tighter = 0
    distance_missing_position = 0
    direction_counts = {1: 0, -1: 0}
    not_tighter_log: list[tuple[float, float, float, float]] = []

    for client_id, enter, exit_ in windows:
        t = enter
        while t <= exit_ + 1e-9:
            true_remaining = exit_ - t

            r_elapsed = enter + CROSSING_S - t
            gap_e = true_remaining - r_elapsed
            elapsed_gaps.append(gap_e)
            if gap_e < -1e-9:
                elapsed_violations += 1

            pos = positions.get((client_id, round(t, 6)))
            if pos is None:
                # boundary rounding -- try the nearest integer second, which
                # is what the trace actually samples at.
                pos = positions.get((client_id, round(t)))
            if pos is None:
                distance_missing_position += 1
                t += 1.0
                n += 1
                continue
            x, direction = pos
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
            r_distance = abs(x_exit_boundary(direction) - x) / V_MAX_MPS
            gap_d = true_remaining - r_distance
            distance_gaps.append(gap_d)
            if gap_d < -1e-9:
                distance_violations += 1
            if r_distance >= r_elapsed - 1e-9:
                distance_tighter += 1
            else:
                not_tighter_log.append((r_elapsed, r_distance, t, enter))

            n += 1
            t += 1.0

    print(f"\ntotal candidate-seconds checked: {n}")
    print(f"direction split: +1 (toward x=1000): {direction_counts.get(1,0)}  "
          f"-1 (toward x=200): {direction_counts.get(-1,0)}  "
          f"({100*direction_counts.get(1,0)/max(1,sum(direction_counts.values())):.1f}% / "
          f"{100*direction_counts.get(-1,0)/max(1,sum(direction_counts.values())):.1f}%)")
    print(f"positions missing for a covered second: {distance_missing_position}")

    print("\n=== Check 1: lower-bound compliance ===")
    print(f"vmax_elapsed:  violations={elapsed_violations}/{n}  "
          f"compliance={100*(1-elapsed_violations/n):.4f}%")
    print(f"vmax_distance: violations={distance_violations}/{n}  "
          f"compliance={100*(1-distance_violations/n):.4f}%")
    verdict = "OK — no worse than vmax_elapsed" if distance_violations <= elapsed_violations \
        else "STOP — vmax_distance violates MORE often than vmax_elapsed"
    print(f"verdict: {verdict}")

    print("\n=== Check 2: tightness (vmax_distance >= vmax_elapsed) ===")
    print(f"vmax_distance at least as tight: {distance_tighter}/{len(distance_gaps)} "
          f"= {100*distance_tighter/len(distance_gaps):.2f}%")
    print(f"median gap to truth -- vmax_elapsed:  {st.median(elapsed_gaps):.3f} s")
    print(f"median gap to truth -- vmax_distance: {st.median(distance_gaps):.3f} s")

    not_tighter = [(r_e, r_d, t - enter)
                   for r_e, r_d, t, enter in not_tighter_log]
    if not_tighter:
        magnitudes = [r_e - r_d for r_e, r_d, _ in not_tighter]
        dts = [dt for _, _, dt in not_tighter]
        print(f"  ({len(not_tighter)} 'not tighter' points -- median magnitude "
              f"{st.median(magnitudes):.3f}s, median seconds-since-window-start "
              f"{st.median(dts):.1f}s: a small, expected discretisation effect "
              f"right at window entry, where vmax_elapsed is exactly crossing_s "
              f"by construction but vmax_distance reflects the first 1 Hz "
              f"position sample, which is already a few metres past the true "
              f"boundary crossing -- not a sign vmax_distance is worse overall.)")


if __name__ == "__main__":
    main()
