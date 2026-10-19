#!/usr/bin/env python3
"""wen_closed.py -- Wen-aligned closed-population mobility trace (SUMO + TraCI).

Spec: Wen et al., Information Sciences 664 (2024), S5.3.4 highway setting.
  * one-way straight road, 2000 m, 2 lanes, lane speed 34 m/s
  * 30 vehicles veh00..veh29, closed population (re-entry keeps the same ID)
  * one desired speed per vehicle, v ~ U[17, 34] m/s, drawn once, constant
  * sigma = 0, speedDev = 0
  * step-length 0.1 s, 300 s warm-up discarded, 30000 s recorded, 1.0 s sampling

Closed loop (--loop):
  wrap (default in kinematic mode) -- the vehicle never leaves the network: on the
      step where it would pass x = 2000 m, TraCI moveTo() puts it back at
      x = x + V_k*dt - 2000 on the same lane, same ID, same vType. Re-entry is
      instantaneous by construction. Needed because SUMO's insertion gap check
      refuses a remove-and-reinsert at x = 0 with departSpeed="desired" whenever
      the leader is slower (a 54 m gap is not enough at 32 m/s), which pushed the
      G3 re-entry-delay p95 to 3.8 s. Costs one step of travel per lap, i.e. lap
      time is 2000/V_k - 0.1 s (about -0.15%, well inside the G4 +-3% band).
  reinsert (default in krauss mode) -- spec-literal: wait for the arrival, then
      re-add the same ID at departPos 0 on the lane with the larger gap, retrying
      every step, and log the resulting delay.

Motion model (--model, approved 2026-10-15):
  kinematic (default) -- each vehicle holds its constant V_k via TraCI
      setSpeedMode(0) + setSpeed(V_k), i.e. vehicles do not yield to each other.
      This is Wen's own model (Fig. 2: distance = T_com * V_k).
  krauss              -- SUMO default Krauss car-following + LC2013. Faithful to
      the microsimulation, but on a 2 km / 2-lane closed loop 30 vehicles with
      speeds 17-34 m/s bunch into a standing platoon behind the slowest ones
      (gate rows G1/G3/G4 fail by ~-34%/18.6 s/+53%). Kept for the record.
  * no RSU in SUMO -- coverage is applied downstream from x_m

Outputs (in --out):
  wen.nod.xml wen.edg.xml wen.net.xml wen.rou.xml wen.sumocfg
  fleet_speeds.csv  veh_id, desired_speed_mps          (written BEFORE SUMO starts)
  trace.csv.gz      t, veh_id, x_m, lane, speed_mps, lap
  reentry_log.csv   veh_id, lap, t_arrive, t_depart, delay_s, lane, phase
  lap_log.csv       veh_id, lap, t_depart, t_arrive, x0_m, dist_m, travel_s
  dwell_log.csv     veh_id, lap, t_enter, t_exit, dwell_s     (x in [400,1600])
  lane_stats.csv    veh_id, time_lane0_s, time_lane1_s, lane_changes, laps_completed
  sim_meta.yaml

Usage
    python wen_closed.py --out wen_closed_v1_run
    python wen_closed.py --out /tmp/smoke --duration 600 --warmup 60
"""
from __future__ import annotations

import argparse
import csv
import gzip
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

try:
    import traci
    import traci.constants as tc
    import sumolib
except ImportError as exc:  # pragma: no cover
    raise ImportError("traci/sumolib not importable; use the conda python with SUMO tools") from exc

# ---------------------------------------------------------------- locked spec
ROAD_LENGTH_M = 2000.0
N_LANES = 2
LANE_SPEED_MPS = 34.0
N_VEH = 30
V_MIN_MPS, V_MAX_MPS = 17.0, 34.0
SPEED_SEED = 20261015
STEP_LENGTH_S = 0.1
SAMPLE_INTERVAL_S = 1.0
WARMUP_S = 300.0
DURATION_S = 30000.0
# coverage window used for the validation gate only (S2)
S2_LO_M, S2_HI_M = 400.0, 1600.0
# vType geometry: SUMO defaults for the default passenger vehicle
VEH_LENGTH_M = 5.0
VEH_MIN_GAP_M = 2.5

ROUTE_ID = "r0"
MODEL_KINEMATIC = "kinematic"
LOOP_WRAP, LOOP_REINSERT = "wrap", "reinsert"
# SUMO retires a vehicle once its front is within POSITION_EPS of the lane end,
# so the wrap has to fire one step before that, not before x = 2000 exactly.
POSITION_EPS_M = 0.1
EDGE_ID = "E0"

_NODES_XML = """<nodes>
    <node id="N0" x="0.00" y="0.00"/>
    <node id="N1" x="{L:.2f}" y="0.00"/>
</nodes>
"""

_EDGES_XML = """<edges>
    <edge id="{e}" from="N0" to="N1" numLanes="{n}" speed="{v:.2f}"/>
</edges>
"""

_SUMOCFG_XML = """<configuration>
    <input>
        <net-file value="wen.net.xml"/>
        <route-files value="wen.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{end:.1f}"/>
        <step-length value="{step}"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="{collision}"/>
        <default.speeddev value="0"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <duration-log.statistics value="false"/>
    </report>
</configuration>
"""


def veh_id(i: int) -> str:
    return f"veh{i:02d}"


def vtype_id(i: int) -> str:
    return f"vt{i:02d}"


def draw_speeds(seed: int = SPEED_SEED) -> np.ndarray:
    """One desired speed per vehicle, drawn once, tied to the vehicle ID."""
    rng = np.random.default_rng(seed)
    return rng.uniform(V_MIN_MPS, V_MAX_MPS, size=N_VEH)


def build_inputs(out: Path, speeds: np.ndarray, model: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    nod, edg, net, rou, cfg = (out / f"wen.{s}" for s in
                               ("nod.xml", "edg.xml", "net.xml", "rou.xml", "sumocfg"))
    nod.write_text(_NODES_XML.format(L=ROAD_LENGTH_M))
    edg.write_text(_EDGES_XML.format(e=EDGE_ID, n=N_LANES, v=LANE_SPEED_MPS))

    netconvert = sumolib.checkBinary("netconvert")
    subprocess.run(
        [netconvert,
         "--node-files", str(nod),
         "--edge-files", str(edg),
         "--output-file", str(net),
         "--no-turnarounds", "true",
         "--offset.disable-normalization", "true",
         "--no-warnings", "true"],
        check=True,
    )

    # one vType per vehicle: speedFactor = v_i / lane speed, no driver noise
    checks = ' insertionChecks="none"' if model == MODEL_KINEMATIC else ""
    lines = ["<routes>"]
    for i, v in enumerate(speeds):
        lines.append(
            f'    <vType id="{vtype_id(i)}" vClass="passenger" maxSpeed="{V_MAX_MPS:.2f}"'
            f' speedFactor="{v / LANE_SPEED_MPS:.9f}" speedDev="0" sigma="0"'
            f' length="{VEH_LENGTH_M}" minGap="{VEH_MIN_GAP_M}"'
            f' carFollowModel="Krauss" laneChangeModel="LC2013"{checks}/>'
        )
    lines.append(f'    <route id="{ROUTE_ID}" edges="{EDGE_ID}"/>')
    # initial placement: evenly spread, alternating lanes, at the desired speed
    spacing = ROAD_LENGTH_M / N_VEH
    for i in range(N_VEH):
        lines.append(
            f'    <vehicle id="{veh_id(i)}" type="{vtype_id(i)}" route="{ROUTE_ID}" depart="0.00"'
            f' departPos="{i * spacing:.2f}" departLane="{i % N_LANES}" departSpeed="desired"/>'
        )
    lines.append("</routes>")
    rou.write_text("\n".join(lines) + "\n")

    cfg.write_text(_SUMOCFG_XML.format(
        end=WARMUP_S + DURATION_S, step=STEP_LENGTH_S,
        collision="none" if model == MODEL_KINEMATIC else "warn"))


# ------------------------------------------------------------------ closed loop
def _lane_front_gap(by_lane: dict[int, list[float]], lane: int) -> float:
    """Free distance ahead of an insertion at x=0 on `lane` (inf if lane empty)."""
    xs = by_lane.get(lane)
    return float("inf") if not xs else min(xs) - VEH_LENGTH_M


def run(out: Path, duration: float, warmup: float, gui: bool = False,
        model: str = MODEL_KINEMATIC, loop: str | None = None) -> dict:
    speeds = draw_speeds()
    if loop is None:
        loop = LOOP_WRAP if model == MODEL_KINEMATIC else LOOP_REINSERT
    build_inputs(out, speeds, model)

    # fleet_speeds.csv is written before SUMO starts (spec step 1.2)
    with (out / "fleet_speeds.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["veh_id", "desired_speed_mps"])
        for i, v in enumerate(speeds):
            w.writerow([veh_id(i), f"{v:.9f}"])

    v_of = {veh_id(i): float(speeds[i]) for i in range(N_VEH)}
    t_of = {veh_id(i): vtype_id(i) for i in range(N_VEH)}

    binary = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    traci.start([binary, "-c", str(out / "wen.sumocfg"), "--step-length", str(STEP_LENGTH_S)])

    sub_vars = (tc.VAR_LANEPOSITION, tc.VAR_LANE_INDEX, tc.VAR_SPEED)
    kinematic = model == MODEL_KINEMATIC
    wrapping = loop == LOOP_WRAP
    lane_ids = [f"{EDGE_ID}_{ln}" for ln in range(N_LANES)]

    def on_insert(vid: str) -> None:
        """Subscribe, and in kinematic mode pin the vehicle to its constant V_k."""
        traci.vehicle.subscribe(vid, sub_vars)
        if kinematic:
            traci.vehicle.setSpeedMode(vid, 0)   # ignore all safety constraints
            traci.vehicle.setSpeed(vid, v_of[vid])

    for vid in traci.vehicle.getIDList():
        on_insert(vid)

    lap = {vid: 0 for vid in v_of}
    pending: dict[str, float] = {}          # vid -> t_arrive  (awaiting re-insertion)
    added: dict[str, tuple[float, int]] = {}  # vid -> (t_arrive, lane) handed to SUMO
    prev_x: dict[str, float] = {}
    prev_lane: dict[str, int] = {}
    depart_t: dict[str, float] = {}         # vid -> t it entered at x=0 (this pass)
    depart_x: dict[str, float] = {}         # vid -> x right after that insertion
    cov_enter: dict[str, float] = {}        # vid -> t of crossing x=400

    reentries: list[tuple] = []
    laps: list[tuple] = []
    dwells: list[tuple] = []
    lane_time = {vid: [0.0] * N_LANES for vid in v_of}
    lane_changes = {vid: 0 for vid in v_of}
    laps_done = {vid: 0 for vid in v_of}
    missing_samples = 0
    present_hist: list[int] = []

    gz = gzip.open(out / "trace.csv.gz", "wt", newline="", compresslevel=6)
    trace = csv.writer(gz)
    trace.writerow(["t", "veh_id", "x_m", "lane", "speed_mps", "lap"])

    step_dt = STEP_LENGTH_S
    n_steps = int(round((warmup + duration) / step_dt))
    sample_every = int(round(SAMPLE_INTERVAL_S / step_dt))
    warm_steps = int(round(warmup / step_dt))
    t0 = time.time()

    for step in range(1, n_steps + 1):
        traci.simulationStep()
        t = step * step_dt
        post = step > warm_steps
        t_out = t - warmup

        res = traci.vehicle.getAllSubscriptionResults()
        for vid in traci.simulation.getDepartedIDList():
            on_insert(vid)
            if vid in added:
                t_arrive, lane = added.pop(vid)
                lap[vid] += 1
                reentries.append((vid, lap[vid], round(t_arrive, 3), round(t, 3),
                                  round(t - t_arrive, 3), lane, "post" if post else "warmup"))
                cov_enter.pop(vid, None)
                prev_x.pop(vid, None)
                prev_lane.pop(vid, None)
            depart_t[vid] = t
            depart_x[vid] = traci.vehicle.getLanePosition(vid)

        by_lane: dict[int, list[float]] = {}
        for vid, d in res.items():
            by_lane.setdefault(d[tc.VAR_LANE_INDEX], []).append(d[tc.VAR_LANEPOSITION])

        # per-step bookkeeping: lane dwell, lane changes, coverage crossings
        for vid, d in res.items():
            x = d[tc.VAR_LANEPOSITION]
            ln = d[tc.VAR_LANE_INDEX]
            if post:
                lane_time[vid][ln] += step_dt
                px = prev_x.get(vid)
                if px is not None:
                    if px < S2_LO_M <= x:
                        cov_enter[vid] = t
                    if px < S2_HI_M <= x and vid in cov_enter:
                        te = cov_enter.pop(vid)
                        dwells.append((vid, lap[vid], round(te, 3), round(t, 3), round(t - te, 3)))
                    if prev_lane.get(vid) is not None and prev_lane[vid] != ln:
                        lane_changes[vid] += 1
            prev_x[vid] = x
            prev_lane[vid] = ln

        # arrivals -> pending re-entry (same ID, same vType)
        for vid in traci.simulation.getArrivedIDList():
            if post and vid in depart_t and depart_t[vid] > warmup:
                dist = ROAD_LENGTH_M - depart_x[vid]
                laps.append((vid, lap[vid], round(depart_t[vid], 3), round(t, 3),
                             round(depart_x[vid], 3), round(dist, 3),
                             round(t - depart_t[vid], 3)))
                laps_done[vid] += 1
            pending[vid] = t
            prev_x.pop(vid, None)
            prev_lane.pop(vid, None)
            cov_enter.pop(vid, None)

        # re-insert: lane with the larger gap, retried every step until it fits
        for vid in sorted(pending, key=lambda k: pending[k]):
            gaps = [_lane_front_gap(by_lane, ln) for ln in range(N_LANES)]
            lane = int(np.argmax(gaps))
            traci.vehicle.add(vid, ROUTE_ID, typeID=t_of[vid], depart="now",
                              departLane=str(lane), departPos="0", departSpeed="desired")
            added[vid] = (pending.pop(vid), lane)

        # sample the trace every 1.0 s of simulated time, after warm-up
        if post and step % sample_every == 0:
            ts = round(t_out, 3)
            present_hist.append(len(res))
            missing_samples += N_VEH - len(res)
            for vid, d in res.items():
                trace.writerow([f"{ts:.1f}", vid, f"{d[tc.VAR_LANEPOSITION]:.3f}",
                                d[tc.VAR_LANE_INDEX], f"{d[tc.VAR_SPEED]:.4f}", lap[vid]])

        if wrapping:
            for vid, d in list(res.items()):
                v = v_of[vid]
                x = d[tc.VAR_LANEPOSITION]
                if x + v * step_dt < ROAD_LENGTH_M - POSITION_EPS_M:
                    continue
                ln = d[tc.VAR_LANE_INDEX]
                p_new = max(0.0, x + v * step_dt - ROAD_LENGTH_M)
                traci.vehicle.moveTo(vid, lane_ids[ln], p_new)
                traci.vehicle.setSpeed(vid, v)
                lap[vid] += 1
                reentries.append((vid, lap[vid], round(t, 3), round(t, 3), 0.0, ln,
                                  "post" if post else "warmup"))
                if post and vid in depart_t and depart_t[vid] > warmup:
                    laps.append((vid, lap[vid], round(depart_t[vid], 3), round(t, 3),
                                 round(depart_x[vid], 3), ROAD_LENGTH_M,
                                 round(t - depart_t[vid], 3)))
                    laps_done[vid] += 1
                depart_t[vid] = t
                depart_x[vid] = p_new
                prev_x[vid] = p_new
                prev_lane[vid] = ln
                cov_enter.pop(vid, None)

        if step % 200000 == 0 or step == n_steps:
            print(f"  step {step}/{n_steps}  t={t:.0f}s  elapsed={time.time()-t0:.0f}s",
                  flush=True)

    sumo_v = traci.getVersion()[1]
    traci.close()
    gz.close()

    def dump(name, header, rows):
        with (out / name).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    dump("reentry_log.csv",
         ["veh_id", "lap", "t_arrive", "t_depart", "delay_s", "lane", "phase"], reentries)
    dump("lap_log.csv",
         ["veh_id", "lap", "t_depart", "t_arrive", "x0_m", "dist_m", "travel_s"], laps)
    dump("dwell_log.csv", ["veh_id", "lap", "t_enter", "t_exit", "dwell_s"], dwells)
    dump("lane_stats.csv",
         ["veh_id", "time_lane0_s", "time_lane1_s", "lane_changes", "laps_completed"],
         [(vid, f"{lane_time[vid][0]:.1f}", f"{lane_time[vid][1]:.1f}",
           lane_changes[vid], laps_done[vid]) for vid in sorted(v_of)])

    meta = {
        "sumo_version": sumo_v,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "speed_rng": "numpy.random.default_rng (PCG64)",
        "speed_rng_seed": SPEED_SEED,
        "road_length_m": ROAD_LENGTH_M,
        "lanes": N_LANES,
        "lane_speed_mps": LANE_SPEED_MPS,
        "n_vehicles": N_VEH,
        "desired_speed_dist": f"U[{V_MIN_MPS}, {V_MAX_MPS}] m/s, one draw per vehicle ID",
        "sigma": 0,
        "speed_dev": 0,
        "step_length_s": STEP_LENGTH_S,
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "warmup_s": warmup,
        "duration_s": duration,
        "veh_length_m": VEH_LENGTH_M,
        "veh_min_gap_m": VEH_MIN_GAP_M,
        "motion_model": model,
        "closed_loop": loop,
        "car_follow_model": ("bypassed: TraCI setSpeedMode(0) + setSpeed(V_k), constant speed"
                             if kinematic else "Krauss (SUMO default)"),
        "lane_change_model": "LC2013 (SUMO default)",
        "insertion_checks": "none" if kinematic else "SUMO default",
        "s2_coverage_window_m": [S2_LO_M, S2_HI_M],
        "rsu_modelled_in_sumo": False,
        "lap_counting": "cumulative re-entries since simulation start (warm-up included)",
        "trace_time_base": "t = 0 at end of warm-up",
        "n_reentries_total": len(reentries),
        "missing_vehicle_samples": missing_samples,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (out / "sim_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(yaml.safe_dump(meta, sort_keys=False))
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--duration", type=float, default=DURATION_S)
    p.add_argument("--warmup", type=float, default=WARMUP_S)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--model", choices=[MODEL_KINEMATIC, "krauss"], default=MODEL_KINEMATIC)
    p.add_argument("--loop", choices=[LOOP_WRAP, LOOP_REINSERT], default=None,
                   help="default: wrap for kinematic, reinsert for krauss")
    a = p.parse_args(argv)
    run(a.out, a.duration, a.warmup, a.gui, a.model, a.loop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
