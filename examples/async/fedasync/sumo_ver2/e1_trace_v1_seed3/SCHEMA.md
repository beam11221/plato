# E1 trace bundle -- data schema

Bundle version: `e1_trace_v1`
Scope: **E1-A** -- the RSU is a *coverage predicate only*. `uplink_rate_bps` is a
reserved constant column and is not read by any logic in this stage.

Time convention: `t` in `trace.csv` is rebased so **t = 0 is the end of the
150 s warm-up**. Warm-up rows carry **t < 0**. Steady-state analysis uses t >= 0.

## trace.csv  (one row per vehicle per time step, trace_dt_s = 1.0 s)

| column | dtype | unit | meaning |
|---|---|---|---|
| `t` | float | s | sim time, rebased so t=0 = end of warm-up (t<0 = warm-up) |
| `veh_id` | str | - | SUMO vehicle id (`fEB.*` eastbound, `fWB.*` westbound) |
| `x_m` | float | m | position along the road (0 = west end, road_length = east end) |
| `lane` | int | - | lane index within the vehicle's direction (0-based) |
| `direction` | int | - | +1 eastbound, -1 westbound |
| `speed_mps` | float | m/s | instantaneous speed |
| `fl_capable` | int(0/1) | - | 1 if FL-capable; hashed from veh_id (P = p_fl_capable) |
| `tier` | str | - | compute tier `xavier`/`orin`/`thor`, or `none` if not FL-capable; hashed from veh_id |
| `in_coverage` | int(0/1) | - | 1 iff `abs(x_m - rsu_x_m) <= coverage_radius_m` (geometry only) |
| `dist_to_rsu_m` | float | m | `abs(x_m - rsu_x_m)` |
| `uplink_rate_bps` | float | bit/s | RESERVED constant; not used in E1-A |

The FL **pool** at time t is `{fl_capable==1 AND in_coverage==1}`.

## vehicles.csv  (one row per vehicle that appears in the recorded run)

| column | dtype | unit | meaning |
|---|---|---|---|
| `veh_id` | str | - | SUMO vehicle id |
| `fl_capable` | int(0/1) | - | 1 if FL-capable (hashed from veh_id) |
| `tier` | str | - | compute tier, or `none` |
| `direction` | int | - | +1 eastbound, -1 westbound |
| `first_seen_t` | float | s | first recorded t for this vehicle |
| `last_seen_t` | float | s | last recorded t for this vehicle |
| `residence_s` | float | s | total time spent with in_coverage==1 |
| `n_coverage_intervals` | int | - | number of distinct in-coverage intervals (normally 1) |
| `entered_coverage` | int(0/1) | - | 1 if the vehicle was ever in coverage |
| `t_enter_coverage_s` | float | s | first t in coverage (NaN if never) |
| `t_exit_coverage_s` | float | s | last t in coverage (NaN if never) |
