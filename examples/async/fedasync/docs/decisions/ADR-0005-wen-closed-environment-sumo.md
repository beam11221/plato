# ADR-0005: The Wen closed-population SUMO environment (imported from the SUMO side)

Status: accepted on the SUMO side, 15 Oct 2026. It was imported here on 16 Oct 2026 under the next free number.
Original filename: `docs/adr/ADR-0001-wen-closed-environment.md` in the SUMO repository. That number clashes with this folder's ADR-0001 series.

> **The original text is not in this repository yet.** The bundle that reached this machine is `sumo_ver2/bundle_wen_closed_v1/`, an extracted directory without the zip. It references the ADR in `manifest.yaml` (`adr:`), but does not ship the file.
>
> This ADR holds the number and records what the bundle itself documents. Paste the original text below this note when it arrives, and keep the note.

## What the bundle records (from `wen_closed.py`, `sim_meta.yaml`, `gate_report.md`)

**Road and fleet.**
- The road is 2,000 m long, one-way, with 2 lanes and a lane speed of 34 m/s.
- The population is closed: 30 vehicles `veh00..veh29`, and each re-enters at the road start with the same ID.

**Speeds.** Each vehicle draws one desired speed `V_k ~ U[17, 34]` m/s. The draw is `numpy.random.default_rng(20261015)`, taken in ID order. The realised range is 17.077–32.590 m/s.

**Motion is kinematic (amendment 1, approved after Krauss failed G1, G3 and G4).**
- TraCI `setSpeedMode(0)` + `setSpeed(V_k)` hold each speed constant. Vehicles do not interact, and the `lane` column is inert.
- The loop is closed by a wrap. On the step where a vehicle would pass `2000 - 0.1` m, it is moved to `x + V_k*dt - 2000`.
- That costs one 0.1 s step per lap, so the lap time is `2000/V_k - 0.1` s.
- The Krauss run is kept as evidence of platooning, which is a limitation of Wen's constant-speed assumption.

**Gate G2 (amendment 2).** The upper bound was relaxed to 32.5 m/s. The seed was not changed.

**Timing.** Step 0.1 s, a 300 s warm-up that is discarded, then 30,000 s recorded at 1 s sampling. In `trace.csv.gz`, t = 0 is the end of warm-up. `dwell_log`, `lap_log` and `reentry_log` use the SUMO clock, which includes the warm-up.

**The trace must not be looped.** A longer run needs a longer trace.

**RSU.** The RSU is not modelled in SUMO. It sits at x = 1,000 m with a 10 m lateral offset. S1 has 100% coverage; S2 covers x ∈ [400, 1600].

**Gates.** All 8 gates pass (`gate_report.md`).

## Consequences on the PLATO side (Prompt B)

- `wen_env/wen_trace.py` checks integrity: the pinned `manifest.yaml` sha256 plus every per-file sha256 it lists, and the zip's sha256 when a zip is supplied.
- Positions are interpolated within a lap only. Because of the wrap cost, the global form `(x0 + V_k t) mod 2000` drifts by about `V_k × 0.1` m per lap, up to about 1 km by t = 30,000 s, and is not used.
