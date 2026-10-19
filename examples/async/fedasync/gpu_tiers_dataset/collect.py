"""
Turn finished Stage 3 runs into the two files the brief asks for.

    results/accuracy_curves_reconstructed.csv   long format, one row per round
        experiment, arm, seed, round, sim_time_plato, sim_time_reconstructed, accuracy

    results/run_summary.csv                     one row per run

Data only. `EXPERIMENT_alpha_sweep.md`'s rule for `../separability/collect.py`
applies here too: emit data, rank nothing, explain nothing away.

Definitions
-----------
Mirrors `../separability/collect.py`'s methodology exactly where it applies,
adapted to the reconstructed clock and to this folder's five arms:

    plateau           median accuracy over the last 10% of evaluations
    noise_sd          SD of residuals around a centred 101-evaluation rolling
                      median -- the brief's own definition
    noise_sd_diff     SD(diff(accuracy))/sqrt(2), a second curvature-free
                      estimator. NOT requested by the brief; kept because
                      `../separability/collect.py` reports both and they should
                      agree -- free from the same curve, costs nothing to add.
    tta_h_<t>         simulated HOURS, from `sim_time_reconstructed`, to the
                      first evaluation of the SMOOTHED curve (11-eval rolling
                      median) reaching t%. Raw (unsmoothed) crossings are also
                      emitted as tta_h_raw_<t> -- a crossing read off one noisy
                      evaluation is what the project's own noise-floor writeup
                      (gpu_tiers/HANDOFF.md 4.13) warns against.
    rounds_to_<t>     the same crossing in ROUNDS, which carries none of the
                      clock's noise
    mean_staleness,
    max_staleness     over EVERY merge in the run, from
                      `clock_reconstruction.staleness_per_merge` -- verified
                      against PLATO's own logged `actual=` staleness for the
                      arms that log it, and the only source of the two that
                      also covers `baseline` (which never logs staleness as
                      text at all: it never touches Polaris)
    dispatches,
    dispatch_<tier>,
    share_<tier>      who actually got selected -- Table F's control check
    seconds_per_round sim_time_reconstructed at the last round / rounds completed
    fleet_believed_e_t_s,
    fleet_realised_e_t_s,
    fleet_e_t_gap_pct the belief-vs-reality gap on E[T] for the FLEET this run
                      used (`build_fleet.py`; see `stage2_evidence/
                      jitter_check_cv15.txt` section 4 for the same numbers
                      computed the same way) -- one triple per fleet, attached
                      to every run built on it

Only the brief's four thresholds -- 70 / 75 / 78 / 80 -- are computed. Unlike
`../separability/collect.py`, no extra lower thresholds are added: the brief's
own stop condition is "any cell's plateau leaves no threshold 5 points clear",
to be REPORTED, not silently patched over by inventing thresholds that were not
asked for.

`sim_time_plato` is kept in the long CSV for audit only, exactly as the brief
asks -- time-to-target is read from `sim_time_reconstructed` alone, everywhere
in this file.

Usage:
    python3 collect.py                       # every run under configs/ with results
    python3 collect.py --tag _det1
    python3 collect.py --runs gpu_a0p8_N20_C4_s4126 cpu_a0p8_N20_C4_s4126
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import tomllib
from pathlib import Path

import clock_reconstruction as cr
import make_stage2_jitter_report as jitter_report

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"

TIERS = ("xavier", "orin", "thor")
THRESHOLDS = (70, 75, 78, 80)
PLATEAU_FRACTION = 0.10
NOISE_WINDOW = 101
SMOOTH_WINDOW = 11


# ---------------------------------------------------------------------------
# The evaluation curve, from a reconstructed run's per-round rows
# ---------------------------------------------------------------------------

def evaluation_rows(result: dict) -> list[dict]:
    """Only the rounds PLATO actually evaluated, dropping the carried-forward
    repeats between them.

    An evaluation lands exactly on a multiple of `eval_interval`
    (`fedasync_server.py::_process_reports`: it skips testing, and reuses the
    last accuracy, whenever `current_round % eval_interval != 0`). Filtering on
    that is exact and needs no accuracy-value heuristics -- unlike
    `../separability/collect.py`, which cross-checks an eval-interval filter
    against a "value changed" filter because it does not have this folder's
    config already in hand at that point; here `result["config"]` always does.
    """
    eval_interval = int(result["config"]["server"].get("eval_interval", 1))
    return [row for row in result["rows"] if row["round"] % eval_interval == 0]


def rolling_median(values: list[float], window: int) -> list[float]:
    """Centred rolling median, shrinking the window at the edges.

    Shrinking rather than padding keeps the first and last evaluations usable:
    a padded edge would drag the smoothed curve toward an invented value, and
    the low-threshold crossings land exactly among the early evaluations.
    """
    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(st.median(values[lo:hi]))
    return out


def first_crossing(
    rows: list[dict], accuracies: list[float], target: float
) -> tuple[int | None, float | None]:
    """(round, reconstructed hours) of the first evaluation at or above target."""
    for row, value in zip(rows, accuracies):
        if value >= target:
            return row["round"], row["sim_time_reconstructed"] / 3600.0
    return None, None


def curve_stats(rows: list[dict]) -> dict[str, object]:
    accuracies = [row["accuracy"] for row in rows]
    if not accuracies:
        return {}

    tail = accuracies[-max(1, int(len(accuracies) * PLATEAU_FRACTION)):]
    smoothed = rolling_median(accuracies, SMOOTH_WINDOW)
    baseline = rolling_median(accuracies, NOISE_WINDOW)
    residuals = [a - b for a, b in zip(accuracies, baseline)]
    diffs = [b - a for a, b in zip(accuracies, accuracies[1:])]

    stats: dict[str, object] = {
        "evaluations": len(accuracies),
        "best_accuracy": round(max(accuracies), 6),
        "plateau_accuracy": round(st.median(tail), 6),
        "noise_sd": round(st.stdev(residuals), 6) if len(residuals) > 1 else "",
        "noise_sd_diff": round(st.stdev(diffs) / 2 ** 0.5, 6) if len(diffs) > 1 else "",
    }
    for target in THRESHOLDS:
        rounds_to, hours = first_crossing(rows, smoothed, target / 100.0)
        raw_rounds, raw_hours = first_crossing(rows, accuracies, target / 100.0)
        stats[f"tta_h_{target}"] = round(hours, 6) if hours is not None else ""
        stats[f"rounds_to_{target}"] = rounds_to if rounds_to is not None else ""
        stats[f"tta_h_raw_{target}"] = round(raw_hours, 6) if raw_hours is not None else ""
        stats[f"rounds_to_raw_{target}"] = raw_rounds if raw_rounds is not None else ""
    return stats


# ---------------------------------------------------------------------------
# Server-log statistics: who was dispatched, and how stale each report was
# ---------------------------------------------------------------------------

def client_tiers_for(config: dict) -> dict[int, str]:
    """The tier map of the FLEET this run used -- not a fixed input path.

    Unlike `../separability/collect.py`, which reads a hard-coded
    `input/clients_n{N}.csv`, this folder's fleets live at
    `fleets/<name>/clients.csv` and the name varies with N, sigma and seed.
    `[gpu_tier] clients_csv` in the run's OWN saved config always points at the
    exact file the run used, which is the one thing that cannot be stale.
    """
    path = Path(config["gpu_tier"]["clients_csv"])
    with path.open(newline="") as fh:
        return {int(r["client_id"]): r["tier_id"] for r in csv.DictReader(fh)}


def dispatch_stats(result: dict, tiers: dict[int, str]) -> dict[str, object]:
    counts = {tier: 0 for tier in TIERS}
    for _round, client_id in result["dispatches"]:
        tier = tiers.get(client_id)
        if tier:
            counts[tier] += 1
    total = sum(counts.values())
    stats: dict[str, object] = {"dispatches": total}
    for tier in TIERS:
        stats[f"dispatch_{tier}"] = counts[tier]
        stats[f"share_{tier}"] = round(counts[tier] / total, 6) if total else ""
    return stats


def staleness_stats(result: dict) -> dict[str, object]:
    """mean/max report staleness, in rounds, over every merge in the run.

    From `clock_reconstruction.staleness_per_merge`: `merge_round -
    dispatch_round`, matched the same way the reconstructed clock is. Works for
    every arm, including `baseline`, which never logs staleness as text.
    """
    values = result["staleness_per_merge"]
    if not values:
        return {"mean_staleness": "", "max_staleness": ""}
    return {
        "mean_staleness": round(st.fmean(values), 6),
        "max_staleness": max(values),
    }


# ---------------------------------------------------------------------------
# The fleet's own belief-vs-reality gap on E[T]
# ---------------------------------------------------------------------------

_FLEET_MANIFEST_CACHE: dict[Path, dict] = {}


def fleet_manifest_for(config: dict) -> dict:
    """The manifest of the fleet this run's clients actually slept on.

    Cached per process: a campaign's 23 runs reuse only two or three distinct
    fleets (N=20 and N=40 at one seed), so re-reading and re-parsing the same
    manifest.json 23 times would be silent waste, not a correctness issue --
    but the cache is keyed on the resolved path, so two runs on genuinely
    different fleets are never confused with each other.
    """
    sleep_csv = Path(config["clients"]["simulation_distribution"]["csv_path"])
    manifest_path = sleep_csv.parent / "manifest.json"
    if manifest_path not in _FLEET_MANIFEST_CACHE:
        _FLEET_MANIFEST_CACHE[manifest_path] = json.loads(manifest_path.read_text())
    return _FLEET_MANIFEST_CACHE[manifest_path]


def fleet_stats(config: dict) -> dict[str, object]:
    manifest = fleet_manifest_for(config)
    believed = jitter_report.datasheet_expected_t(manifest)
    realised = manifest["E_T_s"]
    return {
        "fleet": manifest["fleet"],
        "fleet_believed_e_t_s": round(believed, 6),
        "fleet_realised_e_t_s": round(realised, 6),
        "fleet_e_t_gap_pct": round(100 * (realised / believed - 1), 4),
    }


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_dir_for(name: str) -> Path:
    return cr.run_dir(name)


def describe(name: str) -> dict[str, object] | None:
    config_path = CONFIGS / f"{name}.toml"
    if not config_path.exists():
        print(f"  skip {name}: no config")
        return None
    try:
        directory = cr.run_dir(name)
    except SystemExit:
        print(f"  skip {name}: no run directory yet")
        return None
    if not (directory / "results").exists() or not list(
        (directory / "results").glob("*.csv")
    ):
        print(f"  skip {name}: no results CSV yet ({directory})")
        return None

    result = cr.reconstruct(directory)
    rows = evaluation_rows(result)
    if not rows:
        print(f"  skip {name}: no evaluations yet")
        return None

    config = result["config"]
    tiers = client_tiers_for(config)
    last = result["rows"][-1]

    row: dict[str, object] = {
        "experiment": rows[0].get("experiment", ""),
        "arm": rows[0].get("arm", ""),
        "seed": config["clients"]["random_seed"],
        "alpha": f"{config['data']['concentration']:g}",
        "total_clients": config["clients"]["total_clients"],
        "concurrency": config["clients"]["per_round"],
        "staleness_bound": config["server"]["staleness_bound"],
        "exclude_processing_time_from_clock":
            config["server"].get("exclude_processing_time_from_clock", False),
        "rounds_requested": config["trainer"]["rounds"],
        "rounds_completed": last["round"],
        "sim_time_plato_final_s": round(last["sim_time_plato"], 6),
        "sim_time_reconstructed_final_s": round(last["sim_time_reconstructed"], 6),
        "seconds_per_round": round(
            last["sim_time_reconstructed"] / last["round"], 6
        ) if last["round"] else "",
        "clients_verified": result["clients_verified"],
        "merge_order_ok": result["merge_order_ok"],
        "status": "done" if (directory / f"{name}.done").exists()
        or (HERE / "logs" / f"{name}.done").exists() else "incomplete",
    }
    row.update(curve_stats(rows))
    row.update(dispatch_stats(result, tiers))
    row.update(staleness_stats(result))
    row.update(fleet_stats(config))
    return row


def curve_rows(name: str) -> list[dict[str, object]]:
    """Every round of one run, both clocks -- exactly the long-CSV shape."""
    directory = cr.run_dir(name)
    result = cr.reconstruct(directory)
    return [
        {
            "experiment": row["experiment"],
            "arm": row["arm"],
            "seed": row["seed"],
            "round": row["round"],
            "sim_time_plato": round(row["sim_time_plato"], 6),
            "sim_time_reconstructed": round(row["sim_time_reconstructed"], 6),
            "accuracy": row["accuracy"],
        }
        for row in result["rows"]
    ]


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Run names (config stems). Default: every config.")
    parser.add_argument("--tag", default=None, help="Only runs whose name ends with this.")
    parser.add_argument("--outdir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    names = args.runs or sorted(p.stem for p in CONFIGS.glob("*.toml"))
    if args.tag:
        names = [n for n in names if n.endswith(args.tag)]
    elif args.runs is None:
        # make_configs.THROWAWAY_TAGS is the single registry; see the same fix
        # in preflight.py for why a second hand-maintained copy is the wrong
        # shape (it already drifted once).
        import make_configs  # noqa: PLC0415
        names = [n for n in names
                if not any(n.endswith(t) for t in make_configs.THROWAWAY_TAGS)]

    print(f"Collecting {len(names)} run(s)")
    summary = [row for row in (describe(name) for name in names) if row]
    if not summary:
        raise SystemExit("Nothing to collect yet.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    summary_path = (args.outdir / "run_summary.csv" if not args.tag
                    else args.outdir / f"run_summary{args.tag}.csv")
    columns = list(dict.fromkeys(key for row in summary for key in row))
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {summary_path} ({len(summary)} rows)")

    curves_path = (args.outdir / "accuracy_curves_reconstructed.csv" if not args.tag
                   else args.outdir / f"accuracy_curves_reconstructed{args.tag}.csv")
    all_curve_rows = [r for name in names for r in curve_rows(name)]
    with curves_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "experiment", "arm", "seed", "round",
            "sim_time_plato", "sim_time_reconstructed", "accuracy",
        ])
        writer.writeheader()
        writer.writerows(all_curve_rows)
    print(f"wrote {curves_path} ({len(all_curve_rows)} rows)")


if __name__ == "__main__":
    main()
