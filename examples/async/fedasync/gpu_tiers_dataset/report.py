"""
Stage 3 reporting: Tables A-F, the six written-down predictions, and the four
stop conditions -- read entirely from `results/run_summary.csv`
(`collect.py`'s output). No other source of numbers.

Data only, per the brief. Where the brief itself gives an explicit numeric rule
(">10%", "5 points below"), this module applies exactly that rule and states the
result. Where the brief's own wording is qualitative ("a null", "if they tie",
"the same tier mix"), this module prints the NUMBERS the call would be made
from and does not invent a threshold to decide it -- that decision belongs to
whoever reads the table, not to this script. Which is which is called out at
each prediction/condition below.

Cell identity
-------------
`collect.py` sets `experiment = make_configs.cell_of(run)`, e.g.
"a0p8_N20_C4_s4126" -- shared by every arm run on that cell, distinct from
`arm`. The one exception is `_rep2`: its `experiment` carries the `_rep2` suffix
(`run_name()` appends the tag after the seed), which is exactly what keeps it
OUT of the ordinary C-sweep grouping in Tables B/D/F and routes it to Table A
alone, where it belongs.

Usage:
    python3 report.py                                  # results/run_summary.csv
    python3 report.py --summary results/run_summary_det3.csv --out /dev/stdout
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EVIDENCE = HERE / "stage3_evidence"

THRESHOLDS = (70, 75, 78, 80)
USABLE_MARGIN = 0.05          # "at least 5 points below the lowest plateau"
REP2_STOP_PCT = 10.0          # brief's own explicit stop-condition number


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_NUMERIC_SUFFIXES = (
    "seed", "total_clients", "concurrency", "staleness_bound",
    "rounds_requested", "rounds_completed", "evaluations", "dispatches",
    "max_staleness",
)


def _coerce(key: str, value: str) -> object:
    """Numeric where it clearly is, blank stays blank, everything else stays a
    string. `csv.DictReader` hands back strings for everything; the tables
    below do arithmetic on plateaus, staleness, TTA and shares, all of which
    must survive round-tripping through the CSV as real numbers, not text."""
    if value == "":
        return ""
    if key in _NUMERIC_SUFFIXES or key in ("total_clients", "concurrency"):
        try:
            return int(value)
        except ValueError:
            pass
    if key in ("clients_verified",):
        try:
            return int(value)
        except ValueError:
            return value
    if key in ("merge_order_ok",):
        return value in ("True", "true", "1")
    try:
        return float(value)
    except ValueError:
        return value


def load_summary(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist -- run collect.py first; report.py reads "
            f"nothing else."
        )
    with path.open(newline="") as fh:
        rows = [
            {key: _coerce(key, value) for key, value in row.items()}
            for row in csv.DictReader(fh)
        ]
    if not rows:
        raise SystemExit(f"{path} has no rows.")
    return rows


def is_rep2(row: dict) -> bool:
    return str(row["experiment"]).endswith("_rep2") or "_rep2" in str(row.get("_name", ""))


def cell_key(row: dict) -> str:
    """The cell a row belongs to, with `_rep2` folded back onto its real cell.

    `_rep2`'s own `experiment` carries the tag (`a0p8_N20_C4_s4126_rep2`), which
    is exactly right for keeping it OUT of the ordinary per-cell arm grouping
    (Tables B/D/F) -- but Table A needs to find its non-`_rep2` twin, which this
    strips back to.
    """
    experiment = str(row["experiment"])
    return experiment[: -len("_rep2")] if experiment.endswith("_rep2") else experiment


def real_cells(rows: list[dict]) -> dict[str, list[dict]]:
    """{cell -> [rows]}, `_rep2` excluded -- the grouping every table but A uses."""
    cells: dict[str, list[dict]] = {}
    for row in rows:
        if is_rep2(row):
            continue
        cells.setdefault(cell_key(row), []).append(row)
    return cells


# ---------------------------------------------------------------------------
# The usable threshold -- one number per cell, from the measured plateaus
# ---------------------------------------------------------------------------

def usable_threshold(cell_rows: list[dict]) -> int | None:
    """Highest of THRESHOLDS at least `USABLE_MARGIN` below the LOWEST plateau
    in the cell. None if no threshold clears every arm's plateau -- the stop
    condition "any cell's plateau leaves no threshold 5 points clear".

    Convention carried over unchanged from
    `../separability/EXPERIMENT_alpha_sweep.md`'s own rule: "the highest of
    ... that sits at least 5 points below the LOWEST plateau in the cell."
    """
    plateaus = [row["plateau_accuracy"] for row in cell_rows
               if row.get("plateau_accuracy") not in ("", None)]
    if not plateaus:
        return None
    lowest = min(plateaus)
    # A tiny epsilon against float noise at the exact boundary: e.g.
    # 0.85 - 0.05 == 0.7999999999999999 in binary floating point, one ULP
    # short of 80%'s 0.8 -- which would wrongly reject a threshold that is
    # legitimately "exactly 5 points below" the measured plateau.
    usable = [t for t in THRESHOLDS if t / 100.0 <= lowest - USABLE_MARGIN + 1e-9]
    return max(usable) if usable else None


def tta_at(row: dict, threshold: int) -> object:
    return row.get(f"tta_h_{threshold}", "")


def rounds_at(row: dict, threshold: int) -> object:
    return row.get(f"rounds_to_{threshold}", "")


def pct_diff(value: object, baseline: object) -> str:
    """100 * (value - baseline) / baseline, or a note when either side never
    crossed the threshold at all -- a blank divided by anything is not 0%,
    it is "no crossing", and the two must never be confused in a table meant to
    carry the campaign's actual verdicts."""
    if value in ("", None) or baseline in ("", None):
        return "no crossing"
    if baseline == 0:
        return "baseline crossed at t=0"
    return f"{100 * (value - baseline) / baseline:+.2f}%"


# ---------------------------------------------------------------------------
# Table A -- the gate: gpu vs its own byte-identical repeat
# ---------------------------------------------------------------------------

def table_a(rows: list[dict]) -> tuple[str, dict]:
    """Returns (markdown, {"stop": bool, "worst_pct": float | None})."""
    rep2 = [r for r in rows if is_rep2(r)]
    if not rep2:
        return ("*(no `_rep2` run in this summary yet -- Table A needs Batch 1's "
                "C=4 cell)*\n"), {"stop": False, "worst_pct": None}
    rep2 = rep2[0]
    cell = cell_key(rep2)
    twin = next((r for r in rows if not is_rep2(r) and cell_key(r) == cell
                and r["arm"] == "gpu"), None)
    if twin is None:
        return (f"*(found `_rep2` for cell {cell} but no matching `gpu` run)*\n",
                {"stop": False, "worst_pct": None})

    lines = [
        "| | gpu | gpu (`_rep2`) | difference |",
        "|---|---|---|---|",
        f"| plateau | {twin['plateau_accuracy']:.4f} | {rep2['plateau_accuracy']:.4f} | "
        f"{pct_diff(rep2['plateau_accuracy'], twin['plateau_accuracy'])} |",
        f"| noise (SD) | {twin['noise_sd']:.4f} | {rep2['noise_sd']:.4f} | -- |",
    ]
    worst_pct = 0.0
    for t in THRESHOLDS:
        a, b = tta_at(twin, t), tta_at(rep2, t)
        diff = pct_diff(b, a)
        lines.append(f"| TTA @ {t}% (h) | {a if a != '' else '--'} | "
                     f"{b if b != '' else '--'} | {diff} |")
        if diff not in ("no crossing", "baseline crossed at t=0"):
            worst_pct = max(worst_pct, abs(float(diff.rstrip("%"))))

    stop = worst_pct > REP2_STOP_PCT
    lines.append("")
    lines.append(
        f"Largest |difference| across the four thresholds: **{worst_pct:.2f}%** "
        f"(stop condition: > {REP2_STOP_PCT:g}%). "
        + ("**STOP CONDITION TRIGGERED.**" if stop else "Under the stop threshold.")
    )
    return "\n".join(lines) + "\n", {"stop": stop, "worst_pct": worst_pct}


# ---------------------------------------------------------------------------
# The shared per-cell, per-arm TTA-vs-baseline builder (Tables B and D)
# ---------------------------------------------------------------------------

def cell_table(cells: dict[str, list[dict]], axis_label: str,
              axis_value) -> tuple[str, dict[str, int | None]]:
    """One markdown table over the given cells; returns (markdown, {cell:
    usable_threshold})."""
    lines = [f"| {axis_label} | arm | plateau | usable threshold | "
            f"TTA (h) | vs baseline |",
            "|---|---|---|---|---|---|"]
    thresholds: dict[str, int | None] = {}
    for cell in sorted(cells):
        cell_rows = cells[cell]
        threshold = usable_threshold(cell_rows)
        thresholds[cell] = threshold
        baseline_row = next((r for r in cell_rows if r["arm"] == "baseline"), None)
        baseline_tta = tta_at(baseline_row, threshold) if (
            threshold is not None and baseline_row is not None) else ""

        if threshold is None:
            lines.append(f"| {axis_value(cell_rows[0])} | *(all arms)* | -- | "
                         f"**none usable** | -- | -- |")
            continue

        for row in sorted(cell_rows, key=lambda r: r["arm"]):
            tta = tta_at(row, threshold)
            vs_baseline = ("--" if row["arm"] == "baseline"
                           else pct_diff(tta, baseline_tta))
            lines.append(
                f"| {axis_value(row)} | {row['arm']} | "
                f"{row['plateau_accuracy']:.4f} | {threshold}% | "
                f"{tta if tta != '' else '--'} | {vs_baseline} |"
            )
    return "\n".join(lines) + "\n", thresholds


def table_b(rows: list[dict]) -> tuple[str, dict]:
    """C sweep, at N=20."""
    cells = {k: v for k, v in real_cells(rows).items()
            if v and v[0]["total_clients"] == 20}
    return cell_table(cells, "C", lambda r: r["concurrency"])


def table_d(rows: list[dict]) -> tuple[str, dict]:
    """Fleet size, at C=4."""
    cells = {k: v for k, v in real_cells(rows).items()
            if v and v[0]["concurrency"] == 4}
    return cell_table(cells, "N", lambda r: r["total_clients"])


# ---------------------------------------------------------------------------
# Table C -- staleness vs C
# ---------------------------------------------------------------------------

def table_c(rows: list[dict]) -> str:
    cells = {k: v for k, v in real_cells(rows).items()
            if v and v[0]["total_clients"] == 20}
    lines = ["| C | arm | mean staleness | max staleness | C - 1 |",
            "|---|---|---|---|---|"]
    for cell in sorted(cells, key=lambda c: cells[c][0]["concurrency"]):
        cell_rows = cells[cell]
        c = cell_rows[0]["concurrency"]
        for row in sorted(cell_rows, key=lambda r: r["arm"]):
            mean_s = row.get("mean_staleness", "")
            lines.append(
                f"| {c} | {row['arm']} | "
                f"{mean_s if mean_s == '' else f'{mean_s:.3f}'} | "
                f"{row.get('max_staleness', '')} | {c - 1} |"
            )
        pooled = [r["mean_staleness"] for r in cell_rows
                 if r.get("mean_staleness") not in ("", None)]
        if pooled:
            lines.append(f"| {c} | *(mean across arms)* | {st.fmean(pooled):.3f} | "
                         f"-- | {c - 1} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table E -- gpu vs gpu_oracle
# ---------------------------------------------------------------------------

def table_e(rows: list[dict]) -> str:
    lines = ["| C | metric | gpu | gpu_oracle |", "|---|---|---|---|"]
    cells = real_cells(rows)
    for cell in sorted(cells):
        cell_rows = cells[cell]
        if cell_rows[0]["total_clients"] != 20:
            continue
        gpu = next((r for r in cell_rows if r["arm"] == "gpu"), None)
        oracle = next((r for r in cell_rows if r["arm"] == "gpu_oracle"), None)
        if gpu is None or oracle is None:
            continue
        c = cell_rows[0]["concurrency"]
        threshold = usable_threshold([r for r in cell_rows
                                     if r["arm"] in ("gpu", "gpu_oracle",
                                                     "baseline", "cpu", "observed")])
        lines.append(f"| {c} | plateau | {gpu['plateau_accuracy']:.4f} | "
                     f"{oracle['plateau_accuracy']:.4f} |")
        if threshold is not None:
            lines.append(f"| {c} | TTA @ {threshold}% (h) | "
                         f"{tta_at(gpu, threshold) or '--'} | "
                         f"{tta_at(oracle, threshold) or '--'} |")
        lines.append(f"| {c} | mean staleness | "
                     f"{gpu.get('mean_staleness', '')} | "
                     f"{oracle.get('mean_staleness', '')} |")
    if len(lines) == 2:
        lines.append("| -- | *(no gpu/gpu_oracle pair found yet -- Batch 3 "
                     "runs last)* | -- | -- |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table F -- control: dispatch share per tier, every run
# ---------------------------------------------------------------------------

def table_f(rows: list[dict]) -> tuple[str, list[str]]:
    """Returns (markdown, warnings). A warning is a HEURISTIC flag for the
    "same tier mix" stop condition -- the brief gives no numeric epsilon for
    "the same", so this prints the actual shares for a human to judge and
    additionally calls out cells where every arm's shares sit within 1
    percentage point of each other on every tier, which is the shape "the
    selector is not working" would take.
    """
    lines = ["| experiment | arm | dispatches | share xavier | share orin | share thor |",
            "|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: (r["experiment"], r["arm"])):
        lines.append(
            f"| {row['experiment']} | {row['arm']} | {row.get('dispatches', '')} | "
            f"{row.get('share_xavier', '')} | {row.get('share_orin', '')} | "
            f"{row.get('share_thor', '')} |"
        )

    warnings = []
    for cell, cell_rows in sorted(real_cells(rows).items()):
        if len(cell_rows) < 2:
            continue
        for tier in ("xavier", "orin", "thor"):
            shares = [r[f"share_{tier}"] for r in cell_rows
                     if r.get(f"share_{tier}") not in ("", None)]
            if len(shares) >= 2 and max(shares) - min(shares) < 0.01:
                warnings.append(
                    f"cell {cell}: every arm's share_{tier} is within 1 point "
                    f"({min(shares):.4f}-{max(shares):.4f}) -- check whether "
                    f"the selector is doing anything on this tier"
                )
    return "\n".join(lines) + "\n", warnings


# ---------------------------------------------------------------------------
# The six written-down predictions -- numbers only, no verdicts
# ---------------------------------------------------------------------------

def predictions_section(rows: list[dict]) -> str:
    cells = real_cells(rows)
    lines = []

    def cell_at_c(c):
        return next((v for k, v in cells.items()
                    if v and v[0]["total_clients"] == 20
                    and v[0]["concurrency"] == c), None)

    lines.append("### 1 -- C=1 gives a null across all four arms")
    c1 = cell_at_c(1)
    if c1:
        for row in sorted(c1, key=lambda r: r["arm"]):
            lines.append(f"- {row['arm']}: plateau {row['plateau_accuracy']:.4f}, "
                         f"mean_staleness {row.get('mean_staleness', '')}")
    else:
        lines.append("- C=1 cell not in this summary yet")
    lines.append("")

    lines.append("### 2 -- advantage grows C=1 -> ~C=4, flattens/falls by C=8")
    lines.append("See Table B for the per-C, per-arm TTA-vs-baseline numbers "
                 "across C=1/2/4/8.")
    lines.append("")

    lines.append("### 3 -- cpu is slower than baseline in every cell with C>1")
    for c in (2, 4, 8):
        cell_rows = cell_at_c(c)
        if not cell_rows:
            continue
        cpu = next((r for r in cell_rows if r["arm"] == "cpu"), None)
        baseline = next((r for r in cell_rows if r["arm"] == "baseline"), None)
        if cpu and baseline:
            threshold = usable_threshold(cell_rows)
            lines.append(
                f"- C={c}: cpu {tta_at(cpu, threshold) if threshold else ''} h vs "
                f"baseline {tta_at(baseline, threshold) if threshold else ''} h "
                f"@ {threshold}%   (secs/round: cpu {cpu.get('seconds_per_round','')}, "
                f"baseline {baseline.get('seconds_per_round','')})"
            )
    lines.append("")

    lines.append("### 4 -- N=40 gives the same advantage as N=20, at C=4")
    lines.append("See Table D.")
    lines.append("")

    lines.append("### 5 -- gpu loses to gpu_oracle (a tie would mean the "
                 "jitter did not make estimation matter)")
    lines.append("See Table E.")
    lines.append("")

    lines.append("### 6 -- gpu vs observed stays undecided at one seed")
    for c in (1, 2, 4, 8):
        cell_rows = cell_at_c(c)
        if not cell_rows:
            continue
        gpu = next((r for r in cell_rows if r["arm"] == "gpu"), None)
        observed = next((r for r in cell_rows if r["arm"] == "observed"), None)
        if gpu and observed:
            threshold = usable_threshold(cell_rows)
            lines.append(
                f"- C={c}: gpu {tta_at(gpu, threshold) if threshold else ''} h vs "
                f"observed {tta_at(observed, threshold) if threshold else ''} h "
                f"@ {threshold}%"
            )
    lines.append("")

    lines.append("Each of the six is reported as numbers above, not as a "
                 "held/not-held verdict: the brief's own criteria for several "
                 "of them (\"a null\", \"flattens\", \"a tie\") are qualitative, "
                 "and a threshold invented here to force a boolean would be "
                 "exactly the kind of explaining-away the brief forbids. "
                 "Predictions 5's exception -- gpu vs gpu_oracle margin -- and "
                 "the stop conditions below are the numbers the brief DOES "
                 "give an explicit rule for.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------

def stop_conditions(rows: list[dict], table_a_result: dict,
                    table_b_thresholds: dict, table_d_thresholds: dict,
                    table_f_warnings: list[str]) -> list[str]:
    """The four stop conditions. Two have an explicit numeric rule in the
    brief and are evaluated as hard pass/fail here; two are qualitative and are
    surfaced as flags from the tables above for a human to confirm."""
    findings = []

    if table_a_result.get("stop"):
        findings.append(
            f"STOP (explicit rule): _rep2 differs from gpu by "
            f"{table_a_result['worst_pct']:.2f}% on time-to-target, over the "
            f"{REP2_STOP_PCT:g}% the brief sets. One seed cannot support any "
            f"comparison in this campaign; reconsider the remaining runs "
            f"before spending them."
        )

    no_threshold_cells = [cell for cell, t in
                          {**table_b_thresholds, **table_d_thresholds}.items()
                          if t is None]
    if no_threshold_cells:
        findings.append(
            f"STOP (explicit rule): no threshold clears 5 points below the "
            f"lowest plateau in {len(no_threshold_cells)} cell(s): "
            f"{', '.join(no_threshold_cells)}."
        )

    if table_f_warnings:
        findings.append(
            "FLAG (qualitative -- confirm from Table F): possible same-tier-mix "
            "dispatching. " + "; ".join(table_f_warnings)
        )

    cells = real_cells(rows)
    staleness_flags = []
    for cell, cell_rows in cells.items():
        if not cell_rows or cell_rows[0]["total_clients"] != 20:
            continue
        c = cell_rows[0]["concurrency"]
        pooled = [r["mean_staleness"] for r in cell_rows
                 if r.get("mean_staleness") not in ("", None)]
        if not pooled or c <= 1:
            continue
        mean_pooled = st.fmean(pooled)
        expected = c - 1
        if expected > 0 and abs(mean_pooled - expected) / expected > 0.5:
            staleness_flags.append(
                f"C={c}: mean staleness {mean_pooled:.2f} vs expected ~{expected}"
            )
    if staleness_flags:
        findings.append(
            "FLAG (qualitative -- confirm from Table C): mean staleness far "
            "from C-1. " + "; ".join(staleness_flags)
        )

    return findings


# ---------------------------------------------------------------------------

def build_report(rows: list[dict]) -> str:
    lines = ["# Stage 3 — sweep report", ""]

    lines.append("## Table A — the gate: gpu vs `_rep2`")
    table_a_md, table_a_result = table_a(rows)
    lines.append(table_a_md)

    lines.append("## Table B — C sweep (N=20)")
    table_b_md, table_b_thresholds = table_b(rows)
    lines.append(table_b_md)

    lines.append("## Table C — staleness vs C")
    lines.append(table_c(rows))

    lines.append("## Table D — fleet size (C=4)")
    table_d_md, table_d_thresholds = table_d(rows)
    lines.append(table_d_md)

    lines.append("## Table E — the upper bound: gpu vs gpu_oracle")
    lines.append(table_e(rows))

    lines.append("## Table F — control: dispatch share per tier")
    table_f_md, table_f_warnings = table_f(rows)
    lines.append(table_f_md)

    lines.append("## The six predictions, written down before the runs")
    lines.append(predictions_section(rows))

    lines.append("## Stop conditions")
    findings = stop_conditions(rows, table_a_result, table_b_thresholds,
                               table_d_thresholds, table_f_warnings)
    if findings:
        for f in findings:
            lines.append(f"- {f}")
    else:
        lines.append("None triggered.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--summary", type=Path, default=RESULTS / "run_summary.csv")
    parser.add_argument("--out", type=Path, default=EVIDENCE / "report.md")
    args = parser.parse_args()

    rows = load_summary(args.summary)
    report = build_report(rows)

    if str(args.out) == "/dev/stdout":
        print(report)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
