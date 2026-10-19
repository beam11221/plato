"""
One health-check pass over the running arms. Exits non-zero if anything is wrong.

Written to be run repeatedly -- by a watchdog, by a monitoring loop, or by hand.
Every check here corresponds to a failure this project has actually hit, so a
clean pass means the specific things that went wrong before are not happening now.

    STALLED    the log stopped growing while the process is alive. This is what
               the round-719 deadlock looked like; the process burns CPU and
               nothing ever wakes it. The single most important check.
    DEADLOCK   the deadlock's own signature in the log, in case it recurs in a
               form the staleness check misses.
    CRASH      a traceback.
    PIPELINE   cycles >= rounds, and the held cycles cannot outnumber the wasted
               updates that caused them. A violation means the refill accounting
               drifted, which silently changes the concurrency the whole
               experiment is calibrated against.
    WASTE      per-tier waste far from what the trace predicts means the fleet,
               the sleep times and the coverage table are not the same fleet.
    DISK       the allocation is shared with the other machine; an arm whose
               cleaner died takes both machines down.

Usage:
    python3 health_check.py                            # the n20 cell
    python3 health_check.py --cell n100 --arms cpu gpu # this machine's n100 arms
    python3 health_check.py --stale-minutes 10
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
RUNS_DIR = HERE.parent / "runs" / "sumo_close_loop"

ARMS = ("baseline", "gpu", "cpu", "observed")

# Waste predicted from the coverage trace, for dispatch uniform within a window.
# A live arm steers, so it will not match exactly; these are the outer bounds
# beyond which something is wrong with the inputs rather than with the selector.
WASTE_BOUNDS = {"xavier": (25.0, 55.0), "orin": (0.0, 20.0), "thor": (0.0, 20.0)}

REPORT_RE = re.compile(r"\[SumoRsu\] report round=(\d+)")
WASTE_RE = re.compile(
    r"waste round=(\d+) cycles=(\d+) total=(\d+)/(\d+) \(([\d.]+)%\)(.*)"
)
TIER_RE = re.compile(r"(\w+)=(\d+)/(\d+) \(([\d.]+)%\)")

problems: list[str] = []
notes: list[str] = []


def problem(arm: str, kind: str, detail: str) -> None:
    problems.append(f"{kind:<9} {arm:<9} {detail}")


def note(text: str) -> None:
    notes.append(text)


def running_arms(cell: str) -> dict[str, bool]:
    """Which arms have a live process on THIS machine."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", "sumo_fedasync.py"], capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        return {}
    return {arm: f"{arm}_{cell}.toml" in out for arm in ARMS}


def check_arm(arm: str, cell: str, stale_minutes: float, is_local: bool) -> None:
    log = LOG_DIR / f"{arm}_{cell}.log"
    if not log.exists():
        return

    text = log.read_text(errors="replace")
    age_min = (time.time() - log.stat().st_mtime) / 60

    done = (LOG_DIR / f"{arm}_{cell}.done").exists()
    if done:
        note(f"{arm}: FINISHED")
        return

    # -- STALLED -----------------------------------------------------------
    # A finished arm writes a .done marker; a live one writes constantly. A log
    # that is neither is either hung or was stopped.
    if age_min > stale_minutes:
        state = "process alive -> HUNG" if is_local else "no local process"
        tail = text.rstrip().rsplit("\n", 1)[-1][:90]
        problem(arm, "STALLED", f"log idle {age_min:.0f} min ({state}) | {tail}")

    # -- DEADLOCK ----------------------------------------------------------
    if "No eligible clients (busy=0" in text:
        n = text.count("No eligible clients (busy=0")
        problem(arm, "DEADLOCK", f"{n}x 'No eligible clients (busy=0' — pre-fix code?")

    # -- CRASH -------------------------------------------------------------
    if "Traceback" in text:
        problem(arm, "CRASH", f"{text.count('Traceback')} traceback(s)")

    matches = WASTE_RE.findall(text)
    if not matches:
        note(f"{arm}: {len(REPORT_RE.findall(text))} reports, no summary yet")
        return
    rounds, cycles, wasted, total, pct, tiers = matches[-1]
    rounds, cycles, wasted, total = int(rounds), int(cycles), int(wasted), int(total)

    # -- PIPELINE ----------------------------------------------------------
    # Two invariants that are true by construction, whatever the arm or cell:
    #
    #   cycles >= rounds            a held cycle never un-does a round
    #   cycles - rounds <= wasted   holding a round needs >= 1 wasted update
    #
    # An earlier version tried to reconstruct the held count by counting rounds
    # that carry more than one report line. That is NOT an invariant: several
    # reports can arrive while the round is unchanged for reasons other than
    # waste (the startup fill, or two reports landing between two merges), so it
    # raised false alarms on the n100 cell. Prefer a check that cannot be wrong
    # over one that is more precise when it happens to hold -- a monitor that
    # cries wolf is worse than no monitor.
    if cycles < rounds:
        problem(arm, "PIPELINE", f"cycles {cycles} < rounds {rounds}")
    if cycles - rounds > wasted:
        problem(
            arm,
            "PIPELINE",
            f"{cycles - rounds} held cycles but only {wasted} wasted updates",
        )

    # -- WASTE -------------------------------------------------------------
    per_tier = {t: float(p) for t, _, _, p in TIER_RE.findall(tiers)}
    for tier, (lo, hi) in WASTE_BOUNDS.items():
        if tier in per_tier and not lo <= per_tier[tier] <= hi:
            problem(
                arm,
                "WASTE",
                f"{tier} waste {per_tier[tier]:.1f}% outside [{lo}, {hi}] — "
                "check that the fleet, sleep times and coverage table agree",
            )

    accuracy = ""
    results = sorted((RUNS_DIR / f"{arm}_{cell}" / "results").glob("*.csv"))
    if results:
        rows = list(csv.DictReader(results[0].open()))
        if rows and rows[-1].get("accuracy"):
            accuracy = f"  acc {100 * float(rows[-1]['accuracy']):.1f}%"

    note(
        f"{arm}: round {rounds:>5} / cycles {cycles:>5}  waste {pct:>5}%"
        f"  (xavier {per_tier.get('xavier', 0):.1f}%){accuracy}"
    )


def check_disk(quota_gb: int) -> None:
    if not RUNS_DIR.parent.exists():
        return
    out = subprocess.run(
        ["du", "-sBG", str(RUNS_DIR.parent)], capture_output=True, text=True
    ).stdout
    used = int(re.sub(r"[^0-9]", "", out.split()[0]) or 0)
    note(f"disk: {used} GB of the {quota_gb} GB shared allocation")
    if used > quota_gb * 0.8:
        problem("-", "DISK", f"{used} GB of {quota_gb} GB used — a cleaner may be dead")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--cell", default="n20", choices=("n20", "n100"))
    parser.add_argument("--stale-minutes", type=float, default=8.0)
    parser.add_argument("--quota-gb", type=int, default=200)
    args = parser.parse_args()

    live = running_arms(args.cell)
    for arm in args.arms:
        check_arm(arm, args.cell, args.stale_minutes, live.get(arm, False))
    check_disk(args.quota_gb)

    stamp = time.strftime("%F %T")
    for line in notes:
        print(f"[{stamp}] {line}")

    if problems:
        print(f"\n[{stamp}] {len(problems)} PROBLEM(S):")
        for line in problems:
            print(f"  {line}")
        sys.exit(1)

    print(f"[{stamp}] all checks passed")


if __name__ == "__main__":
    main()
