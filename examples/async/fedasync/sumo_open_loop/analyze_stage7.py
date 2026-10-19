"""
Stage 7 analysis — open population (prompts/STAGE7_open_population.md, Step 4).

Self-contained, for the reason every stage's analyser is: each cell stays
reproducible without risking a previous one's finished output.

Reads each run's own server log. Everything this stage needs is already in
those logs, emitted by `sumo_open_server.py` as it ran:

    [Stage7] dispatch round= slot= veh_id= tier= observations= G_i= G_i_at_prior= tau_hat=
    [Stage7] report   round= slot= veh_id= tier= dispatch_t= finish_t= uploaded= raw_in_range= sleep=
    [Stage7] recycle  slot_id= outgoing_veh_id= incoming_veh_id= tier_out= tier_in= state_reset=
    [Stage7] summary  p_recur= recycles= state_resets= slot_exhausted= occupancy_digest=

The simulated clock is REBUILT from the dispatch order and the per-vehicle
sleep the log carries, exactly as the closed-loop cells rebuild theirs — never
read from PLATO's `elapsed_time`, which carries host `processing_time` and is
not comparable across runs that shared a GPU.

Usage:
    python3 analyze_stage7.py              # every stage7_* run found
    python3 analyze_stage7.py --stage 7b   # every stage7b_<arm>_<regime> run
    python3 analyze_stage7.py --validate   # parse + sanity checks only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parent
RUNS = FEDASYNC / "runs" / "sumo_open_loop"
PREFIX = "stage7"  # set from --stage in main()

# Stage 7b appends identity fields to the dispatch line.
IDENTITY_FIELDS = re.compile(r"identity_id=(\S+) pred_tier=(\S+) tau_obs=(\S+)")
EVIDENCE = FEDASYNC / "gpu_tiers_dataset" / "stage7_evidence"

sys.path.insert(0, str(HERE))
from sumo_open_mobility import digest_at  # noqa: E402

CONCURRENCY = 4
TIERS = ("xavier", "orin", "thor")
WINDOWS = ((1, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000))
CHECKPOINTS = (100, 250, 500, 1000, 2000, 4000)
THRESHOLDS = (70, 75, 78, 80)
PLATEAU_FRACTION = 0.10
SMOOTH_WINDOW = 11
NOISE_WINDOW = 101
EVAL_INTERVAL = 10
COVERAGE_CSV = HERE / "sumo_trace" / "e1_open_poisson_seed4126_range400" / "coverage_windows.csv"
SLOTS = 20

ROUND_LINE = re.compile(r"Starting round (\d+)/(\d+)")
MERGE_LINE = re.compile(
    r"FedAsync: step \d+ - client #(\d+), effective_round=(\d+), staleness=(\d+)"
)
DISPATCH_LINE = re.compile(
    r"\[Stage7\] dispatch round=(-?\d+) slot=(\d+) veh_id=(\S+) tier=(\S+) "
    r"observations=(\d+) G_i=(\S+) G_i_at_prior=([01]) tau_hat=(\S+)"
)
REPORT_LINE = re.compile(
    r"\[Stage7\] report round=(\d+) slot=(\d+) veh_id=(\S+) tier=(\S+) "
    r"dispatch_t=([\d.eE+-]+) finish_t=([\d.eE+-]+) uploaded=([01]) "
    r"raw_in_range=([01]) sleep=([\d.eE+-]+)"
)
SUMMARY_LINE = re.compile(
    r"\[Stage7\] summary p_recur=(\S+) recycles=(\d+) state_resets=(\d+) "
    r"slot_exhausted=(\d+) occupancy_digest=(\S+)"
)
RECYCLE_LINE = re.compile(
    r"\[Stage7\] recycle slot_id=(\S+) outgoing_veh_id=(\S+) "
    r"incoming_veh_id=(\S+) .* state_reset=([01])"
)


@dataclass
class Dispatch:
    round: int
    slot: int
    veh_id: str
    tier: str
    observations: int
    g_i: float
    g_i_at_prior: bool
    tau_hat: float
    sleep: float | None = None
    raw_in_range: bool | None = None
    identity_id: str | None = None
    pred_tier: str | None = None
    start: float = 0.0
    finish: float = 0.0
    merge_round: int | None = None
    staleness: int | None = None


@dataclass
class Run:
    name: str
    p_recur: float | None = None
    dispatches: list[Dispatch] = field(default_factory=list)
    curve: list[dict] = field(default_factory=list)
    clock: dict[int, float] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    recycles: int = 0
    state_resets: int = 0
    recycle_seq: list[tuple] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def server_log(directory: Path) -> Path:
    # A run dir can hold leftovers of an earlier, aborted start (e.g. the
    # 2026-09-29 Stage 11 wiring check, or a run killed by a stop).  The run
    # being analysed is the most recent server process; its results CSV carries
    # the same pid (results_csv), so a mismatch still fails loudly.
    # Only a server that wrote results/<pid>.csv is a run: a process that merely
    # loaded the config (e.g. a test) leaves a stub server log and no CSV.
    logs = sorted(directory.glob("server_*.log"), key=lambda p: p.stat().st_mtime)
    runs = [p for p in logs if (directory / "results" / f"{p.stem.split('_', 1)[1]}.csv").exists()]
    if not runs:
        raise SystemExit(f"no server log with a results CSV under {directory} ({len(logs)} logs)")
    return runs[-1]


def results_csv(directory: Path) -> Path:
    pid = server_log(directory).stem.split("_", 1)[1]
    found = directory / "results" / f"{pid}.csv"
    if not found.exists():
        raise SystemExit(f"no results/{pid}.csv for server_{pid}.log under {directory}")
    return found


def parse(name: str) -> Run:
    directory = RUNS / f"{PREFIX}_{name}"
    run = Run(name=name)
    pending: dict[tuple[int, int], Dispatch] = {}
    order: list[Dispatch] = []

    with server_log(directory).open(errors="replace") as fh:
        for line in fh:
            m = DISPATCH_LINE.search(line)
            if m:
                d = Dispatch(
                    round=int(m.group(1)), slot=int(m.group(2)), veh_id=m.group(3),
                    tier=m.group(4), observations=int(m.group(5)),
                    g_i=float(m.group(6)) if m.group(6) != "nan" else float("nan"),
                    g_i_at_prior=m.group(7) == "1",
                    tau_hat=float(m.group(8)) if m.group(8) != "nan" else float("nan"),
                )
                extra = IDENTITY_FIELDS.search(line)
                if extra:
                    d.identity_id = extra.group(1)
                    d.pred_tier = None if extra.group(2) == "-" else extra.group(2)
                run.dispatches.append(d)
                pending[(d.round, d.slot)] = d
                order.append(d)
                continue

            m = REPORT_LINE.search(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                d = pending.pop(key, None)
                if d is None:
                    continue
                d.sleep = float(m.group(9))
                d.raw_in_range = m.group(8) == "1"
                continue

            m = MERGE_LINE.search(line)
            if m:
                slot, eff_round, stale = int(m.group(1)), int(m.group(2)), int(m.group(3))
                for d in reversed(order):
                    if d.slot == slot and d.merge_round is None and d.sleep is not None:
                        d.merge_round = eff_round
                        d.staleness = stale
                        break
                continue

            m = SUMMARY_LINE.search(line)
            if m:
                run.p_recur = float(m.group(1))
                run.summary = {
                    "recycles": int(m.group(2)), "state_resets": int(m.group(3)),
                    "slot_exhausted": int(m.group(4)), "occupancy_digest": m.group(5),
                }
                continue

            m = RECYCLE_LINE.search(line)
            if m:
                run.recycles += 1
                run.state_resets += int(m.group(4))
                run.recycle_seq.append(m.group(1, 2, 3))

    # Rebuild the simulated clock: a C-slot queue fed by the dispatch order,
    # with each job's duration the VEHICLE's own sleep as logged.
    slots = [0.0] * CONCURRENCY
    for d in run.dispatches:
        if d.sleep is None:
            continue
        i = min(range(CONCURRENCY), key=lambda k: slots[k])
        d.start = slots[i]
        d.finish = d.start + d.sleep
        slots[i] = d.finish

    clock: dict[int, float] = {}
    for d in run.dispatches:
        if d.merge_round is not None:
            clock[d.merge_round] = max(clock.get(d.merge_round, 0.0), d.finish)
    run.clock = clock

    with results_csv(directory).open(newline="") as fh:
        for row in csv.DictReader(fh):
            rnd = int(row["round"])
            if rnd in clock:
                run.curve.append({
                    "round": rnd, "accuracy": float(row["accuracy"]),
                    "sim_time_reconstructed": clock[rnd],
                })
    return run


# --- metrics ---------------------------------------------------------------

def in_window(rnd, lo, hi):
    return (lo <= rnd <= hi) if lo == 1 else (lo < rnd <= hi)


def percentile(values, p):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (k - lo)


def clock_at(run, rnd):
    if rnd in run.clock:
        return run.clock[rnd]
    earlier = [r for r in run.clock if r <= rnd]
    return run.clock[max(earlier)] if earlier else None


def waste(run):
    reported = [d for d in run.dispatches if d.raw_in_range is not None]
    def tally(items):
        n = len(items)
        w = sum(1 for d in items if not d.raw_in_range)
        return {"dispatches": n, "wasted": w, "wasted_frac": w / n if n else 0.0}
    return {
        "overall": tally(reported),
        "per_tier": {t: tally([d for d in reported if d.tier == t]) for t in TIERS},
    }


def dispatch_share(run):
    reported = [d for d in run.dispatches if d.raw_in_range is not None]
    out = {}
    c = Counter(d.tier for d in reported)
    out["overall"] = {t: c[t] / len(reported) for t in TIERS} if reported else {}
    for lo, hi in WINDOWS:
        items = [d for d in reported if in_window(d.round, lo, hi)]
        cc = Counter(d.tier for d in items)
        out[f"{lo}-{hi}"] = {t: cc[t] / len(items) for t in TIERS} if items else {}
    return out


def tier_mix_by_information(run):
    """Stage 7b Predictions 1-2: tier mix of dispatches made on NO observation
    of the identity vs on >= 1.  Prediction 1 is read on the first group."""
    out = {}
    for label, keep in (("zero_obs", lambda d: d.observations == 0),
                        ("ge1_obs", lambda d: d.observations >= 1)):
        sel = [d for d in run.dispatches if keep(d)]
        c = Counter(d.tier for d in sel)
        n = len(sel)
        out[label] = {"n": n, **{t: (c[t] / n if n else None) for t in TIERS}}
    return out


def predictor_agreement(run):
    """Fraction of dispatches whose predictor tier equals the occupant tier."""
    scored = [d for d in run.dispatches if d.pred_tier is not None]
    if not scored:
        return None
    return sum(d.pred_tier == d.tier for d in scored) / len(scored)


def kappa(run):
    """Measured dispatches per vehicle before it exits."""
    per_vehicle = Counter(d.veh_id for d in run.dispatches)
    counts = list(per_vehicle.values())
    return {
        "vehicles_dispatched": len(per_vehicle),
        "mean": st.mean(counts), "median": st.median(counts),
        "p10": percentile(counts, 0.10), "p90": percentile(counts, 0.90),
        "max": max(counts),
        "distribution": dict(sorted(Counter(counts).items())),
    }


def information_state(run):
    """How much the server knew about the occupant it was ranking."""
    obs = [d.observations for d in run.dispatches]
    at_prior = [d.g_i_at_prior for d in run.dispatches if not math.isnan(d.g_i)]
    out = {
        "observations_mean": st.mean(obs) if obs else None,
        "observations_median": st.median(obs) if obs else None,
        "observations_max": max(obs) if obs else None,
        "fraction_zero_observations": sum(1 for o in obs if o == 0) / len(obs) if obs else None,
        "per_window": {},
    }
    if at_prior:
        out["fraction_G_i_at_prior"] = sum(at_prior) / len(at_prior)
    for lo, hi in WINDOWS:
        items = [d for d in run.dispatches if in_window(d.round, lo, hi)]
        if not items:
            continue
        known = [d for d in items if not math.isnan(d.g_i)]
        out["per_window"][f"{lo}-{hi}"] = {
            "observations_mean": st.mean([d.observations for d in items]),
            "fraction_zero_observations":
                sum(1 for d in items if d.observations == 0) / len(items),
            "fraction_G_i_at_prior":
                (sum(1 for d in known if d.g_i_at_prior) / len(known)) if known else None,
        }
    return out


def s_tau(run):
    """s(tau_hat) = 1/sqrt(1+tau_hat), per tier — the staleness discount."""
    out = {}
    for t in TIERS:
        vals = [1.0 / math.sqrt(1.0 + d.tau_hat)
                for d in run.dispatches
                if d.tier == t and not math.isnan(d.tau_hat)]
        if vals:
            out[t] = {"mean": st.mean(vals), "median": st.median(vals), "n": len(vals)}
    # How often an unseen vehicle's default tau_hat=0.01 (s=0.995) outranks
    # a known-fast one -- the specific question the stage prompt asks.
    unseen = [d for d in run.dispatches
              if not math.isnan(d.tau_hat) and abs(d.tau_hat - 0.01) < 1e-9]
    out["_unseen_default_fraction"] = (
        len(unseen) / len(run.dispatches) if run.dispatches else None
    )
    return out


def staleness_stats(run):
    vals = [d.staleness for d in run.dispatches if d.staleness is not None]
    return {"mean": st.mean(vals) if vals else None,
            "max": max(vals) if vals else None, "n": len(vals)}


def conservation(run):
    reported = [d for d in run.dispatches if d.raw_in_range is not None]
    if not run.clock or not reported:
        return {}
    rounds = max(run.clock)
    measured = run.clock[rounds]
    mean_sleep = st.mean([d.sleep for d in reported])
    w = sum(1 for d in reported if not d.raw_in_range) / len(reported)
    predicted = rounds * mean_sleep / (CONCURRENCY * (1 - w))
    return {"rounds": rounds, "measured": measured, "predicted": predicted,
            "rel_error": (predicted - measured) / measured, "mean_sleep": mean_sleep}


def rolling_median(v, w):
    half = w // 2
    return [st.median(v[max(0, i - half): min(len(v), i + half + 1)]) for i in range(len(v))]


def curve_stats(run):
    rows = [r for r in run.curve if r["round"] % EVAL_INTERVAL == 0]
    if not rows:
        return {}
    acc = [r["accuracy"] for r in rows]
    tail = acc[-max(1, int(len(acc) * PLATEAU_FRACTION)):]
    smoothed = rolling_median(acc, SMOOTH_WINDOW)
    base = rolling_median(acc, NOISE_WINDOW)
    resid = [a - b for a, b in zip(acc, base)]
    out = {"plateau": st.median(tail), "best": max(acc),
           "noise_sd": st.stdev(resid) if len(resid) > 1 else None}
    for tgt in THRESHOLDS:
        hit = next((r for r, v in zip(rows, smoothed) if v >= tgt / 100.0), None)
        out[f"rounds_to_{tgt}"] = hit["round"] if hit else None
        out[f"tta_h_{tgt}"] = hit["sim_time_reconstructed"] / 3600.0 if hit else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--stage", default="7", help="7 or 7b")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    global PREFIX
    PREFIX = f"stage{args.stage}"
    if args.out is None:
        args.out = EVIDENCE / f"{PREFIX}_analysis.json"

    # <arm>_<regime> only: stage7b_* also matches the stage7b_preflight_* runs.
    names = sorted(
        p.name[len(PREFIX) + 1:] for p in RUNS.glob(f"{PREFIX}_*")
        if (p / "results").exists() and len(p.name[len(PREFIX) + 1:].split("_")) == 2
    )
    runs = {}
    for n in names:
        r = parse(n)
        runs[n] = r
        done = max(r.clock) if r.clock else 0
        print(f"{n:18s} dispatches={len(r.dispatches):5d} reported="
              f"{sum(1 for d in r.dispatches if d.raw_in_range is not None):5d} "
              f"rounds={done} p_recur={r.p_recur}")
    if args.validate:
        return 0

    payload = {}
    for n, r in runs.items():
        payload[n] = {
            "p_recur": r.p_recur,
            "conservation": conservation(r),
            "clock_at": {str(c): clock_at(r, c) for c in CHECKPOINTS},
            "waste": waste(r),
            "dispatch_share": dispatch_share(r),
            "kappa": kappa(r),
            "tier_mix_by_information": tier_mix_by_information(r),
            "predictor_tier_agreement": predictor_agreement(r),
            "information_state": information_state(r),
            "s_tau": s_tau(r),
            "staleness": staleness_stats(r),
            "curve": curve_stats(r),
            "recycles": r.recycles, "state_resets": r.state_resets,
            "summary": r.summary,
        }

    # The cross-run invariant, compared at a COMMON simulated time (see
    # sumo_open_mobility.digest_at -- end-of-run digests are prefixes of
    # differing length and will differ even when nothing is wrong).
    finals = [max(r.clock) and clock_at(r, max(r.clock)) for r in runs.values() if r.clock]
    if finals:
        common_t = min(finals)
        expected, rec, turned = digest_at(COVERAGE_CSV, SLOTS, common_t)
        seqs = {n: r.recycle_seq for n, r in runs.items() if r.recycle_seq}
        prefix_n = min(len(v) for v in seqs.values())
        realised = {
            n: hashlib.md5(repr(v[:prefix_n]).encode()).hexdigest()
            for n, v in seqs.items()
        }
        distinct = sorted(set(realised.values()))
        first_divergence = None
        if len(distinct) > 1:
            names = sorted(seqs)
            ref = seqs[names[0]]
            for other in names[1:]:
                for i, (a, b) in enumerate(zip(ref[:prefix_n], seqs[other][:prefix_n])):
                    if a != b:
                        first_divergence = {
                            "runs": [names[0], other], "index": i,
                            "values": [list(a), list(b)],
                        }
                        break
                if first_divergence:
                    break
        payload["_occupancy_invariant"] = {
            "common_time_s": common_t, "expected_digest": expected,
            "expected_recycles": rec, "expected_turned_away": turned,
            "realised_prefix_recycles": prefix_n,
            "realised_prefix_digests": realised,
            "identical_across_runs": len(distinct) == 1,
            "first_divergence": first_divergence,
            "note": "each run's own end-of-run digest is a prefix at ITS final "
                    "time; compare at this common t instead",
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
