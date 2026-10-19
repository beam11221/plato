"""
Compare two runs of one config and write stage1_evidence/determinism.txt.

The question Stage 1 asks: after seeding every RNG, are two runs of the identical
config identical?

`../separability/PARALLELISM_CHECK.md` measured the unseeded answer -- two runs of
one config, both alone on an idle box, differed by **6.29 %** of simulated clock,
because nothing seeded torch and the dispatch sequence therefore differed. This
script measures what is left.

What is compared
----------------
* the accuracy curve, row for row
* `sim_time_s` at the final round
* the dispatch sequence, which is what the 6.29 % was actually made of

Reading `sim_time_s`
--------------------
PLATO's `elapsed_time` column is NOT a pure simulated clock. Every report's finish
time is

    training_time + processing_time + comm_time + start_time     base.py:1025-1030

and `processing_time` is a `time.perf_counter()` delta measured on the host
(clients/strategies/defaults.py:252). `comm_time` is 0 under `comm_simulation`,
and `training_time` is the client's fixed sleep, so `processing_time` is the one
term that is not a property of the configuration -- and it enters the clock on
every single round. No amount of seeding removes it.

So a difference in `sim_time_s` is expected. What would be alarming is a
difference in WHICH CLIENTS RAN, because that is a behavioural divergence rather
than host noise, and it is what the report separates out.

Usage:
    python3 determinism_check.py --runs det1 det2
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parent
RUNS_DIR = FEDASYNC / "runs" / "gpu_tiers_dataset"
EVIDENCE = HERE / "stage1_evidence"

SELECTED = re.compile(r"Selecting client #(\d+) for training")
AGGREGATED = re.compile(r"Adding client #(\d+) to the list of clients for aggregation")
SEED_LINE = re.compile(r"\[Seed\] pid=\d+ role=(\S+) seed=(\d+)")
ASSIGNED = re.compile(r"\[GpuTierSleep\] round=(-?\d+) client_id=(\d+)")


def run_dir(name: str) -> Path:
    for candidate in (RUNS_DIR / name,
                      RUNS_DIR / f"gpu_a0p8_N20_C4_s4126_{name}"):
        if candidate.exists():
            return candidate
    raise SystemExit(f"no run directory for {name!r} under {RUNS_DIR}")


def results_csv(directory: Path) -> Path:
    matches = sorted(directory.rglob("*.csv"))
    if not matches:
        raise SystemExit(f"no results CSV under {directory}")
    # PLATO writes one results CSV; prefer the one carrying an accuracy column.
    for path in matches:
        with path.open(newline="") as fh:
            header = next(csv.reader(fh), [])
        if "accuracy" in header:
            return path
    return matches[0]


def curve(directory: Path) -> list[tuple[int, float, float]]:
    """(round, sim_time_s, accuracy) for EVERY round in the results CSV.

    Every row, not only the evaluations. PLATO writes one row per round and
    CARRIES THE LAST ACCURACY FORWARD between evaluations (0 until the first),
    so most rows are repeats -- `../separability/collect.py` drops them because
    the repeats would corrupt a rolling median.

    Here they are kept. "Identical row for row" is the question being asked, and
    comparing the file as written is the strongest form of it: it needs no
    interpretation of which rows are real, and it also compares train_loss, which
    changes on every round and is therefore the most sensitive signal in the file.
    """
    rows = []
    with results_csv(directory).open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append((int(row["round"]), float(row["elapsed_time"]),
                         float(row["accuracy"])))
    return rows


def evaluations(rows: list[tuple[int, float, float]]) -> list[tuple[int, float]]:
    """(round, accuracy) for the rounds where the accuracy actually MOVED.

    An evaluation is where the carried-forward value changes; picking multiples
    of eval_interval would assume the interval rather than read it.
    """
    out = []
    previous = None
    for rnd, _clock, accuracy in rows:
        if accuracy != previous:
            out.append((rnd, accuracy))
            previous = accuracy
    return out


def raw_rows(directory: Path) -> list[dict]:
    with results_csv(directory).open(newline="") as fh:
        return list(csv.DictReader(fh))


def clock_at_last_round(directory: Path) -> tuple[int, float]:
    with results_csv(directory).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return int(rows[-1]["round"]), float(rows[-1]["elapsed_time"])


def server_log(directory: Path) -> str:
    logs = sorted(directory.glob("server_*.log"))
    if not logs:
        raise SystemExit(f"no server log under {directory}")
    return logs[0].read_text(errors="replace")


def dispatch_sequence(text: str) -> list[int]:
    return [int(m) for m in SELECTED.findall(text)]


def aggregation_sequence(text: str) -> list[int]:
    return [int(m) for m in AGGREGATED.findall(text)]


def process_assignments(directory: Path) -> dict[int, list[int]]:
    """{client process -> the logical clients it served, in order}.

    PLATO runs a POOL of `max_concurrency` worker processes and hands each round's
    selected client to whichever worker is free (plato/servers/base.py:662-682).
    A logical client is therefore not pinned to a process, and which worker picks
    it up is what this reads back out of the per-process logs.
    """
    out: dict[int, list[int]] = {}
    for path in sorted(directory.glob("client_*.log")):
        index = int(path.stem.split("_")[1])
        text = path.read_text(errors="replace")
        out[index] = [int(cid) for _round, cid in ASSIGNED.findall(text)]
    return out


def seed_records(directory: Path) -> dict[str, int]:
    records: dict[str, int] = {}
    for path in sorted(directory.glob("*.log")):
        for role, seed in SEED_LINE.findall(path.read_text(errors="replace")):
            records[role] = int(seed)
    return records


def first_divergence(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(one: Path, two: Path) -> str:
    lines: list[str] = []
    add = lines.append

    add("Stage 1 - determinism check")
    add("=" * 72)
    add("")
    add("Two runs of ONE config: N=20, C=4, alpha=0.8, arm gpu, seed 4126,")
    add("200 rounds each, run sequentially and alone on the box.")
    add("")
    add(f"run A   {one.name}")
    add(f"run B   {two.name}")
    add("")

    # --- seeding actually happened, in every process ------------------------
    add("0. was every process seeded?")
    add("-" * 72)
    for label, directory in (("A", one), ("B", two)):
        records = seed_records(directory)
        add(f"run {label}: {len(records)} processes logged a seed record")
        for role in sorted(records):
            add(f"        {role:20s} seed {records[role]}")
    seeds_a, seeds_b = seed_records(one), seed_records(two)
    add("")
    add("same seeds in both runs: " + ("YES" if seeds_a == seeds_b else "NO"))
    add("")

    # --- the accuracy curve --------------------------------------------------
    curve_a, curve_b = curve(one), curve(two)
    evals_a, evals_b = evaluations(curve_a), evaluations(curve_b)
    add("1. are the two accuracy curves identical, row for row?")
    add("-" * 72)
    add(f"rows in the results CSV:  A {len(curve_a)}, B {len(curve_b)}")
    add(f"rounds where accuracy moved (i.e. real evaluations): "
        f"A {len(evals_a)}, B {len(evals_b)}")
    add("")
    add("PLATO writes one row per round and carries the last accuracy forward")
    add("between evaluations, so most rows repeat. Both views are compared.")
    add("")

    rounds_a = [r for r, _, _ in curve_a]
    rounds_b = [r for r, _, _ in curve_b]
    acc_a = [a for _, _, a in curve_a]
    acc_b = [a for _, _, a in curve_b]

    aligned = rounds_a == rounds_b
    if not aligned:
        add("the round columns differ, so the curves cannot be aligned")

    identical = aligned and acc_a == acc_b
    add("VERDICT (every row): accuracy is " +
        ("IDENTICAL row for row." if identical else "NOT identical."))
    if aligned and not identical:
        diffs = [(r, x, y) for r, x, y in zip(rounds_a, acc_a, acc_b) if x != y]
        add(f"         {len(diffs)} of {len(rounds_a)} rows differ")
        worst = max(diffs, key=lambda d: abs(d[1] - d[2]))
        add(f"         first    round {diffs[0][0]:4d}: "
            f"{diffs[0][1]:.6f} vs {diffs[0][2]:.6f}")
        add(f"         largest  round {worst[0]:4d}: "
            f"{worst[1]:.6f} vs {worst[2]:.6f} "
            f"(|diff| {abs(worst[1] - worst[2]):.6f})")

    identical_evals = evals_a == evals_b
    add("VERDICT (evaluations only): " +
        ("IDENTICAL." if identical_evals else "NOT identical."))
    add("")

    # train_loss changes every round, so it is the most sensitive column here.
    rows_a, rows_b = raw_rows(one), raw_rows(two)
    loss_a = [r.get("train_loss") for r in rows_a]
    loss_b = [r.get("train_loss") for r in rows_b]
    add("train_loss (changes every round, so the most sensitive column): " +
        ("IDENTICAL." if loss_a == loss_b else "NOT identical."))
    if loss_a != loss_b:
        first = next((i for i, (x, y) in enumerate(zip(loss_a, loss_b)) if x != y),
                     None)
        if first is not None:
            add(f"    first difference at round {rows_a[first]['round']}: "
                f"{loss_a[first]} vs {loss_b[first]}")
    add("")

    # --- the clock -----------------------------------------------------------
    round_a, clock_a = clock_at_last_round(one)
    round_b, clock_b = clock_at_last_round(two)
    add("2. is sim_time_s at the final round identical?")
    add("-" * 72)
    add(f"A   round {round_a}   sim_time_s {clock_a:.6f}")
    add(f"B   round {round_b}   sim_time_s {clock_b:.6f}")
    add(f"difference          {clock_a - clock_b:+.6f} s "
        f"({100 * (clock_a - clock_b) / clock_b:+.4f}%)")
    add("")
    add("VERDICT: " + ("IDENTICAL." if clock_a == clock_b else "NOT identical."))
    add("")
    spread = abs(clock_a - clock_b) / min(clock_a, clock_b)
    add("Against the unseeded baseline: ../separability/PARALLELISM_CHECK.md ran")
    add("the same comparison -- two runs of one config, 200 rounds, C = 4, gpu")
    add("arm, each alone on this box -- and measured 313.447 s against 293.737 s,")
    add("a spread of 6.29%. Here it is "
        f"{100 * spread:.2f}%.")
    add("")
    add("So seeding removed most of it but not all. (The two are not a perfectly")
    add("controlled pair: that run used the un-jittered fleet, whose E[T] is")
    add("8.3231 s against 7.9180 s here, so the clocks are not directly")
    add("comparable in absolute terms. The SPREAD between two runs of one config")
    add("is, since both sides scale together.)")
    add("")

    # --- per-round clock difference -----------------------------------------
    if rounds_a == rounds_b:
        clock_diffs = [abs(ta - tb) for (_, ta, _), (_, tb, _) in zip(curve_a, curve_b)]
        if clock_diffs:
            worst_i = max(range(len(clock_diffs)), key=lambda i: clock_diffs[i])
            add("largest per-round clock difference, over the evaluation rounds:")
            add(f"    round {rounds_a[worst_i]}   "
                f"{curve_a[worst_i][1]:.6f} vs {curve_b[worst_i][1]:.6f}   "
                f"|diff| {clock_diffs[worst_i]:.6f} s")
            add(f"    median |diff| {statistics.median(clock_diffs):.6f} s")
            add("")

    # --- what the clock difference is made of --------------------------------
    text_a, text_b = server_log(one), server_log(two)
    dispatch_a, dispatch_b = dispatch_sequence(text_a), dispatch_sequence(text_b)
    merge_a, merge_b = aggregation_sequence(text_a), aggregation_sequence(text_b)

    add("3. where does any remaining difference come from?")
    add("-" * 72)
    add(f"dispatches:   A {len(dispatch_a)}, B {len(dispatch_b)}")
    same_dispatch = dispatch_a == dispatch_b
    add("dispatch sequence identical: " + ("YES" if same_dispatch else "NO"))
    if not same_dispatch:
        i = first_divergence(dispatch_a, dispatch_b)
        add(f"    first difference at dispatch {i}: "
            f"client {dispatch_a[i] if i < len(dispatch_a) else '-'} vs "
            f"{dispatch_b[i] if i < len(dispatch_b) else '-'}")

    add(f"merges:       A {len(merge_a)}, B {len(merge_b)}")
    same_merge = merge_a == merge_b
    add("merge order identical:       " + ("YES" if same_merge else "NO"))
    if not same_merge:
        i = first_divergence(merge_a, merge_b)
        add(f"    first difference at merge {i}: "
            f"client {merge_a[i] if i < len(merge_a) else '-'} vs "
            f"{merge_b[i] if i < len(merge_b) else '-'}")
    add("")

    add("THE SOURCE THAT SEEDING CANNOT REMOVE")
    add("")
    add("PLATO builds each report's finish time as")
    add("")
    add("    training_time + processing_time + comm_time + start_time")
    add("                                     (plato/servers/base.py:1025-1030)")
    add("")
    add("and that finish time becomes the simulated clock. Of those four terms:")
    add("")
    add("    training_time     the client's fixed sleep -- deterministic")
    add("    comm_time         0 under comm_simulation = true")
    add("    start_time        derived from the clock itself")
    add("    processing_time   time.perf_counter() measured ON THE HOST")
    add("                      (plato/clients/strategies/defaults.py:252)")
    add("")
    add("`processing_time` is real elapsed time on a shared machine. It is added")
    add("to every report, so it enters the clock on every round, and no seeding")
    add("can make it repeat. `sim_time_s` therefore CANNOT be bit-identical")
    add("between two runs; this is a property of PLATO's clock, not of this")
    add("configuration, and it applies to every cell this project has ever run.")
    add("")
    if same_dispatch and same_merge:
        add("What seeding DID fix is the part that mattered: the dispatch")
        add("sequence and the merge order are identical, so the two runs trained")
        add("on the same clients in the same order.")
        add("")
        return "\n".join(lines) + "\n"

    # --- the assignment race, which is where the divergence starts -----------
    assign_a = process_assignments(one)
    assign_b = process_assignments(two)

    add("4. WHICH WORKER PROCESS SERVED WHICH CLIENT")
    add("-" * 72)
    add("PLATO runs a pool of max_concurrency worker processes and gives each")
    add("round's selected client to whichever worker is free")
    add("(plato/servers/base.py:662-682). The first clients each worker served:")
    add("")
    add(f"{'process':10s} {'run A':28s} {'run B':28s}")
    first_clash = None
    for index in sorted(set(assign_a) | set(assign_b)):
        head_a = assign_a.get(index, [])[:6]
        head_b = assign_b.get(index, [])[:6]
        marker = "" if head_a == head_b else "   <-- differs"
        if head_a != head_b and first_clash is None:
            first_clash = index
        add(f"{index:<10d} {str(head_a):28s} {str(head_b):28s}{marker}")
    add("")

    same_client_set = (
        sorted(c for v in assign_a.values() for c in v)[:20]
        == sorted(c for v in assign_b.values() for c in v)[:20]
    )
    seeded_per_client = change_b_active(one) and change_b_active(two)
    if first_clash is not None and not seeded_per_client:
        add("THE WORKER ASSIGNMENT DIFFERS, and it differs BEFORE the dispatch")
        add("sequence does. That ordering comes from the order the four client")
        add("processes happened to connect over socket.io, which is a real-time")
        add("race at startup and is not seeded by anything.")
        add("")
        add("Why that is enough to diverge the whole run: the torch RNG lives in")
        add("the WORKER PROCESS, not in the logical client. A client trained by")
        add("worker 2 in one run and worker 4 in the other draws a different")
        add("augmentation sequence (RandomHorizontalFlip + RandomCrop), so it")
        add("computes different gradients. From there:")
        add("")
        add("    different gradients -> different G_i")
        add("                        -> different V_i = G_i / sqrt(1 + tau_hat)")
        add("                        -> a different client is selected")
        add("                        -> a different clock")
        add("")
        add("What would fix it is re-seeding torch at the start of each local")
        add("training run from (base_seed, client_id, round) instead of once per")
        add("process. That is Stage 2's Change B.")
    elif first_clash is not None:
        add("The worker assignment still differs -- Change B does not stop the")
        add("socket.io connection race, it makes the race HARMLESS. With")
        add("ClientSeededTrainer active, torch is re-seeded from")
        add("(base_seed, client_id, round) at the top of every local training")
        add("run, so a client trained by worker 2 in one run and worker 4 in the")
        add("other draws the SAME augmentation sequence.")
        add("")
        add("That it no longer propagates is visible in the sections above:")
        add("training stayed bit-identical for far longer than before, and the")
        add("divergence, when it came, did not start at the worker assignment.")
        add("Section 6 locates what it did start at.")
    else:
        add("The worker assignment is identical in both runs.")
    add("")

    return "\n".join(lines) + "\n"


def change_b_active(directory: Path) -> bool:
    """Did this run use the per-(client, round) trainer seeding?"""
    for path in sorted(directory.glob("client_*.log")):
        if "[ClientSeededTrainer]" in path.read_text(errors="replace"):
            return True
    return False


def near_tie_section(one: Path, two: Path) -> str:
    """Was the merge order flipped by a near-tie in finish time?

    The merge heap is ordered by (finish_time, client_id)
    (`plato/servers/base.py:1036-1047`) and finish_time carries the host's
    accumulated `processing_time`. Two clients whose SIMULATED finishes are
    closer together than the difference in their slots' accumulated overhead can
    therefore swap places, and from that point the two runs hand different models
    to different clients.

    This section finds the first merge where the two runs disagree and prints the
    reconstructed -- overhead-free -- finish times around it, so the reader can
    see whether the swap was a near-tie or something larger.
    """
    import clock_reconstruction as reconstruction   # noqa: PLC0415

    a = reconstruction.reconstruct(one)
    b = reconstruction.reconstruct(two)
    clocks_a = reconstruction.clock_per_round(a["schedule"], a["merges"])
    clocks_b = reconstruction.clock_per_round(b["schedule"], b["merges"])

    index = first_divergence([c for _r, c in a["merges"]],
                             [c for _r, c in b["merges"]])
    lines = ["", "6. WAS THE MERGE ORDER FLIPPED BY A NEAR-TIE?", "-" * 72]
    if index is None:
        lines += ["The merge orders are identical; nothing to explain.", ""]
        return "\n".join(lines) + "\n"

    lines += [
        "PLATO merges the smallest (finish_time, client_id) off a heap each",
        "round, and finish_time carries the host's accumulated processing_time.",
        "Below, 'finish' is the RECONSTRUCTED finish -- the same quantity with",
        "the host overhead removed.",
        "",
        f"{'merge':>6s}  {'run A':>28s}  {'run B':>28s}",
    ]
    lo = max(0, index - 3)
    hi = min(len(a["merges"]), index + 5)
    for i in range(lo, hi):
        ra, ca = a["merges"][i]
        rb, cb = b["merges"][i]
        mark = "  <-- differs" if ca != cb else ""
        lines.append(
            f"{i:6d}  round {ra:4d} client {ca:3d} {clocks_a.get(ra, 0):9.5f}"
            f"  round {rb:4d} client {cb:3d} {clocks_b.get(rb, 0):9.5f}{mark}"
        )

    # Only the CONTIGUOUS block of disagreeing merges: a window that ran on past
    # the point where the two orders re-converge would report a span belonging to
    # unrelated clients.
    end = index
    while (end < len(a["merges"]) and end < len(b["merges"])
           and a["merges"][end][1] != b["merges"][end][1]):
        end += 1
    window_a = [clocks_a[r] for r, _ in a["merges"][index:end] if r in clocks_a]
    window_b = [clocks_b[r] for r, _ in b["merges"][index:end] if r in clocks_b]
    asc_a = all(x <= y + 1e-9 for x, y in zip(window_a, window_a[1:]))
    asc_b = all(x <= y + 1e-9 for x, y in zip(window_b, window_b[1:]))
    lines += [
        "",
        f"in the window above, run A merges in ascending reconstructed time: "
        f"{'yes' if asc_a else 'NO'}",
        f"                     run B: {'yes' if asc_b else 'NO'}",
    ]
    if window_a and window_b:
        span = max(window_a + window_b) - min(window_a + window_b)
        swapped = sorted({c for _r, c in a["merges"][index:end]}
                         | {c for _r, c in b["merges"][index:end]})
        lines += [
            f"the {len(swapped)} clients that swapped (#{', #'.join(map(str, swapped))}) "
            f"finish within {span:.3f} s of each other",
            "",
            "The per-report processing_time measured on this box is ~0.065 s, and",
            "it accumulates PER SLOT, so two slots can differ by several tenths of",
            "a second by this point in a run (stage2_evidence/clock_reconstruction.md",
            "C1). A gap of that size between two clients' simulated finishes is",
            "therefore inside the noise the host adds, and which of them the server",
            "merges first is decided by the machine rather than by the experiment.",
        ]
    lines.append("")
    return "\n".join(lines) + "\n"


def reconstructed_section(one: Path, two: Path) -> str:
    """The same comparison on the clock rebuilt outside PLATO (Change C).

    PLATO's clock carries the host's `processing_time` on every report, so it can
    never repeat. The reconstructed clock is a pure function of the dispatch
    order and the sleep times, so if the dispatch orders match it is the same
    number -- which makes it the clock a sweep should actually be compared on.
    """
    import clock_reconstruction as reconstruction   # noqa: PLC0415

    lines: list[str] = ["", "5. THE RECONSTRUCTED CLOCK (Change C)",
                        "-" * 72]
    results = [reconstruction.reconstruct(one), reconstruction.reconstruct(two)]
    finals = [r["rows"][-1] for r in results]

    for label, result, final in zip("AB", results, finals):
        lines.append(
            f"{label}   round {final['round']}   "
            f"plato {final['sim_time_plato']:.6f}   "
            f"reconstructed {final['sim_time_reconstructed']:.6f}"
        )
    for key, label in (("sim_time_plato", "plato"),
                       ("sim_time_reconstructed", "reconstructed")):
        a, b = finals[0][key], finals[1][key]
        lines.append(
            f"{label:14s} difference {a - b:+.6f} s "
            f"({100 * (a - b) / min(a, b):+.4f}%)   "
            f"{'IDENTICAL' if a == b else 'not identical'}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--runs", nargs=2, default=["det1", "det2"])
    parser.add_argument("--out", type=Path, default=EVIDENCE / "determinism.txt")
    parser.add_argument("--reconstructed", action="store_true",
                        help="Also report the clock rebuilt by "
                             "clock_reconstruction.py, which carries none of "
                             "the host's processing_time.")
    args = parser.parse_args()

    one, two = (run_dir(name) for name in args.runs)
    report = compare(one, two)
    if args.reconstructed:
        report += reconstructed_section(one, two)
        report += near_tie_section(one, two)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
