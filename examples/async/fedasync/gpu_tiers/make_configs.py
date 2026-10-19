"""
Render the run configs for the heterogeneity sweep from templates/.

Three arms x nine gaps = 27 TOML files in configs/:

    baseline_xa<A>_gap<G>.toml   fedasync_gpu_tier.py            port 8300 + G
    gpu_xa<A>_gap<G>.toml        fedasync_gpu_tier_polaris_vi.py port 8100 + G
    cpu_xa<A>_gap<G>.toml        fedasync_gpu_tier_polaris_vi.py port 8200 + G

Substitutes {{GAP}}, {{ANCHOR}} and {{PORT}}, then refuses to write anything
unless every rendered file parses as TOML, every path it references exists, and
no two configs share a port.  Edit the templates, never configs/ — this script
overwrites the whole directory on every run.

Usage:
    python make_configs.py                     # anchor 20, all nine gaps
    python make_configs.py --anchor 40
    python make_configs.py --gaps 01 20 40
    python make_configs.py --check             # validate only, write nothing
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tomllib
from pathlib import Path

_HERE = Path(__file__).parent
TEMPLATE_DIR = _HERE / "templates"
OUTPUT_DIR = _HERE / "configs"

# Five gaps, not nine: the sweep has to finish inside a day, and 01/10/20/30/40
# already spans the whole effective range (1.0 -> 8.8) with the mismatch = 1
# crossing between 01 and 10 still inside it.  Pass --gaps to get the rest back.
DEFAULT_GAPS = ["01", "10", "20", "30", "40"]
ALL_GAPS = ["01", "05", "10", "15", "20", "25", "30", "35", "40"]

# Every path in the rendered configs is absolute, so a second machine only needs
# --root pointing at its own copy of the example directory.
DEFAULT_ROOT = _HERE.parent

# arm -> (template, output prefix, port base, entry point)
ARMS = {
    "baseline": (
        "fedasync_gpu_tier_baseline.toml.tmpl",
        "baseline",
        8300,
        "fedasync_gpu_tier.py",
    ),
    "gpu": (
        "fedasync_polaris_vi_gpu_arm.toml.tmpl",
        "gpu",
        8100,
        "fedasync_gpu_tier_polaris_vi.py",
    ),
    "cpu": (
        "fedasync_polaris_vi_cpu_arm.toml.tmpl",
        "cpu",
        8200,
        "fedasync_gpu_tier_polaris_vi.py",
    ),
    # The reactive arm: Polaris's observed-tau estimator (surgical/ADR-0002).
    # Same entry point as gpu/cpu — only [server] staleness_predictor differs.
    # It has no template in gpu_tiers/templates/; it is used by the surgical
    # experiment, which passes --templates surgical/templates.
    "observed": (
        "fedasync_polaris_vi_observed_arm.toml.tmpl",
        "observed",
        8400,
        "fedasync_gpu_tier_polaris_vi.py",
    ),
}

# arm -> the ONE value [server] staleness_predictor may hold, or None for an arm
# that must not carry the key at all.
#
# This is what makes an arm that arm, and it is the only semantic difference
# between the gpu, cpu and observed configs.  It is checked here, at generation
# time, because the runtime read used to fall back silently: an unrecognised
# value became the observed-tau predictor, so a typo in the gpu or cpu config
# would produce a run behaviourally identical to the observed ARM of the same
# sweep, with nothing in the results to reveal the duplicate
# (fedasync_polaris_vi_server.resolve_staleness_predictor, which now raises, is
# the runtime backstop for hand-written configs).
#
# baseline maps to None on purpose: fedasync_gpu_tier.py never reads the key, so
# writing it there would be a dead key implying an effect that does not exist
# (HANDOFF § 4.7).
ARM_PREDICTOR: dict[str, str | None] = {
    "baseline": None,
    "gpu": "gpu_tops",
    "cpu": "cpu",
    "observed": "observed",
}

# Keys that must be identical across all three arms at the same gap, or the runs
# are not comparable.  Anything genuinely arm-specific is listed in ARM_SPECIFIC.
SHARED_KEYS = [
    ("clients", "total_clients"), ("clients", "per_round"),
    ("clients", "speed_simulation"), ("clients", "comm_simulation"),
    ("clients", "sleep_simulation"), ("clients", "avg_training_time"),
    ("clients", "max_sleep_time"), ("clients", "simulation_distribution"),
    ("server", "staleness_bound"), ("server", "minimum_clients_aggregated"),
    ("server", "mixing_hyperparameter"), ("server", "adaptive_mixing"),
    ("server", "eval_interval"), ("server", "staleness_weighting_function"),
    ("server", "simulate_wall_time"), ("server", "synchronous"),
    ("data", "partition_size"), ("data", "sampler"), ("data", "concentration"),
    ("data", "random_seed"), ("data", "dataset_name"),
    ("trainer", "rounds"), ("trainer", "target_accuracy"), ("trainer", "epochs"),
    ("trainer", "batch_size"), ("trainer", "model_name"), ("trainer", "max_concurrency"),
]
ARM_SPECIFIC = ("base_path", "port", "staleness_predictor", "gpu_tier")

# [gpu_tier] as a whole is arm-specific — swapping tiers_csv is what MAKES the cpu
# arm the cpu arm.  But not every key inside it is: these describe the measurement
# apparatus, not an arm's belief, and must match or the arms land on different
# staleness scales.  They need their own check precisely because the table they
# live in is exempt from SHARED_KEYS.
#
#   anchor_tier / anchor_sleep : must equal what generate_sleep_times.py was given
#                                for the shared sleep CSV, or `residual` mixes units
#                                (HANDOFF § 4.1).
#   t_round_model              : which inter-merge-interval model the predictor uses.
#                                "min" and "queue" differ by up to 9x at gap01, so
#                                two arms on different models are not comparable at
#                                all (HANDOFF § 4.11).
#   comm_bandwidth_mbps /
#   payload_mbit               : the comm-sweep's C = D/B (BRIEF_comm_sweep.md).
#                                Differs ACROSS cells (that is the sweep's x-axis)
#                                but must be identical between the gpu and cpu arm
#                                of the SAME cell, same treatment as t_round_model.
#                                comm_in_predictor is deliberately NOT here: the
#                                belief-vs-reality ablation is a single gpu-only
#                                cell that intentionally sets it apart from its own
#                                sibling, not something compared across arms.
SHARED_GPU_TIER_KEYS = (
    "anchor_tier", "anchor_sleep", "t_round_model",
    "comm_bandwidth_mbps", "payload_mbit",
)

# Trailing bandwidth-cell tag on a mix/sleep-file label, e.g. "m50_35_15_b100" or
# the belief-ablation "m50_35_15_b010nc" -> "m50_35_15".  Bandwidth does not
# change the fleet composition, only the sleep values, so the comparability
# checks below must ignore it when matching a config's clients_csv against its
# sleep CSV.
_BW_SUFFIX_RE = re.compile(r"_b\d+[a-z]*$")


def _strip_bw_suffix(label: str) -> str:
    return _BW_SUFFIX_RE.sub("", label)


def port_for(arm: str, gap: str, seed_index: int, mix_index: int = 0, bw_index: int = 0) -> int:
    """A unique TCP port per (arm, gap, seed, mix, bandwidth-cell).

    Two servers on one port is a silent hang, not a crash, so the scheme has to
    have room for every run that could ever be in flight at once.

    Seed index 0 with no mix keeps the historical ports (`port_base + G`, bases
    8100/8200/8300 → 8101-8340) so a single-seed regeneration is byte-identical
    to what the running sweep already uses.  Every later seed is allocated from
    9000 upward, which is clear of that whole historical range — allocating them
    from 8000 instead would have put baseline's third seed on top of cpu's first.
    Each arm gets a 500-wide block, each seed a 100-wide slot inside it:

        baseline   9000 + 100*(seed_index-1) + G   ->  9001 ..  9440
        gpu        9500 + 100*(seed_index-1) + G   ->  9501 ..  9940
        cpu       10000 + 100*(seed_index-1) + G   -> 10001 .. 10440

    The mix sweep gets its own range from 11000 up, one 1000-wide block per arm
    and a 100-wide slot per mix, so a mix run can never land on a gap-sweep or
    multi-seed run even when both are in flight:

        baseline  11000 + 100*(mix_index-1) + G   -> 11001 .. 11940
        gpu       12000 + 100*(mix_index-1) + G   -> 12001 .. 12940
        cpu       13000 + 100*(mix_index-1) + G   -> 13001 .. 13940

    Mixes and multiple seeds together are refused rather than folded into one
    number: there is no spare dimension in these blocks, and a scheme invented
    here and never run is exactly how the 8000-block collision happened.

    The comm sweep gets its own range from 14000 up (BRIEF_comm_sweep.md § 4.6),
    clear of the mix sweep's 11000/12000/13000 even though both pin the fleet at
    m50_35_15 and can be in flight together:

        baseline  14000 + 100*(bw_index-1) + G   -> 14001 .. 14440
        gpu       15000 + 100*(bw_index-1) + G   -> 15001 .. 15440
        cpu       16000 + 100*(bw_index-1) + G   -> 16001 .. 16440

    Combined with a fixed mix (m50_35_15) and gap (G=30) this sweep needs only a
    handful of these slots; the 100-wide-per-bw-cell spacing is kept for the same
    reason the mix blocks are 100-wide — headroom, not a requirement.
    """
    _, _, port_base, _ = ARMS[arm]
    if bw_index:
        if seed_index or mix_index:
            raise SystemExit(
                "port_for(): a bandwidth sweep combined with multi-seed or "
                "multi-mix needs a fourth dimension in the port plan — widen "
                "the blocks here first."
            )
        if bw_index > 10:
            raise SystemExit(
                f"bandwidth index {bw_index} exceeds the 10 slots reserved per "
                f"arm; widen the blocks in port_for() before adding more cells."
            )
        block = {"baseline": 14000, "gpu": 15000, "cpu": 16000}[arm]
        return block + 100 * (bw_index - 1) + int(gap)
    if mix_index:
        if seed_index:
            raise SystemExit(
                "port_for(): a multi-seed AND multi-mix sweep needs a third "
                "dimension in the port plan — widen the blocks here first."
            )
        if mix_index > 10:
            raise SystemExit(
                f"mix index {mix_index} exceeds the 10 slots reserved per arm; "
                f"widen the blocks in port_for() before adding more mixes."
            )
        # observed sits at 21000, clear of cpu's multi-seed block (10000-10440)
        # and of every mix/comm block already in use.
        block = {"baseline": 11000, "gpu": 12000, "cpu": 13000, "observed": 21000}[arm]
        return block + 100 * (mix_index - 1) + int(gap)
    if seed_index == 0:
        return port_base + int(gap)
    if seed_index > 5:
        raise SystemExit(
            f"seed index {seed_index} exceeds the 5 slots reserved per arm; widen "
            f"the blocks in port_for() before adding more seeds."
        )
    block = {"baseline": 9000, "gpu": 9500, "cpu": 10000}[arm]
    return block + 100 * (seed_index - 1) + int(gap)


def render(
    arm: str,
    gap: str,
    anchor: int,
    root: Path,
    seed: int = 8888,
    seed_index: int = 0,
    rounds: int = 10000,
    runs_dir: str = "gap_sweep",
    multi_seed: bool = False,
    mix: str = "",
    mix_index: int = 0,
    t_round_model: str = "min",
    target_accuracy: float = 0.6,
    bw_index: int = 0,
    bw_label: str = "",
    run_label: str = "",
    comm_bandwidth_mbps: float | None = None,
    payload_mbit: float = 341.52,
    comm_in_predictor: bool = True,
    template_dir: Path | None = None,
    port_offset: int = 0,
) -> tuple[str, str, int]:
    """Return (filename, rendered text, port) for one arm/gap/seed/mix/bw combination.

    ``multi_seed`` controls only whether the seed appears in the run's name and
    config filename.  With a single seed it stays off, so the rendered files are
    byte-identical to the pre-seed layout and the sweep already in flight keeps
    reading exactly the configs it started with.

    ``mix`` is a fleet-composition label from make_mixes.py (``m70_20_10``) or
    empty for the historical 50/35/15 fleet.  It selects both the clients file
    and the sleep file, which is why one placeholder feeds both — see the note in
    the templates.

    ``bw_label``/``bw_index``/``comm_bandwidth_mbps``/``payload_mbit`` are the
    comm-latency sweep's axis (BRIEF_comm_sweep.md).  ``bw_label`` (e.g.
    ``"b100"``) selects the REALITY sleep CSV, which depends only on the actual
    bandwidth.  ``run_label`` names the run/config identity and defaults to
    ``bw_label``; they differ only for the belief-vs-reality ablation, which
    reuses B=10's sleep CSV (reality is unchanged) under a distinct run identity
    (``comm_in_predictor=False`` changes BELIEF only).  Both empty (the default)
    keeps every existing sweep byte-identical: ``comm_bandwidth_mbps=None`` omits
    the whole comm block from ``[gpu_tier]``, so a rendered config carries no
    trace of this feature unless a bandwidth is actually requested.
    """
    template, prefix, _, _ = ARMS[arm]
    # port_offset shifts a whole sweep clear of every other sweep's block.  The
    # uniqueness check in validate() only sees one render at a time, so it cannot
    # catch a collision ACROSS sweeps; keeping each sweep in its own range is what
    # prevents "two servers on one port", which hangs silently rather than failing.
    port = port_for(arm, gap, seed_index, mix_index, bw_index) + port_offset
    suffix = f"_s{seed}" if multi_seed else ""
    mix_suffix = f"_{mix}" if mix else ""
    bw_suffix = f"_{bw_label}" if bw_label else ""
    run_bw_suffix = f"_{run_label}" if run_label else bw_suffix
    comm_block = ""
    if comm_bandwidth_mbps is not None:
        comm_block = (
            "\n# Communication latency C = D/B, added on top of compute time\n"
            "# (BRIEF_comm_sweep.md).  Uplink only — downlink is a base-station\n"
            "# broadcast at far higher transmit power and is conventionally\n"
            "# neglected in this literature.  Absent this table entirely -> C = 0,\n"
            "# byte-identical to every run made before this feature existed.\n"
            f"comm_bandwidth_mbps = {comm_bandwidth_mbps:g}\n"
            f"payload_mbit = {payload_mbit:g}\n"
            f"comm_in_predictor = {'true' if comm_in_predictor else 'false'}"
        )
    text = ((template_dir or TEMPLATE_DIR) / template).read_text()
    text = (
        text.replace("{{GAP}}", gap)
        .replace("{{ANCHOR}}", str(anchor))
        .replace("{{PORT}}", str(port))
        .replace("{{ROOT}}", str(root))
        .replace("{{SEED}}", str(seed))
        .replace("{{ROUNDS}}", str(rounds))
        .replace("{{RUNDIR}}", runs_dir)
        .replace("{{SEEDSFX}}", suffix)
        .replace("{{MIXSFX}}", mix_suffix)
        .replace("{{BWSFX}}", bw_suffix)
        .replace("{{RUNBWSFX}}", run_bw_suffix)
        .replace("{{TROUND}}", t_round_model)
        .replace("{{TARGETACC}}", f"{target_accuracy:g}")
    )
    # {{COMMBLOCK}} sits alone on its own line in the templates; substituting ""
    # would leave a blank line behind and break every existing sweep's
    # byte-identical-regeneration guarantee, so drop the whole line instead when
    # there is no comm block to insert.
    text = (
        text.replace("{{COMMBLOCK}}", comm_block)
        if comm_block
        else "\n".join(line for line in text.splitlines() if line != "{{COMMBLOCK}}") + "\n"
    )
    if "{{" in text:
        leftover = {tok.split("}}")[0] for tok in text.split("{{")[1:]}
        raise SystemExit(f"{template}: unsubstituted placeholder(s) {sorted(leftover)}")
    return f"{prefix}_xa{anchor}_gap{gap}{mix_suffix}{run_bw_suffix}{suffix}.toml", text, port


def referenced_paths(cfg: dict) -> list[str]:
    """Every filesystem path a rendered config depends on."""
    paths = [cfg["clients"]["simulation_distribution"]["csv_path"]]
    gpu_tier = cfg.get("gpu_tier", {})
    paths += [gpu_tier[k] for k in ("tiers_csv", "clients_csv") if k in gpu_tier]
    return paths


def mix_labels(cfg: dict) -> tuple[str, str]:
    """(label the clients file names, label the sleep file names).

    The fleet's composition is written in two places — who is in which tier
    (``[gpu_tier] clients_csv``) and how long each client sleeps
    (``[clients.simulation_distribution] csv_path``) — and nothing downstream
    checks that they describe the same fleet.  Crossing them is silent: the
    clients would sleep by one composition while the predictor believed another,
    and the run would look completely normal.
    """
    clients_stem = Path(cfg.get("gpu_tier", {}).get("clients_csv", "clients.csv")).stem
    from_clients = clients_stem.removeprefix("clients").lstrip("_")
    sleep_stem = Path(cfg["clients"]["simulation_distribution"]["csv_path"]).stem
    from_sleep = sleep_stem.split("_gap", 1)[-1][2:].lstrip("_") if "_gap" in sleep_stem else ""
    # A bandwidth-cell suffix on the sleep filename (e.g. "_b100") describes the
    # comm-sweep's REALITY axis, not the fleet composition — strip it before
    # comparing against the clients_csv-derived label (make_configs.py's render()
    # keeps the two independent so a bandwidth cell can reuse the same fleet).
    return from_clients, _strip_bw_suffix(from_sleep)


def validate(
    rendered: dict[str, tuple[str, dict, int]], check_paths: bool = True
) -> list[str]:
    """Return a list of problems; empty means safe to write.

    check_paths is off when generating for another machine (--root elsewhere):
    those paths describe that machine's filesystem, not this one.
    """
    problems: list[str] = []

    # Every referenced input must already exist (make_gap_sweep.sh generates them).
    if check_paths:
        for name, (_, cfg, _) in rendered.items():
            for path in referenced_paths(cfg):
                if not Path(path).exists():
                    problems.append(f"{name}: missing input {path}")

    # The two halves of the fleet description must name the same composition, and
    # it must be the one in the config's own filename.
    for name, (_, cfg, _) in rendered.items():
        from_clients, from_sleep = mix_labels(cfg)
        if from_clients != from_sleep:
            problems.append(
                f"{name}: clients_csv says mix {from_clients or '50_35_15 (base)'!r} "
                f"but the sleep CSV says {from_sleep or '50_35_15 (base)'!r}"
            )
        stem = name.removesuffix(".toml")
        # The fleet label is whatever sits between the gap digits and an optional
        # _s<seed>.  It used to be found by searching for "_m", which silently
        # returned "no label" for any tag that does not start with m — a fleet
        # SIZE tag (_n100) then read as the base fleet and the check passed a
        # config whose clients file described a different fleet entirely.
        tail = re.sub(r"_s\d+$", "", re.sub(r"^\d+", "", stem.split("_gap", 1)[-1]))
        in_name = _strip_bw_suffix(tail.lstrip("_"))
        if in_name != from_clients:
            problems.append(
                f"{name}: filename says mix {in_name or 'base'!r} but it reads "
                f"clients_csv for {from_clients or 'base'!r}"
            )

    # Ports must be unique — two servers on one port is a silent hang, not a crash.
    ports: dict[int, str] = {}
    for name, (_, cfg, port) in rendered.items():
        if port in ports:
            problems.append(f"{name}: port {port} already used by {ports[port]}")
        ports[port] = name

    # The predictor must match the arm the filename claims.  See ARM_PREDICTOR.
    for name, (_, cfg, _) in rendered.items():
        arm = name.split("_xa")[0]
        if arm not in ARM_PREDICTOR:
            problems.append(f"{name}: unknown arm {arm!r}")
            continue
        want = ARM_PREDICTOR[arm]
        got = cfg.get("server", {}).get("staleness_predictor")
        if want is None:
            if got is not None:
                problems.append(
                    f"{name}: the {arm} arm has no predictor, but the config "
                    f"sets staleness_predictor = {got!r} (HANDOFF § 4.7 — a dead "
                    "key implying an effect that does not exist)"
                )
        elif got is None:
            problems.append(
                f"{name}: the {arm} arm requires staleness_predictor = {want!r}, "
                "but the key is absent — it would silently run as 'observed'"
            )
        elif got != want:
            problems.append(
                f"{name}: staleness_predictor = {got!r}, but the {arm} arm "
                f"requires {want!r}"
            )

    # Comparability: the three arms at a given gap may differ only in ARM_SPECIFIC.
    # The grouping key is (gap, seed), not gap alone — arms are only comparable
    # within one seed, and `random_seed` is itself a SHARED_KEY, so grouping by gap
    # across a multi-seed sweep would report every seed as a violation of itself.
    by_gap: dict[str, dict[str, dict]] = {}
    for name, (_, cfg, _) in rendered.items():
        arm = name.split("_xa")[0]
        tail = name.rsplit("_gap", 1)[1].removesuffix(".toml")   # "20" or "20_s1234"
        by_gap.setdefault(tail, {})[arm] = cfg
    for gap, arms in by_gap.items():
        if "gpu" not in arms:
            continue
        ref = arms["gpu"]
        for arm, cfg in arms.items():
            for section, key in SHARED_KEYS:
                got, want = cfg.get(section, {}).get(key), ref.get(section, {}).get(key)
                if got != want:
                    problems.append(
                        f"gap{gap} {arm}: {section}.{key} = {got!r}, "
                        f"but gpu arm has {want!r}"
                    )
            if cfg["parameters"]["optimizer"] != ref["parameters"]["optimizer"]:
                problems.append(f"gap{gap} {arm}: parameters.optimizer differs from gpu arm")

            # Sub-keys of the otherwise arm-specific [gpu_tier] table that still
            # have to agree — see SHARED_GPU_TIER_KEYS.  Only between arms that
            # HAVE a predictor: the baseline selects randomly and never computes a
            # staleness, so anchoring and T_round do not exist for it, and § 4.7
            # is explicit that it must not carry dead keys implying otherwise.
            if "staleness_predictor" not in cfg.get("server", {}):
                continue
            for key in SHARED_GPU_TIER_KEYS:
                got = cfg.get("gpu_tier", {}).get(key)
                want = ref.get("gpu_tier", {}).get(key)
                if got != want:
                    problems.append(
                        f"gap{gap} {arm}: gpu_tier.{key} = {got!r}, "
                        f"but gpu arm has {want!r}"
                    )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--anchor", type=int, default=20)
    parser.add_argument("--gaps", nargs="+", default=DEFAULT_GAPS)
    parser.add_argument(
        "--all-gaps", action="store_true", help=f"Shorthand for --gaps {' '.join(ALL_GAPS)}"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Absolute path of the fedasync example directory on the machine that "
        "will RUN these configs (default: this checkout). Set it when generating "
        "configs for a second machine with a different layout.",
    )
    # Default stays the three legacy arms so the gap/multi-seed/mix/comm sweeps
    # regenerate byte-identically; "observed" is opt-in and has no blueprint in
    # gpu_tiers/templates (it belongs to the surgical experiment).
    parser.add_argument(
        "--arms", nargs="+", choices=sorted(ARMS),
        default=["baseline", "cpu", "gpu"],
    )
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--templates", type=Path, default=None,
        help="Directory of blueprints to render (default: gpu_tiers/templates). "
        "The surgical experiment keeps its own set, since it changes the fleet "
        "size, the partition and the clock model.",
    )
    parser.add_argument(
        "--port-offset", type=int, default=0,
        help="Shift every port by this much, to keep a sweep clear of the blocks "
        "already in use (gap 8100-8440, multi-seed 9000+, mix 11000+, comm 14000+). "
        "Default 0 keeps every existing sweep byte-identical.",
    )
    parser.add_argument("--check", action="store_true", help="Validate only; write nothing.")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[8888],
        help="One config set per seed.  With a single seed the output is "
        "byte-identical to the pre-seed layout (no seed in the filename or run "
        "name); with more than one, every run carries an _s<seed> suffix.",
    )
    parser.add_argument(
        "--rounds", type=int, default=10000,
        help="Rounds per run.  The slowest run in the seed-8888 sweep reached 55%% "
        "accuracy at round 3160, so 5000 covers every cell with margin at half the "
        "cost — which is what makes three seeds affordable (default: 10000).",
    )
    parser.add_argument(
        "--runs-dir", default="gap_sweep",
        help="Directory under runs/ that the configs write into.  Use a fresh name "
        "for a multi-seed sweep so nothing lands on top of an existing run.",
    )
    parser.add_argument(
        "--mixes", nargs="+", default=[],
        help="Fleet-composition labels from make_mixes.py (e.g. m50_35_15 "
        "m70_20_10).  Each becomes its own config per arm and gap, reading "
        "input/clients_<label>.csv and the matching sleep file.  Omit for the "
        "historical 50/35/15 fleet, whose configs then stay byte-identical.",
    )
    parser.add_argument(
        "--t-round-model", choices=("min", "queue"), default="min",
        help='Which inter-merge-interval model the predictor uses.  "min" is the '
        'historical (and wrong) default and is kept so the gap sweep regenerates '
        'byte-identically; "queue" is correct (HANDOFF § 4.11).  Runs made under '
        "different values are not comparable, so this is part of a sweep's identity.",
    )
    parser.add_argument(
        "--target-accuracy", type=float, default=0.6,
        help="Early stop.  It fires on one raw evaluation, so it truncates runs at "
        "a random moment rather than a comparable one (HANDOFF § 4.13); pass a "
        "value above 1.0 to disable it and give every run the same length "
        "(default: 0.6, which keeps the gap sweep byte-identical).",
    )
    args = parser.parse_args()
    gaps = ALL_GAPS if args.all_gaps else args.gaps
    root = args.root.resolve()

    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit(f"--seeds has duplicates: {args.seeds}")
    multi_seed = len(args.seeds) > 1
    if multi_seed and args.runs_dir == "gap_sweep":
        raise SystemExit(
            "Refusing to write a multi-seed sweep into runs/gap_sweep — that is the "
            "single-seed sweep's directory.  Pass --runs-dir (e.g. gap_sweep_ms)."
        )
    if len(set(args.mixes)) != len(args.mixes):
        raise SystemExit(f"--mixes has duplicates: {args.mixes}")
    if args.mixes and args.runs_dir == "gap_sweep":
        raise SystemExit(
            "Refusing to write a mix sweep into runs/gap_sweep — that is the "
            "gap sweep's directory.  Pass --runs-dir (e.g. mix_sweep)."
        )
    # Empty label = the historical fleet, and mix_index 0 = the historical ports.
    mixes = args.mixes or [""]

    rendered: dict[str, tuple[str, dict, int]] = {}
    combos = [
        (seed_index, seed, mix_index, mix)
        for seed_index, seed in enumerate(args.seeds)
        for mix_index, mix in enumerate(mixes, start=1 if args.mixes else 0)
    ]
    for seed_index, seed, mix_index, mix in combos:
        for arm in args.arms:
            for gap in gaps:
                name, text, port = render(
                    arm, gap, args.anchor, root, seed=seed, seed_index=seed_index,
                    rounds=args.rounds, runs_dir=args.runs_dir, multi_seed=multi_seed,
                    mix=mix, mix_index=mix_index, t_round_model=args.t_round_model,
                    target_accuracy=args.target_accuracy,
                    template_dir=args.templates, port_offset=args.port_offset,
                )
                try:
                    cfg = tomllib.loads(text)
                except tomllib.TOMLDecodeError as exc:
                    raise SystemExit(f"{name}: rendered template is not valid TOML — {exc}")
                rendered[name] = (text, cfg, port)

    local = root == DEFAULT_ROOT.resolve()
    problems = validate(rendered, check_paths=local)
    if problems:
        print(f"Refusing to write — {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    inputs_note = "all inputs exist" if local else f"paths rooted at {root} (not checked)"
    print(f"Validated {len(rendered)} configs (anchor {args.anchor} s, gaps {' '.join(gaps)}): "
          f"TOML parses, {inputs_note}, ports unique, arms comparable.")

    if args.check:
        print("--check given; nothing written.")
        return 0

    # Rewrite from scratch so a removed gap does not leave a stale config behind.
    if args.outdir.exists():
        shutil.rmtree(args.outdir)
    args.outdir.mkdir(parents=True)
    for name, (text, _, _) in sorted(rendered.items()):
        (args.outdir / name).write_text(text)

    print(f"\nWrote {len(rendered)} files to {args.outdir}/\n")
    print(f"{'arm':10}{'entry point':34}{'ports':16}{'files'}")
    for arm in args.arms:
        _, prefix, _, entry = ARMS[arm]
        names = [n for n in rendered if n.startswith(f"{prefix}_xa")]
        # Read the ports back out of `rendered` rather than recomputing them from
        # the block base: with --mixes or --seeds the base is not where they land.
        used = sorted(rendered[n][2] for n in names)
        span = f"{used[0]}-{used[-1]}" if used else "—"
        print(f"{arm:10}{entry:34}{span:16}{len(names)}")

    example = sorted(rendered)[0]
    arm = example.split("_xa")[0]
    entry = ARMS[arm][3]
    # An --outdir outside the example directory is legitimate (a scratch
    # comparison, another machine's tree); it just cannot be printed as a
    # relative path, and raising there would fail AFTER writing every file.
    try:
        rel = args.outdir.resolve().relative_to(_HERE.parent.resolve())
    except ValueError:
        rel = args.outdir.resolve()
    print(f"\nRun one with:\n  cd {_HERE.parent}\n"
          f"  PYTHONPATH=gpu_tiers python {entry} -c {rel}/{example}")
    if multi_seed:
        seeds = " ".join(str(s) for s in args.seeds)
        print(f"\nOr the whole sweep:\n"
              f"  cd {_HERE}\n"
              f"  CONFIG_DIR={args.outdir.resolve()} ./run_sweep.sh gpu cpu baseline "
              f'--seeds "{seeds}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
