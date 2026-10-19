"""
Checkpoint cleaner for the Wen-environment runs (Prompt C, Step 4b).

Why a new tool rather than `gpu_tiers/clean_checkpoints.sh`: that script
cleans ONE run by round number, keeps N rounds, and is started and killed per
run. Prompt C needs one long-lived process over every run, an mtime rule, "keep
only the newest checkpoint of a run that is still running", "delete everything
once a run has finished", and a deletion log.

What it may delete -- nothing else, ever:
  * files directly inside  <runs-root>/<run>/checkpoints/
  * whose name ends in one of CHECKPOINT_EXTENSIONS
    (.safetensors, .pth, .pt, .pkl -- PLATO's global/client weights, optimizer
     and PRNG state, and the per-client model payloads)

Never touched: logs, dispatch/merge/cycle/pool logs, results/, models/,
manifests, evidence/, configs, the fleet, the SUMO bundle -- anything not inside
a `checkpoints` directory under the runs root. Every candidate path is checked
against that rule again immediately before unlinking.

Rules per pass:
  * a run is RUNNING if any live process's command line is wen_fedasync.py with
    a config whose [general] base_path is that run's directory;
  * running run: only ROUND-INDEXED families may go (ROUND_FAMILIES: the
    per-round model checkpoint and its sidecar, the two PRNG states); keep the
    newest file of each, and delete the rest only if older than --min-age
    (default 10 min). The mtime is checked again immediately before unlink.
    Per-client model payloads (`resnet_18_<cid>.pkl`, `resnet_18_client_<cid>.pkl`)
    and `current_round.pkl` are NEVER deleted while the run is alive: on
    16 Oct 2026 the first version deleted `resnet_18_19.pkl` in the same second
    the server rewrote it for a dispatch (stat-then-unlink race), the client
    could not load its model, and S1_gpu_oracle_s4126 hung at round 1,501.
    Those files are bounded (2 per client id, ~2.6 GB per run) and are removed
    with everything else once the run finishes;
  * finished run: delete every checkpoint file;
  * a run directory or file that disappears mid-pass is skipped, never fatal,
    and a failed pass is retried at the next interval: on 16 Oct an uncaught
    FileNotFoundError (a run directory archived during a pass) killed the
    cleaner, and the runs then filled the 200 GB quota;
  * one line per deletion (time, path, bytes) appended to the log.

Usage (from examples/async/fedasync):
  python wen_env/tools/ckpt_cleaner.py --dry-run --once
  nohup python wen_env/tools/ckpt_cleaner.py > wen_env/logs/ckpt_cleaner.out 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import time
import tomllib
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parents[1]
DEFAULT_ROOT = FEDASYNC / "runs" / "wen_env"
DEFAULT_LOG = FEDASYNC / "wen_env" / "logs" / "ckpt_cleaner.log"
CHECKPOINT_DIRNAME = "checkpoints"
CHECKPOINT_EXTENSIONS = (".safetensors", ".pth", ".pt", ".pkl")
# Families a RUNNING run may lose (digits replaced by '#'); see the docstring.
ROUND_FAMILIES = re.compile(
    r"^(checkpoint_.*_#\.safetensors(\.pkl)?|prng_state_#\.pkl|numpy_prng_state_#\.pkl)$")
ENTRY = "wen_fedasync.py"


def running_run_dirs() -> set[Path]:
    """Run directories whose server or client processes are alive."""
    live: set[Path] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            cwd = Path(os.readlink(proc / "cwd"))
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv if a]
        if not any(a.endswith(ENTRY) for a in argv) or "-c" not in argv:
            continue
        cfg = Path(argv[argv.index("-c") + 1])
        cfg = cfg if cfg.is_absolute() else cwd / cfg
        try:
            base = tomllib.loads(cfg.read_text())["general"]["base_path"]
        except (OSError, KeyError, tomllib.TOMLDecodeError):
            continue
        base = Path(base)
        live.add((base if base.is_absolute() else cwd / base).resolve())
    return live


def family(name: str) -> str:
    return re.sub(r"\d+", "#", name)


def is_deletable(path: Path, root: Path) -> bool:
    """The only test that lets a file be unlinked."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return (path.is_file() and not path.is_symlink()
            and path.parent.name == CHECKPOINT_DIRNAME
            and path.parent.parent.parent.resolve() == root.resolve()
            and path.name.endswith(CHECKPOINT_EXTENSIONS))


def _plan_run(ckpt_dir: Path, root: Path, live: set[Path], min_age_s: float,
              now: float) -> list[tuple[Path, int, str]]:
    run_dir = ckpt_dir.parent.resolve()
    files = [p for p in ckpt_dir.iterdir() if is_deletable(p, root)]
    if run_dir not in live:
        return [(p, p.stat().st_size, "run finished") for p in files]

    files = [p for p in files if ROUND_FAMILIES.match(family(p.name))]
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        groups[family(p.name)].append(p)
    newest = {fam: max(members, key=lambda p: p.stat().st_mtime)
              for fam, members in groups.items()}
    out = []
    for p in files:
        st = p.stat()
        if p == newest[family(p.name)] or now - st.st_mtime < min_age_s:
            continue
        out.append((p, st.st_size, "running, superseded, older than min-age"))
    return out


def plan(root: Path, min_age_s: float, now: float) -> list[tuple[Path, int, str]]:
    """(path, bytes, reason) for every file this pass would delete."""
    live = running_run_dirs()
    out: list[tuple[Path, int, str]] = []
    for ckpt_dir in sorted(root.glob(f"*/{CHECKPOINT_DIRNAME}")):
        try:
            out += _plan_run(ckpt_dir, root, live, min_age_s, now)
        except OSError:
            continue        # vanished mid-pass; see the docstring
    return out


def one_pass(root: Path, log: Path, min_age_s: float, dry_run: bool) -> tuple[int, int]:
    victims = plan(root, min_age_s, time.time())
    freed = 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        for path, size, reason in victims:
            stamp = dt.datetime.now().isoformat(timespec="seconds")
            if dry_run:
                print(f"{stamp} DRY-RUN would delete {path} ({size} B; {reason})")
                continue
            try:
                if not is_deletable(path, root):      # re-check right before unlink
                    continue
                if (reason != "run finished"
                        and time.time() - path.stat().st_mtime < min_age_s):
                    continue                          # rewritten since the plan
                path.unlink()
            except OSError:
                continue
            freed += size
            fh.write(f"{stamp}\t{path}\t{size}\t{reason}\n")
    return len(victims), freed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--min-age", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    print(f"[ckpt_cleaner] pid={os.getpid()} root={args.root} "
          f"patterns={[f'*/{CHECKPOINT_DIRNAME}/*{e}' for e in CHECKPOINT_EXTENSIONS]} "
          f"running-run families={ROUND_FAMILIES.pattern} "
          f"min_age={args.min_age:g}s interval={args.interval:g}s dry_run={args.dry_run}",
          flush=True)
    total = 0
    while True:
        try:
            n, freed = one_pass(args.root, args.log, args.min_age, args.dry_run)
            total += freed
            print(f"[ckpt_cleaner] {dt.datetime.now().isoformat(timespec='seconds')} "
                  f"pass: {n} file(s), freed {freed / 2**30:.2f} GiB, "
                  f"total {total / 2**30:.2f} GiB", flush=True)
        except Exception:                             # noqa: BLE001
            # Never die: a dead cleaner is how the quota filled on 16 Oct.
            print(f"[ckpt_cleaner] pass failed, retrying next interval:\n"
                  f"{traceback.format_exc()}", flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
