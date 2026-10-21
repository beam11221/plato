"""
Per-run manifest (Prompt C, Step 5): `<base_path>/results/manifest.yaml`.

Records what a run was: config sha256, fleet, SUMO bundle manifest sha256, seed,
scenario / arm / rule, and the git commit of the locked predictions
(`examples/async/fedasync/docs/predictions_v3.md`).

The predictions must be committed before any FULL run (Prompt C: "copy them
into docs/predictions_v3.md before any full run, commit, and record the commit
hash in every run manifest"). A run with `rounds >= FULL_RUN_ROUNDS` therefore
refuses to start without that commit; shorter (smoke / check) runs record
`predictions_commit: null`.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FEDASYNC = HERE.parent
PREDICTIONS = FEDASYNC / "docs" / "predictions_v3.md"
# Prompt D: no cpu_cal run (of any length) before this addendum is committed.
ADDENDUM_V3_1 = FEDASYNC / "docs" / "predictions_v3_1_addendum.md"
ADDENDUM_ARMS = frozenset({"cpu_cal"})
# Prompt E: no wen run (of any length) before the v3.2 addendum is committed.
ADDENDUM_V3_2 = FEDASYNC / "docs" / "predictions_v3_2_addendum.md"
# Prompt H: no wen_fresh run (of any length) before the v3.3 addendum is committed.
ADDENDUM_V3_3 = FEDASYNC / "docs" / "predictions_v3_3_addendum.md"
# Prompt I: no Study 2 run (of any length) before the v3.4 addendum is committed.
ADDENDUM_V3_4 = FEDASYNC / "docs" / "predictions_v3_4_addendum.md"
STUDY2_ARMS = frozenset({"random_s0", "wen_fresh_s0", "wen_fresh_gpu", "wen_fresh_gpu_s0"})
# Prompt J: no polaris run (of any length) before the v3.5 addendum is committed.
ADDENDUM_V3_5 = FEDASYNC / "docs" / "predictions_v3_5_addendum.md"
FULL_RUN_ROUNDS = 4000
REPO = FEDASYNC.parents[2]
# Prompt I Step 0b.1: the code a run executes, hashed into every manifest.
SOURCE_GLOBS = ("plato/**/*.py", "examples/async/fedasync/*.py",
                "examples/async/fedasync/wen_env/*.py",
                "examples/async/fedasync/gpu_tiers/*.py",
                "examples/async/fedasync/gpu_tiers_dataset/*.py",
                "examples/async/fedasync/sumo_close_loop/*.py")


class PredictionsNotLockedError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_tree() -> dict:
    """sha256 over (path, sha256) of every source file in SOURCE_GLOBS, sorted by
    path, plus the git HEAD and whether any of those files differ from it."""
    files = sorted({f for g in SOURCE_GLOBS for f in REPO.glob(g) if f.is_file()})
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(REPO).as_posix()}\0{_sha256(f)}\n".encode())
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--",
                                 *(f":(glob){g}" for g in SOURCE_GLOBS)],
                                cwd=REPO, capture_output=True, text=True,
                                check=True).stdout.splitlines()
        # A file deleted from the working tree is not hashed, so it is not "dirty".
        dirty = [line for line in status if "D" not in line[:2]]
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, "unknown"
    return {"source_tree_sha256": h.hexdigest(), "source_files": len(files),
            "source_commit": head, "source_dirty": bool(dirty)}


def predictions_commit(path: Path = PREDICTIONS) -> str | None:
    """Hash of the last commit touching `path`, if it is committed and the
    working copy matches it."""
    if not path.exists():
        return None
    repo = FEDASYNC
    try:
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", str(path)],
            cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not commit or dirty:
        return None
    return commit


def write(base_path: Path, config_path: Path, settings: dict, rounds: int,
          seed: int) -> dict:
    commit = predictions_commit()
    addendum = None
    if settings["arm"] in ADDENDUM_ARMS:
        addendum = predictions_commit(ADDENDUM_V3_1)
        if addendum is None:
            raise PredictionsNotLockedError(
                f"arm {settings['arm']} refused: {ADDENDUM_V3_1} is missing, "
                "uncommitted or modified (Prompt D Step 0).")
    addendum_v3_2 = None
    if settings["arm"] == "wen":
        addendum_v3_2 = predictions_commit(ADDENDUM_V3_2)
        if addendum_v3_2 is None:
            raise PredictionsNotLockedError(
                f"arm wen refused: {ADDENDUM_V3_2} is missing, uncommitted or "
                "modified (Prompt E Step 0).")
    addendum_v3_3 = None
    if settings.get("host", settings["arm"]) == "wen_fresh":
        addendum_v3_3 = predictions_commit(ADDENDUM_V3_3)
        if addendum_v3_3 is None:
            raise PredictionsNotLockedError(
                f"arm wen_fresh refused: {ADDENDUM_V3_3} is missing, uncommitted or "
                "modified (Prompt H Step 1).")
    addendum_v3_4 = None
    if settings["arm"] in STUDY2_ARMS:
        addendum_v3_4 = predictions_commit(ADDENDUM_V3_4)
        if addendum_v3_4 is None:
            raise PredictionsNotLockedError(
                f"arm {settings['arm']} refused: {ADDENDUM_V3_4} is missing, uncommitted "
                "or modified (Prompt I Step 1).")
    addendum_v3_5 = None
    if settings["arm"] == "polaris":
        addendum_v3_5 = predictions_commit(ADDENDUM_V3_5)
        if addendum_v3_5 is None:
            raise PredictionsNotLockedError(
                f"arm polaris refused: {ADDENDUM_V3_5} is missing, uncommitted or "
                "modified (Prompt J Step 1).")
    if rounds >= FULL_RUN_ROUNDS and commit is None:
        raise PredictionsNotLockedError(
            f"{rounds}-round run refused: {PREDICTIONS} is missing, uncommitted or "
            "modified. Copy predictions v3 there and commit before any full run.")
    manifest = {
        "exp_id": Path(base_path).name,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "fleet": Path(settings["fleet_dir"]).name,
        "bundle_manifest_sha256": _sha256(Path(settings["bundle_dir"]) / "manifest.yaml"),
        "predictions_commit": commit,
        "predictions_v3_1_addendum_commit": addendum,
        "predictions_v3_2_addendum_commit": addendum_v3_2,
        "predictions_v3_3_addendum_commit": addendum_v3_3,
        "predictions_v3_4_addendum_commit": addendum_v3_4,
        "predictions_v3_5_addendum_commit": addendum_v3_5,
        "wen_t_max": settings.get("wen_t_max"),
        "wen_e_max": settings.get("wen_e_max"),
        "cpu_rho": settings.get("cpu_rho"),
        "cpu_rho_test_override": settings.get("cpu_rho_test_override", False),
        "wen_fresh_stale_f3_test": settings.get("wen_fresh_stale_f3_test", False),
        "host": settings.get("host", settings["arm"]),
        "wen_f12": settings.get("wen_f12"),
        "stage0_estimator": settings.get("stage0_estimator"),
        "study2_plugin_off_test": settings.get("study2_plugin_off_test", False),
        "seed": seed,
        "rounds": rounds,
        "scenario": settings["scenario"],
        "arm": settings["arm"],
        "stage0": settings["stage0"],
        "selection": settings["selection"],
        "epsilon": settings["epsilon"],
        **source_tree(),
    }
    out = Path(base_path) / "results" / "manifest.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest
