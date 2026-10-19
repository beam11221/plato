"""
Seed every RNG a run touches, in every process the run spawns.

Why this exists
---------------
`../separability/PARALLELISM_CHECK.md` measured it: two runs of ONE config, both
alone on an idle box, differ by **6.29 %** of simulated clock. Nothing in PLATO
seeds torch, so the two runs initialise the model differently, compute different
gradients, get different `G_i`, rank `V_i = G_i / sqrt(1 + tau_hat_i)` differently
and therefore dispatch a different sequence of vehicles. That 6 % has been sitting
inside the project's measured noise floor and being read as natural variation.

What was already seeded, and what was not
-----------------------------------------
    RNG                       seeded by                            config key
    ------------------------  -----------------------------------  ----------------
    `random`                  plato/servers/base.py:374            [server] random_seed
                              (server process only, after startup)
    `numpy.random`            plato/config.py:456, inside           [clients] random_seed
                              simulate_client_speed(), every process
    sampler `torch.Generator` plato/samplers/dirichlet.py:81       [data] random_seed
    ------------------------  -----------------------------------  ----------------
    torch global CPU RNG      NOTHING                              --
    torch CUDA RNG            NOTHING                              --
    cuDNN algorithm choice    NOTHING                              --

The torch global RNG is what initialises ResNet-18's weights and what drives the
CIFAR-10 train transform (`RandomHorizontalFlip` + `RandomCrop(32, 4)`,
plato/datasources/torchvision.py:141). Both were free-running.

Where the seeding happens, and why there
----------------------------------------
At MODULE level of the entry script, not inside `main()`.

PLATO starts client processes with `mp.Process(target=plato.client.run, ...)`
under the **spawn** start method (plato/servers/base.py:479). A spawned child is a
fresh interpreter that re-imports the parent's `__main__` module under the name
`__mp_main__` before it unpickles the target -- so module-level code in the entry
script runs in the server process AND in every client process, which is exactly
the coverage needed, with no change to PLATO's core. Verified directly:

    [module-level] pid=1024582 __name__=__main__      argv=['probe.py','-c','x.toml']
    [module-level] pid=1024584 __name__=__mp_main__   argv=['/abs/probe.py','-c','x.toml']

`sys.argv` survives the spawn, which is what lets this module find the config file
without touching PLATO's `Config` singleton. It reads the TOML itself rather than
calling `Config()`: `Config()` has side effects at first call (it parses argv,
creates the base path, installs a logging FileHandler and runs
`simulate_client_speed()`), and provoking those earlier than PLATO expects is a
risk this module has no reason to take.

The per-process offset
----------------------
`PLATO_CLIENT_ID` is set by the server immediately before `proc.start()` and
deleted immediately after, so a spawned child inherits its own id and the server
process sees nothing. Verified:

    [module-level] pid=1024588 PLATO_CLIENT_ID=None    <- the server
    [module-level] pid=1024590 PLATO_CLIENT_ID='1'     <- client process 1
    [module-level] pid=1024591 PLATO_CLIENT_ID='2'

so the server seeds at `seed + 0` and client process k at `seed + k`. Giving every
client process the SAME seed would also be deterministic, but then all C workers
would draw the identical augmentation sequence, which is a correlation the runs do
not currently have and this change is not meant to introduce. The offset is a
small fixed integer, so the whole fleet is still a pure function of the one seed
in the config -- there is no second seed knob.

Note this seeds a process, not a logical client: a client process serves whichever
client id the server assigns it that round (plato/servers/base.py:682), so its RNG
stream is one sequence across all the clients it ever hosts. That is deterministic
given a deterministic dispatch sequence, which is the property being established.

What this does NOT make deterministic
-------------------------------------
`processing_time` is a `time.perf_counter()` delta measured on the host
(plato/clients/strategies/defaults.py:252) and PLATO adds it to every report's
finish time (plato/servers/base.py:1025-1030), which then becomes the simulated
clock. No amount of seeding removes it, so `sim_time_s` cannot be bit-identical
between two runs. See `stage1_evidence/determinism.txt` for what was measured.

`torch.use_deterministic_algorithms(True)` is NOT enabled by default: it requires
`CUBLAS_WORKSPACE_CONFIG` to be set and raises on ops that have no deterministic
kernel, which would turn a nondeterminism question into a crash. Set
`PLATO_DETERMINISTIC_ALGORITHMS=1` to turn it on for an experiment that needs it.
"""

from __future__ import annotations

import os
import random
import sys
import tomllib
from pathlib import Path
from typing import Any

# Filled in by seed_everything(); read by log_seeding() once logging is up.
_RECORD: dict[str, Any] = {}

# The config key the seed is taken from. All three of [clients]/[server]/[data]
# random_seed carry the same value in this project's generated configs; [clients]
# is used because plato/config.py reads it in EVERY process, so a config that
# lacks it is already broken for reasons unrelated to this module.
SEED_SECTION = "clients"
SEED_KEY = "random_seed"


def config_path_from_argv(argv: list[str] | None = None) -> Path:
    """The `-c/--config` value, defaulting the way plato/config.py does.

    Mirrors plato/config.py's parser for this one flag rather than importing it,
    so that reading the seed cannot initialise the Config singleton as a side
    effect.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, token in enumerate(argv):
        if token in ("-c", "--config"):
            if i + 1 < len(argv):
                return Path(argv[i + 1])
            raise SystemExit(f"[Seed] {token} given with no value")
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1])
        if token.startswith("-c") and len(token) > 2:
            return Path(token[2:])
    return Path("./config.toml")          # plato/config.py's own default


def read_seed(config_path: Path) -> int:
    """`[clients] random_seed` from the TOML, or a fatal error.

    Fatal rather than defaulted: a run that silently fell back to a fixed seed
    would look seeded in the log and not be reproducible per config, which is the
    failure mode this whole module exists to remove. It reads the file directly,
    so a seed placed in an INCLUDED TOML would not be found -- the generator in
    this folder always writes it inline, and this error is what says so if that
    ever changes.
    """
    if not config_path.exists():
        raise SystemExit(f"[Seed] config file not found: {config_path}")
    with config_path.open("rb") as fh:
        config = tomllib.load(fh)
    try:
        return int(config[SEED_SECTION][SEED_KEY])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"[Seed] [{SEED_SECTION}] {SEED_KEY} is missing or not an integer in "
            f"{config_path}. Every run in this folder must carry it: it is the "
            f"single seed every RNG is derived from."
        ) from exc


def process_offset() -> int:
    """0 in the server process, k in the k-th spawned client process."""
    raw = os.environ.get("PLATO_CLIENT_ID")
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(
            f"[Seed] PLATO_CLIENT_ID={raw!r} is not an integer; refusing to guess "
            f"a seed offset."
        ) from None


def seed_everything(config_path: Path | None = None) -> dict[str, Any]:
    """Seed python `random`, numpy, torch CPU and CUDA; pin cuDNN.

    Returns the record of what it did. Safe to call more than once -- it simply
    re-seeds to the same values -- but it is meant to be called exactly once, at
    import time of the entry script.
    """
    path = config_path if config_path is not None else config_path_from_argv()
    base_seed = read_seed(path)
    offset = process_offset()
    seed = base_seed + offset

    random.seed(seed)

    record: dict[str, Any] = {
        "config": str(path),
        "base_seed": base_seed,
        "offset": offset,
        "seed": seed,
        "pid": os.getpid(),
        "role": "server" if offset == 0 else f"client_process_{offset}",
        "numpy": False,
        "torch": False,
        "cuda_devices": 0,
        "cudnn_deterministic": False,
        "deterministic_algorithms": False,
    }

    try:
        import numpy as np
    except ImportError:                                  # pragma: no cover
        pass
    else:
        np.random.seed(seed)
        record["numpy"] = True

    try:
        import torch
    except ImportError:                                  # pragma: no cover
        pass
    else:
        torch.manual_seed(seed)
        record["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            record["cuda_devices"] = torch.cuda.device_count()
        # deterministic = True picks a fixed convolution algorithm;
        # benchmark = False stops the autotuner picking a different one per run
        # based on timings, which are host- and contention-dependent.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        record["cudnn_deterministic"] = True

        if os.environ.get("PLATO_DETERMINISTIC_ALGORITHMS") == "1":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            record["deterministic_algorithms"] = True

    _RECORD.clear()
    _RECORD.update(record)
    return record


def record() -> dict[str, Any]:
    """What seed_everything() did in this process; empty if it never ran."""
    return dict(_RECORD)


def format_record(rec: dict[str, Any] | None = None) -> str:
    rec = record() if rec is None else rec
    if not rec:
        return "[Seed] seed_everything() has not run in this process"
    return (
        "[Seed] pid={pid} role={role} seed={seed} "
        "(= [clients] random_seed {base_seed} + offset {offset})  "
        "numpy={numpy} torch={torch} cuda_devices={cuda_devices} "
        "cudnn_deterministic={cudnn_deterministic} "
        "deterministic_algorithms={deterministic_algorithms}".format(**rec)
    )


def log_seeding(logged: list[bool] = []) -> None:  # noqa: B006 - per-process latch
    """Emit the seeding record to PLATO's logging, once per process.

    Called from the entry script rather than from `seed_everything()` because
    seeding runs at import time, BEFORE `Config()` installs the run's
    FileHandler -- a log line emitted then would reach no file, and the evidence
    that seeding actually ran inside each spawned client process is the point.
    """
    if logged:
        return
    logged.append(True)
    import logging                                       # noqa: PLC0415

    logging.info("%s", format_record())
