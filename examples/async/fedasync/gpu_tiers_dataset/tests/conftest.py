"""Put the dataset folder's modules on the path.

`make_fleet.py` and `build_fleet.py` import nothing from PLATO, so these tests
need no GPU, no torch and no Config. The tests that touch the shared runtime
(`gpu_tier_sleep.py`, `fedasync_polaris_vi_server.py`) add what they need
themselves and skip when it is unavailable.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent
FEDASYNC = DATASET_DIR.parent
REPO_ROOT = FEDASYNC.parents[2]          # examples/async/fedasync -> repo root

# Mirrors the run command's `PYTHONPATH=gpu_tiers:.` plus the repo root, so the
# tests that DO exercise the shared runtime (gpu_tier_sleep,
# fedasync_polaris_vi_server) import the same modules a real run would, rather
# than being skipped and quietly covering nothing.
#
# ORDER MATTERS, and reversed(): each insert(0) pushes the previous one down, so
# the list below is written in priority order and applied back to front. This
# folder MUST come first -- `gpu_tiers/make_configs.py` and
# `separability/make_configs.py` are different modules under the same top-level
# name, and whichever directory sys.path reaches first is the one `import
# make_configs` binds to. Getting that wrong does not fail loudly; it tests
# somebody else's generator.
for path in reversed([str(DATASET_DIR), str(FEDASYNC),
                      str(FEDASYNC / "gpu_tiers"), str(REPO_ROOT)]):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _safely_import_plato_servers_base() -> None:
    """Import `plato.servers.base` once, here, with an argv it can parse.

    `plato/servers/base.py:382` declares `def start(self, port=Config().server.port)`
    -- a default argument evaluated at CLASS-DEFINITION time, i.e. at import
    time. So merely importing that module runs `Config()`, and with it
    `argparse.parse_args()` over the whole of `sys.argv`. Under pytest that argv
    is `['pytest', 'tests/...', '-q']`, which argparse rejects, taking the whole
    test session down with a bare `SystemExit: 2` -- not a test failure, a
    collection-time crash.

    Doing it ONCE, here, before any test file gets a chance to trigger it by
    accident: Python caches modules in `sys.modules`, so every later
    `import plato...` (this file's own tests, or PLATO's own class bodies)
    reuses this import and never re-runs `Server`'s class body or re-parses
    argv. That is also why this must be a plain top-level call in conftest.py
    (collected before any test module) rather than a fixture -- a fixture only
    runs once a test has already been selected, which is too late for a module
    whose OWN import blows up the collection phase.

    A minimal, syntactically valid config is enough: nothing here reads its
    values, only its shape. If this project ever needs a real `Config()` inside
    a test, build it explicitly there -- this exists only to survive the import.
    """
    import tempfile
    import textwrap

    if "plato.servers.base" in sys.modules:
        return   # already safe; do not re-trigger anything

    minimal = Path(tempfile.mkdtemp()) / "conftest_minimal.toml"
    minimal.write_text(textwrap.dedent("""
        [clients]
        total_clients = 20
        per_round = 4
        [server]
        address = "127.0.0.1"
        port = 39998
        [data]
        partition_size = 2500
        [trainer]
        type = "basic"
        rounds = 1
        epochs = 1
        model_name = "resnet_18"
        [algorithm]
        type = "fedavg"
        [parameters]
        [results]
        types = "round, accuracy"
    """))

    saved_argv = sys.argv
    sys.argv = [saved_argv[0], "-c", str(minimal)]
    try:
        import plato.servers.base  # noqa: F401
    except Exception:
        # If PLATO's dependencies are not installed in this environment, leave
        # it be -- individual tests already skip via pytest.importorskip.
        pass
    finally:
        sys.argv = saved_argv


_safely_import_plato_servers_base()
