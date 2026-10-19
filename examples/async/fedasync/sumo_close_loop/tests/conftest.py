"""Make this example importable and satisfy plato.config, however pytest ran.

Two setup steps, both of which must happen before any `plato.*` import:

  * The FedAsync examples import each other bare (`import gpu_tier_sleep`,
    `import fedasync_server`), which normally works because run_sweep.sh sets
    PYTHONPATH. Tests are run by hand from anywhere, so they set it up here.

  * `plato.config` parses argv at import time and calls sys.exit without `-c`,
    so it is given a config. Nothing in these tests reads a value from it; `-b`
    redirects base_path to a scratch directory so importing plato does not drop
    log files into a real run directory. Same trick as
    `examples/async/fedasync/tests/test_flush_refill.py`.
"""

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent  # sumo_close_loop/
FEDASYNC_DIR = EXPERIMENT_DIR.parent  # examples/async/fedasync/
REPO_ROOT = FEDASYNC_DIR.parents[2]  # the plato checkout

for path in (REPO_ROOT, FEDASYNC_DIR, FEDASYNC_DIR / "gpu_tiers", EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if not any(arg == "-c" for arg in sys.argv):
    _config = EXPERIMENT_DIR / "configs" / "gpu_n20.toml"
    if not _config.exists():  # before build_inputs.sh has been run
        _config = FEDASYNC_DIR / "surgical" / "configs" / "gpu_xa100_gap15.toml"
    sys.argv = [
        sys.argv[0],
        "-c",
        str(_config),
        "-b",
        tempfile.mkdtemp(prefix="plato-sumo-test-"),
    ]
