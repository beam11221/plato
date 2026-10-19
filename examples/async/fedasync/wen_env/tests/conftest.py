"""Path setup for the wen_env tests, plus a safe one-time PLATO import.

`plato/servers/base.py` evaluates `Config()` at class-definition time, i.e. on
import, which parses sys.argv; under pytest that argv is not a PLATO command
line. Importing it once here with a minimal config keeps collection alive (the
same fix as gpu_tiers_dataset/tests/conftest.py).
"""

import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEN = HERE.parent
FEDASYNC = WEN.parent
REPO_ROOT = FEDASYNC.parents[2]
BUNDLE_DIR = FEDASYNC / "sumo_ver2" / "bundle_wen_closed_v1"
FLEET_DIR = FEDASYNC / "gpu_tiers_dataset" / "fleets" / "n30_x15-o10-t5_phys_p1667_j0p149_s4126"
TIERS_CSV = FEDASYNC / "gpu_tiers_dataset" / "tiers" / "tiers_wen.csv"
RUNS_DIR = FEDASYNC / "runs" / "wen_env"

for path in reversed([str(WEN), str(FEDASYNC), str(FEDASYNC / "gpu_tiers"),
                      str(FEDASYNC / "gpu_tiers_dataset"),
                      str(FEDASYNC / "sumo_close_loop"), str(REPO_ROOT)]):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _safe_plato_import() -> None:
    if "plato.servers.base" in sys.modules:
        return
    cfg = Path(tempfile.mkdtemp()) / "wen_tests.toml"
    cfg.write_text(textwrap.dedent("""
        [clients]
        total_clients = 30
        per_round = 3
        [server]
        address = "127.0.0.1"
        port = 39996
        [data]
        partition_size = 1667
        [trainer]
        type = "basic"
        rounds = 1
        epochs = 1
        model_name = "resnet_18"
        [algorithm]
        type = "fedavg"
    """))
    saved = sys.argv
    sys.argv = [saved[0], "-c", str(cfg)]
    try:
        import plato.servers.base  # noqa: F401
    finally:
        sys.argv = saved


_safe_plato_import()
