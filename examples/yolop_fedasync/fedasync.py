"""
A federated learning training session using FedAsync with YOLOP on BDD100K.

Mirrors the entry-point pattern of examples/async/fedasync/fedasync.py, but
passes custom model / datasource / trainer factories so the framework
bypasses its built-in registries.

Reference: Xie, C., Koyejo, S., Gupta, I. "Asynchronous federated optimization,"
OPT 2020. https://opt-ml.org/papers/2020/paper_28.pdf
"""

import os
import sys
import pathlib

# Cap BLAS/OpenMP threads per process. Without this each DataLoader worker
# spawns up to 48 OpenBLAS threads, quickly exhausting the system thread limit
# when many virtual clients run concurrently.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Make the YOLOP repo importable as `lib.*`. The example loads YOLOP from
# the sibling directory at the repo root: /workspace/{plato,YOLOP}.
# fedasync.py lives at examples/yolop_fedasync/, so parents[2] = /workspace.
_HERE = pathlib.Path(__file__).resolve().parent
_YOLOP_ROOT = _HERE.parents[2] / "YOLOP"
if str(_YOLOP_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLOP_ROOT))

# Make the local example modules importable when running from inside the
# folder (mirrors how examples/async/fedasync/fedasync.py imports its
# fedasync_algorithm / fedasync_server modules).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fedasync_algorithm
import fedasync_server

from plato.clients import simple

from datasources.bdd100k import DataSource
from models.yolop_model import Model
from trainers.yolop_trainer import Trainer


def main():
    """A Plato federated learning training session using FedAsync + YOLOP."""
    algorithm = fedasync_algorithm.Algorithm

    # `simple.Client(model=, datasource=, trainer=)` stores these as
    # custom_* attributes which the lifecycle strategy uses to bypass the
    # registry (see plato/clients/strategies/defaults.py:55-95).
    client = simple.Client(
        algorithm=algorithm,
        model=Model.get,
        datasource=DataSource,
        trainer=Trainer,
    )
    server = fedasync_server.Server(
        algorithm=algorithm,
        model=Model.get,
        datasource=DataSource,
        trainer=Trainer,
    )
    server.run(client)


if __name__ == "__main__":
    main()
