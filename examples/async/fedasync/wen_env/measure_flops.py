"""
Measure the forward cost and parameter count of PLATO's `resnet_18` (Prompt B, Step 1).

Counter: fvcore `FlopCountAnalysis`. Despite the name, fvcore counts one unit
per fused multiply-add, i.e. it reports MACs; this script converts with
FLOPs = 2 x MACs and prints both. Input is one 32x32x3 image (batch 1), the
model in eval mode, 10 classes.

Also prints the parameter count -- the D of the upload model is that count x 32
bits -- and the state_dict tensor count for comparison (BatchNorm running
statistics are buffers, not parameters; they travel in the payload but are not
part of Wen's D).

Usage (from examples/async/fedasync):
    PYTHONPATH=<repo root> python wen_env/measure_flops.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
from pathlib import Path

NOMINAL_FLOPS_FWD = 1.1e9
STOP_TOLERANCE = 0.20


def _minimal_config() -> None:
    """PLATO's model registry reads Config(); give it just enough."""
    cfg = Path(tempfile.mkdtemp()) / "measure.toml"
    cfg.write_text(textwrap.dedent("""
        [clients]
        total_clients = 1
        per_round = 1
        [server]
        address = "127.0.0.1"
        port = 39997
        [data]
        datasource = "Torchvision"
        dataset_name = "CIFAR10"
        partition_size = 1
        [trainer]
        type = "basic"
        rounds = 1
        epochs = 1
        model_name = "resnet_18"
        batch_size = 1
        [algorithm]
        type = "fedavg"
        [parameters]
        [parameters.model]
        num_classes = 10
    """))
    sys.argv = [sys.argv[0], "-c", str(cfg)]


def measure() -> dict:
    _minimal_config()
    import torch
    from fvcore.nn import FlopCountAnalysis

    from plato.models import registry as models_registry

    model = models_registry.get()
    model.eval()
    with torch.no_grad():
        analysis = FlopCountAnalysis(model, torch.zeros(1, 3, 32, 32))
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        macs = int(analysis.total())
        by_op = {k: int(v) for k, v in analysis.by_operator().items()}

    params = sum(p.numel() for p in model.parameters())
    state = sum(t.numel() for t in model.state_dict().values())
    flops = 2 * macs
    return {
        "model": type(model).__module__ + "." + type(model).__name__,
        "counter": "fvcore.nn.FlopCountAnalysis (counts multiply-adds = MACs)",
        "macs_fwd": macs,
        "flops_fwd": flops,
        "macs_by_operator": by_op,
        "deviation_from_1p1_gflop": flops / NOMINAL_FLOPS_FWD - 1.0,
        "parameters": params,
        "state_dict_elements": state,
        "model_bits_D": params * 32,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    out = args.json
    result = measure()
    print(json.dumps(result, indent=2))
    if out is not None:
        out.write_text(json.dumps(result, indent=2) + "\n")
    if abs(result["deviation_from_1p1_gflop"]) > STOP_TOLERANCE:
        print("STOP: measured FLOPs_fwd differs from 1.1 GFLOP by more than 20%.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
