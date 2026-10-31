"""
Prompt M B3: the 24 tier-mix sweep configs (6 mixes x random / gpu / polaris /
wen_fresh, S1, seed 4126).

Each is the arm's Part A config (wen_env/configs/promptM/S1_<arm>_s4126_pm.toml,
itself the Prompt K seed-4126 config) with only these keys changed: base_path,
port, [wen_env] fleet_dir + fleet_mix, the placeholder sleep csv_path, and for
wen_fresh the W5 limits recomputed for the mix (evidence/wen_limits_<mix>.json).

Usage (from examples/async/fedasync):
    python wen_env/tools/promptm_sweep_configs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEN = HERE.parent
sys.path.insert(0, str(WEN))

import wen_constants as K  # noqa: E402
from wen_fleet import M0_FLEET, mix_fleet_dir  # noqa: E402

ARMS = ("random", "gpu", "polaris", "wen_fresh")
OUT = WEN / "configs" / "promptM"
PORT0 = {"random": 22000, "gpu": 22100, "polaris": 22200, "wen_fresh": 22300}


def main() -> int:
    written = []
    for i, mix in enumerate(m for m in K.FLEET_MIXES if m != "M0"):
        limits = json.loads((WEN / "evidence" / f"wen_limits_{mix}.json").read_text())
        for arm in ARMS:
            src = OUT / f"S1_{arm}_s4126_pm.toml"
            text = src.read_text()
            name = f"S1_{mix}_{arm}_s4126"
            fleet = mix_fleet_dir(Path("/x"), mix).name
            t = re.sub(r'base_path = ".*"', f'base_path = "./runs/wen_env/{name}"', text)
            t = re.sub(r"(?m)^port = \d+", f"port = {PORT0[arm] + 10 * i}", t)
            assert t.count(M0_FLEET) == 2, (src, t.count(M0_FLEET))
            t = t.replace(M0_FLEET, fleet)
            t = re.sub(r'(?m)^(tiers_csv = .*)$', rf'\1\nfleet_mix = "{mix}"', t)
            if arm == "wen_fresh":
                t = re.sub(r"(?m)^wen_t_max = .*$", f"wen_t_max = {limits['t_max']!r}", t)
                t = re.sub(r"(?m)^wen_e_max = .*$", f"wen_e_max = {limits['e_max']!r}", t)
            t = re.sub(r"(-c wen_env/configs/)\S+", rf"\1promptM/{name}.toml", t)
            t = re.sub(r"# Prompt M Part A: .*",
                       f"# Prompt M Part B, mix {mix} {K.FLEET_MIXES[mix]}: {src.name} with "
                       "base_path, port, fleet_dir, fleet_mix (and Wen limits) changed.", t)
            t = t.replace("S1 / ", f"S1 {mix} / ", 1)
            (OUT / f"{name}.toml").write_text(t)
            written.append(name)
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
