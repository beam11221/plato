"""
Regression test: an unrecognised `staleness_predictor` must FAIL, not fall back.

Why this test exists
--------------------
`[server] staleness_predictor` is the single key that decides which arm a run is.
It used to be read as

    getattr(server_cfg, "staleness_predictor", "observed")

and then tested with `in ("gpu_tops", "cpu")`, so any value it did not recognise
silently became the observed-tau predictor (gpu_tiers/HANDOFF.md § 4.10).

That was a nuisance while "observed" was only a default.  It becomes a serious
hazard once "observed" is a real ARM of a sweep (surgical/BRIEF_surgical.md): a
typo in the gpu or cpu config then produces a run that is behaviourally
IDENTICAL to another arm of the same experiment, and nothing in the results
distinguishes the duplicate.

Two guards, tested here:
  1. generation time — gpu_tiers/make_configs.py refuses to write a config whose
     predictor does not match the arm implied by its own filename;
  2. run time — fedasync_polaris_vi_server.resolve_staleness_predictor raises,
     for configs that were hand-written rather than generated.

Run:
    cd plato/examples/async/fedasync
    python3 test_predictor_strictness.py

Part 2 needs torch and plato's dependencies; it is SKIPPED with a message when
they are unavailable, so part 1 still runs on a bare interpreter.
"""

from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "gpu_tiers"))

import make_configs  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{PASS if ok else FAIL}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def problems_for(cfgs: dict) -> list[str]:
    """Run make_configs.validate() over a {name: cfg} map, paths unchecked."""
    rendered = {
        name: ("", cfg, 8100 + i) for i, (name, cfg) in enumerate(cfgs.items())
    }
    return make_configs.validate(rendered, check_paths=False)


def mentions(problems: list[str], *needles: str) -> bool:
    return any(all(n in p for n in needles) for p in problems)


# ---------------------------------------------------------------------------
# Part 1 — generation time (no heavy dependencies)
# ---------------------------------------------------------------------------

print("\n1. make_configs.ARM_PREDICTOR covers every arm it can render")
check(
    "every ARMS entry has a predictor rule",
    set(make_configs.ARMS) == set(make_configs.ARM_PREDICTOR),
    f"ARMS={sorted(make_configs.ARMS)} ARM_PREDICTOR={sorted(make_configs.ARM_PREDICTOR)}",
)
check(
    "baseline maps to None (no dead key — HANDOFF § 4.7)",
    make_configs.ARM_PREDICTOR["baseline"] is None,
)

print("\n2. Every config already on disk passes — all four sweeps")
cfgs: dict = {}
for cfg_dir in ("configs", "configs_ms", "configs_mix", "configs_comm"):
    here = {}
    for path in sorted((Path(__file__).parent / "gpu_tiers" / cfg_dir).glob("*.toml")):
        here[path.name] = tomllib.loads(path.read_text())
    probs = [p for p in problems_for(here) if "staleness_predictor" in p or "unknown arm" in p]
    check(f"{cfg_dir} ({len(here)} configs)", not probs, str(probs[:2]))
    cfgs.update(here)
check("the gap sweep is all 15 arms x gaps", len([k for k in cfgs if k.count("_") == 2]) == 15)

print("\n3. A typo is caught instead of silently becoming the observed arm")
base = copy.deepcopy(cfgs["gpu_xa20_gap30.toml"])
typo = copy.deepcopy(base)
typo["server"]["staleness_predictor"] = "GPU_TOPS"
probs = problems_for({"gpu_xa20_gap30.toml": typo})
check("'GPU_TOPS' rejected", mentions(probs, "staleness_predictor", "gpu_tops"), str(probs[:1]))

typo2 = copy.deepcopy(base)
typo2["server"]["staleness_predictor"] = "gpu_top"
probs = problems_for({"gpu_xa20_gap30.toml": typo2})
check("'gpu_top' rejected", mentions(probs, "staleness_predictor"), str(probs[:1]))

print("\n4. A predictor that belongs to another arm is caught")
swapped = copy.deepcopy(base)
swapped["server"]["staleness_predictor"] = "cpu"
probs = problems_for({"gpu_xa20_gap30.toml": swapped})
check("cpu predictor in a gpu-named config rejected", mentions(probs, "requires 'gpu_tops'"), str(probs[:1]))

print("\n5. An absent key is caught for an arm that needs one")
missing = copy.deepcopy(base)
del missing["server"]["staleness_predictor"]
probs = problems_for({"gpu_xa20_gap30.toml": missing})
check("absent key rejected for the gpu arm", mentions(probs, "the key is absent"), str(probs[:1]))

print("\n6. The baseline must not carry the key")
bl = copy.deepcopy(cfgs["baseline_xa20_gap30.toml"])
probs = problems_for({"baseline_xa20_gap30.toml": bl})
check("clean baseline passes", not [p for p in probs if "staleness_predictor" in p])
bl_bad = copy.deepcopy(bl)
bl_bad.setdefault("server", {})["staleness_predictor"] = "observed"
probs = problems_for({"baseline_xa20_gap30.toml": bl_bad})
check("baseline carrying the key rejected", mentions(probs, "no predictor"), str(probs[:1]))

# ---------------------------------------------------------------------------
# Part 2 — run time (needs torch + plato)
# ---------------------------------------------------------------------------

print("\n7. resolve_staleness_predictor() — runtime backstop")
# plato/config.py parses sys.argv at IMPORT time and exits without -c, so point
# it at any valid config before importing the server module.  Which one does not
# matter: only the module-level function is under test here.
_argv = sys.argv[:]
sys.argv = [
    sys.argv[0], "-c",
    str(Path(__file__).parent / "gpu_tiers" / "configs" / "gpu_xa20_gap30.toml"),
]
try:
    import fedasync_polaris_vi_server as srv
except Exception as exc:  # noqa: BLE001 — missing deps, not a test failure
    print(f"  [{SKIP}] needs plato's dependencies ({type(exc).__name__}: {exc})")
    print("         re-run once the environment is repaired (HANDOFF § 2.2)")
else:
    check("absent key -> 'observed'", srv.resolve_staleness_predictor(None) == "observed")
    for name in ("gpu_tops", "cpu", "observed"):
        check(f"{name!r} accepted", srv.resolve_staleness_predictor(name) == name)
    check("case and whitespace normalised", srv.resolve_staleness_predictor("  GPU_TOPS ") == "gpu_tops")
    for bad in ("gpu_top", "observed_tau", "", "history"):
        try:
            srv.resolve_staleness_predictor(bad)
            check(f"{bad!r} raises", False, "no exception")
        except ValueError:
            check(f"{bad!r} raises", True)
sys.argv = _argv

print()
if failures:
    print(f"\033[31m{len(failures)} FAILED:\033[0m " + ", ".join(failures))
    sys.exit(1)
print("\033[32mAll checks passed.\033[0m")
