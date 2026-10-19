"""
Tests for predictor anchoring ([gpu_tier] anchor_tier / anchor_sleep).

What is being protected
-----------------------
The simulator's sleep CSVs are anchored (xavier = 20 s) but the predictor used
to compute raw formula seconds (xavier = 0.055 s).  Ratios still cancelled in
τ̂ = T_pred / T_round, so the bug was invisible there — but select_clients()
also does

    residual = max(0, t_full - (wall_now - start_time))

which mixes the two scales.  In the GPU arm every residual collapsed to 0 and
Phase 1 of gpu_tops_predict_staleness_rounds never fired; in the CPU arm the raw
formula reads tiers_cpu.csv's 0.002 as TFLOPS and predicts 275 s, so residuals
saturated instead.  The two arms therefore produced τ̂ on different magnitudes
(GPU {0,1} vs CPU {2,9}), confounding the very comparison the sweep exists for.

Run:
    cd plato/examples/async/fedasync/gpu_tiers
    python test_predictor_anchor.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
# Repo root, so `import plato` works when run directly rather than under `uv run`.
sys.path.insert(1, str(_HERE.parents[3]))

import gpu_tier_sleep as gts  # noqa: E402

TEMPLATES = _HERE / "templates"
ANCHOR = 20
GAP = "20"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures: list[str] = []


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def load_config(template: str, *, drop_anchor: bool = False, **render_kwargs):
    """Reset every singleton and re-init PLATO's Config from a rendered template.

    Rendering goes through ``make_configs.render`` rather than a local list of
    placeholders.  The local list was the bug: every placeholder added to the
    templates afterwards ({{SEED}}, {{ROUNDS}}, {{RUNDIR}}, then {{MIXSFX}},
    {{TROUND}}, {{TARGETACC}}, {{BWSFX}}, {{RUNBWSFX}}, {{COMMBLOCK}}) was left
    unsubstituted here, and an unsubstituted bare value is not valid TOML — so
    this file failed to parse its own inputs long before it could check
    anything about anchoring.

    ``render_kwargs`` forwards extra keyword arguments to ``make_configs.render``
    (e.g. ``comm_bandwidth_mbps=100.0``) for the comm-latency checks below.
    """
    from plato.config import Config

    import make_configs

    arm = next(a for a, spec in make_configs.ARMS.items() if spec[0] == template)
    _, text, _ = make_configs.render(arm, GAP, ANCHOR, root=_HERE.parent, **render_kwargs)
    if drop_anchor:
        text = "\n".join(
            line for line in text.splitlines()
            if not line.startswith(("anchor_tier", "anchor_sleep"))
        )

    rendered = _HERE / f".test_{template.replace('.tmpl', '')}"
    rendered.write_text(text)

    Config._instance = None
    gts._computer = None
    gts._predictor_scale = None
    gts._comm_latency = None
    gts._comm_in_predictor = None
    gts._t_round_logged = False
    sys.argv = ["test_predictor_anchor.py", "-c", str(rendered)]
    Config()
    return rendered


def check(name: str, got, want, tol: float = 1e-6) -> None:
    ok = abs(float(got) - float(want)) <= tol
    print(f"  [{PASS if ok else FAIL}] {name}: got {got:.6f}, want {want:.6f}")
    if not ok:
        failures.append(name)


def check_true(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{PASS if ok else FAIL}] {name}{f': {detail}' if detail else ''}")
    if not ok:
        failures.append(name)


def read_times_csv(path: Path) -> dict[int, float]:
    with path.open(newline="") as f:
        return {int(r["client_id"]): float(r["sleep_time"]) for r in csv.DictReader(f)}


# Representative client of each tier, per input/clients.csv (50/35/15 split).
CIDS = {"xavier": 1, "orin": 51, "thor": 86}
SIZES = {c: 500 for c in range(1, 101)}
CLIENT_IDS = list(range(1, 101))


# ---------------------------------------------------------------------------
# 1. Backward compatibility — no anchor keys means no rescaling
# ---------------------------------------------------------------------------

print("\n1. Omitting anchor_tier/anchor_sleep keeps the raw formula")
load_config("fedasync_polaris_vi_gpu_arm.toml.tmpl", drop_anchor=True)
check("predictor_scale()", gts.predictor_scale(), 1.0)
check("raw xavier seconds", gts.gpu_tops_predict_seconds(CIDS["xavier"], 500, 1), 0.055)


# ---------------------------------------------------------------------------
# 2. GPU arm — predictions must land exactly on the simulator's sleep CSV
# ---------------------------------------------------------------------------

print("\n2. GPU arm: predicted seconds == the sleep CSV clients actually use")
load_config("fedasync_polaris_vi_gpu_arm.toml.tmpl")
gpu_scale = gts.predictor_scale()
check("predictor_scale()", gpu_scale, ANCHOR / 0.055, tol=1e-4)

reality = read_times_csv(
    _HERE / "output" / f"client_sleep_times_xa{ANCHOR}_gap{GAP}.csv"
)
worst = max(
    abs(gts.gpu_tops_predict_seconds(cid, 500, 1) - reality[cid])
    for cid in CLIENT_IDS
)
check_true(
    "all 100 clients match reality", worst < 1e-6, f"max abs error {worst:.3e} s"
)
for tier, cid in CIDS.items():
    check(f"  {tier}", gts.gpu_tops_predict_seconds(cid, 500, 1), reality[cid])


# ---------------------------------------------------------------------------
# 3. τ̂ is unchanged — the anchor cancels in T_pred / T_round
# ---------------------------------------------------------------------------

print("\n3. tau_hat via T_pred / T_round is invariant to the anchor")
t_round_anchored = gts.compute_pool_T_round_seconds(CLIENT_IDS, SIZES, 1, clients_per_round=10)
tau_anchored = {
    t: gts.seconds_to_rounds(gts.gpu_tops_predict_seconds(c, 500, 1), t_round_anchored)
    for t, c in CIDS.items()
}

load_config("fedasync_polaris_vi_gpu_arm.toml.tmpl", drop_anchor=True)
t_round_raw = gts.compute_pool_T_round_seconds(CLIENT_IDS, SIZES, 1, clients_per_round=10)
for tier, cid in CIDS.items():
    tau_raw = gts.seconds_to_rounds(gts.gpu_tops_predict_seconds(cid, 500, 1), t_round_raw)
    check(f"  tau_hat {tier}", tau_anchored[tier], tau_raw, tol=1e-9)


# ---------------------------------------------------------------------------
# 4. Phase 1 fires once anchored, and provably did not before
# ---------------------------------------------------------------------------
# Reproduces select_clients(): busy clients started at staggered wall-clock
# times, residual = max(0, t_full - elapsed).  Elapsed is always in simulated
# wall-clock seconds; only t_full moves with the anchor.

print("\n4. Phase 1 (residual-aware) fires only when the predictor is anchored")
BUSY = [1, 2, 3, 4, 5, 51, 52, 53, 86, 87]      # 5 xavier, 3 orin, 2 thor
ELAPSED = [0.0, 3.0, 7.0, 11.0, 15.0, 2.0, 6.0, 10.0, 1.0, 5.0]


def phase1_events(candidate: int) -> tuple[int, list[float]]:
    residuals = [
        max(0.0, gts.gpu_tops_predict_seconds(cid, 500, 1) - el)
        for cid, el in zip(BUSY, ELAPSED)
    ]
    t_client = gts.gpu_tops_predict_seconds(candidate, 500, 1)
    return len([r for r in residuals if 0.0 < r < t_client]), residuals


for label, drop in (("raw (the bug)", True), ("anchored (fixed)", False)):
    load_config("fedasync_polaris_vi_gpu_arm.toml.tmpl", drop_anchor=drop)
    events, residuals = phase1_events(CIDS["xavier"])
    nonzero = sum(1 for r in residuals if r > 0.0)
    print(f"  {label}: {nonzero}/10 residuals > 0, Phase-1 events for xavier = {events}")
    if drop:
        check_true("raw: Phase 1 is dead", events == 0, "no busy client ever counted")
    else:
        check_true("anchored: Phase 1 counts finishers", events > 0, f"{events} events")


# ---------------------------------------------------------------------------
# 5. CPU arm — anchored the same way, so tau_hat magnitudes stay comparable
# ---------------------------------------------------------------------------

print("\n5. CPU arm lands on the documented 20 / 20 / 15.38 s belief")
load_config("fedasync_polaris_vi_cpu_arm.toml.tmpl")
expected = read_times_csv(_HERE / "output" / f"client_predicted_times_cpu_xa{ANCHOR}.csv")
for tier, cid in CIDS.items():
    check(f"  {tier}", gts.gpu_tops_predict_seconds(cid, 500, 1), expected[cid])

cpu_events, cpu_residuals = phase1_events(CIDS["thor"])
check_true(
    "CPU arm residuals are on the wall-clock scale",
    max(cpu_residuals) <= ANCHOR,
    f"max residual {max(cpu_residuals):.2f} s (raw would be ~275 s)",
)

# ---------------------------------------------------------------------------
# 6. Comm latency C — added AFTER predictor_scale(), on the anchored scale
# ---------------------------------------------------------------------------
# BRIEF_comm_sweep.md § 4.2: C is a real-seconds quantity in the SAME units as
# the anchored sleep values.  Adding it before predictor_scale() would scale it
# by the 363x anchor factor along with the raw compute term — the same bug in a
# new shape as the un-anchored predictor in § 4.1.

print("\n6. Comm latency C: added on the anchored scale, not scaled itself")
load_config("fedasync_polaris_vi_gpu_arm.toml.tmpl")
check("comm_latency_seconds() absent key", gts.comm_latency_seconds(), 0.0)
baseline_xavier = gts.gpu_tops_predict_seconds(CIDS["xavier"], 500, 1)
baseline_t_round = gts.compute_pool_T_round_seconds(CLIENT_IDS, SIZES, 1, clients_per_round=10)

B_MBPS = 100.0
D_MBIT = 341.52
expected_C = D_MBIT / B_MBPS  # 3.4152 s

load_config(
    "fedasync_polaris_vi_gpu_arm.toml.tmpl",
    comm_bandwidth_mbps=B_MBPS, payload_mbit=D_MBIT,
)
check("comm_latency_seconds()", gts.comm_latency_seconds(), expected_C, tol=1e-6)
check_true("comm_in_predictor() defaults true", gts.comm_in_predictor() is True)
check(
    "gpu_tops_predict_seconds(xavier) = C=0 prediction + C (not (compute+C)*scale)",
    gts.gpu_tops_predict_seconds(CIDS["xavier"], 500, 1),
    baseline_xavier + expected_C,
    tol=1e-6,
)
check(
    "compute_pool_T_round_seconds also carries C (§ 4.4: no un-augmented second path)",
    gts.compute_pool_T_round_seconds(CLIENT_IDS, SIZES, 1, clients_per_round=10),
    baseline_t_round + expected_C,
    tol=1e-6,
)

print("\n7. comm_in_predictor = false: the belief-vs-reality ablation")
load_config(
    "fedasync_polaris_vi_gpu_arm.toml.tmpl",
    comm_bandwidth_mbps=B_MBPS, payload_mbit=D_MBIT, comm_in_predictor=False,
)
check_true("comm_in_predictor() reads false", gts.comm_in_predictor() is False)
check(
    "gpu_tops_predict_seconds ignores C when comm_in_predictor=false",
    gts.gpu_tops_predict_seconds(CIDS["xavier"], 500, 1),
    baseline_xavier,
    tol=1e-6,
)

print("\n8. comm_bandwidth_mbps <= 0 raises rather than silently falling back to C=0")
try:
    load_config(
        "fedasync_polaris_vi_gpu_arm.toml.tmpl",
        comm_bandwidth_mbps=0.0, payload_mbit=D_MBIT,
    )
    gts.comm_latency_seconds()
    check_true("comm_bandwidth_mbps=0 raises ValueError", False, "no exception raised")
except ValueError:
    check_true("comm_bandwidth_mbps=0 raises ValueError", True)

for f in _HERE.glob(".test_*.toml"):
    f.unlink()

print(f"\n{'-' * 62}")
if failures:
    print(f"{FAIL} {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print(f"{PASS} all checks passed")
