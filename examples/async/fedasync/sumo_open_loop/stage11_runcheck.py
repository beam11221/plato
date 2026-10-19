"""
Stage 11: per-run stop conditions, run by run_stage11.sh as each run ends.

    python3 stage11_runcheck.py <prefix> <run> <p_return>

Runs stage10_runcheck.py unchanged (conservation <= 3%, plateau >= 80.1%,
fallback <= 10%, median pool >= 4, predictor agreement 100%, occupancy == 7b),
then adds Stage 11's:
  realised commuter share within 2 points of nominal p_return (exhaustion
      events reported next to it)
  identity digest identical to every earlier run at the same p_return
      (logs/stage11_identity_digests.json)
Exit 0 pass, 1 stop.
"""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
prefix, run, p_return = sys.argv[1], sys.argv[2], float(sys.argv[3])
r10 = subprocess.run([sys.executable, str(HERE / "stage10_runcheck.py"), prefix, run],
                     capture_output=True, text=True)
base_path = HERE / "logs" / f"stage10_runcheck.{prefix}_{run}.json"
if not base_path.exists() or base_path.stat().st_mtime < (HERE / "logs" / f"{prefix}_{run}.log").stat().st_mtime:
    # stage10_runcheck.py crashed before writing its verdict: say why
    raise SystemExit(f"stage10_runcheck.py wrote no verdict (rc={r10.returncode}): "
                     + (r10.stderr.strip() or r10.stdout.strip()).splitlines()[-1][:300])
base = json.loads(base_path.read_text())
stop = list(base["stop"])
summ = None
for line in (HERE / "logs" / f"{prefix}_{run}.log").open(errors="ignore"):
    m = re.search(r"\[Stage11\] summary p_return=(\S+) allocations=(\d+) commuter_allocations=(\d+) "
                  r"realised_share=(\S+) exhaustions=(\d+) .* identity_digest=(\S+)", line)
    if m:
        summ = m.groups()
res = {"prefix": prefix, "run": run, "p_return": p_return, "stage10_checks": base}
if summ is None:
    stop.append("no [Stage11] summary line")
else:
    share, digest = float(summ[3]), summ[5]
    res.update(realised_share=share, allocations=int(summ[1]), exhaustions=int(summ[4]), identity_digest=digest)
    if abs(share - p_return) > 0.02:
        stop.append(f"realised commuter share {share:.4f} vs nominal {p_return} (> 2 pts; exhaustions {summ[4]})")
    # Runs end at different simulated times, so each covers a different number
    # of allocations.  The identity sequence is checked against an INDEPENDENT
    # replay (trace + physics seed) to exactly that count: equal digests mean
    # this run's sequence is the one every run at this p_return shares.
    sys.path.insert(0, str(HERE))
    from preflight_stage11 import replay
    _seq, rep = replay(p_return, int(summ[1]))
    res["identity_digest_replay"] = rep.returning.digest()
    if rep.returning.digest() != digest:
        stop.append(f"identity digest differs from the independent replay at p={p_return}")
res["stop"] = stop
(HERE / "logs" / f"stage11_runcheck.{prefix}_{run}.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps({k: v for k, v in res.items() if k != "stage10_checks"}, default=str))
sys.exit(1 if stop else 0)
