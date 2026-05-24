"""
Parse a FedAsync server log to extract per-aggregation-step records:
  wall_time (HH:MM:SS), effective_round, client_id, staleness, mixing,
  predicted_staleness, pred_error, pred_rel_error

Usage:
    python parse_staleness_log.py <server_*.log> [output.csv]

If no output path is given, the CSV is saved next to the log file with
the same stem and a .staleness.csv suffix.
"""

import csv
import re
import sys

# Matches timestamps like: [INFO][06:58:27]
_TS_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")

# FedAsync: step 1 - client #31, effective_round=13, staleness=3, mixing=0.4500
_STEP_RE = re.compile(
    r"FedAsync: step \d+ - client #(\d+), effective_round=(\d+), staleness=(\d+), mixing=([\d.]+)"
)

# [PolarisVi] Staleness prediction accuracy: cid=59  predicted=1.6099 rounds  actual=3.0000 rounds  error=+1.3901  rel_error=+46.3%
_PRED_RE = re.compile(
    r"\[PolarisVi\] Staleness prediction accuracy: cid=(\d+)\s+"
    r"predicted=([\d.]+) rounds\s+actual=([\d.]+) rounds\s+"
    r"error=([+-][\d.]+)\s+rel_error=([+-][\d.]+)%"
)


def parse_log(log_path):
    # Track the most-recent prediction per client so it can be joined onto
    # the aggregation step that follows in the same log stream.
    last_pred = {}   # cid -> {predicted, error, rel_error}

    records = []
    with open(log_path) as fh:
        for line in fh:
            p = _PRED_RE.search(line)
            if p:
                cid = int(p.group(1))
                last_pred[cid] = {
                    "predicted_staleness": float(p.group(2)),
                    "pred_error":          float(p.group(4)),
                    "pred_rel_error":      float(p.group(5)),
                }
                continue

            m = _STEP_RE.search(line)
            if m:
                ts  = _TS_RE.search(line)
                cid = int(m.group(1))
                pred = last_pred.get(cid, {})
                records.append({
                    "wall_time":          ts.group(1) if ts else "",
                    "effective_round":    int(m.group(2)),
                    "client_id":          cid,
                    "staleness":          int(m.group(3)),
                    "mixing":             float(m.group(4)),
                    "predicted_staleness": pred.get("predicted_staleness", ""),
                    "pred_error":          pred.get("pred_error", ""),
                    "pred_rel_error":      pred.get("pred_rel_error", ""),
                })
    return records


def write_csv(records, out_path):
    fieldnames = [
        "wall_time", "effective_round", "client_id",
        "staleness", "mixing",
        "predicted_staleness", "pred_error", "pred_rel_error",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else log_path.replace(".log", ".staleness.csv")

    records = parse_log(log_path)
    write_csv(records, out_path)
    print(f"Parsed {len(records)} steps -> {out_path}")


if __name__ == "__main__":
    main()
