#!/bin/bash
# Watches the comm-latency sweep to completion, checking in periodically.
set -uo pipefail
cd "$(dirname "$0")"

RUNS=(
  gpu_xa20_gap30_m50_35_15_b100
  cpu_xa20_gap30_m50_35_15_b100
  gpu_xa20_gap30_m50_35_15_b050
  cpu_xa20_gap30_m50_35_15_b050
  gpu_xa20_gap30_m50_35_15_b010nc
  gpu_xa20_gap30_m50_35_15_b020
  cpu_xa20_gap30_m50_35_15_b020
  baseline_xa20_gap30_m50_35_15_b020
  gpu_xa20_gap30_m50_35_15_b010
  cpu_xa20_gap30_m50_35_15_b010
  baseline_xa20_gap30_m50_35_15_b010
)
TOTAL=${#RUNS[@]}
START=$(date +%s)
MAX_HOURS=30

while true; do
    done_count=0
    failed=()
    for r in "${RUNS[@]}"; do
        if [[ -f "logs/${r}.done" ]]; then
            (( done_count++ ))
        elif [[ -f "logs/${r}.log" ]] && ! pgrep -f "configs_comm/${r}.toml" >/dev/null; then
            # log exists, no .done marker, no live process for it -> looks dead
            failed+=("$r")
        fi
    done

    now=$(date +%s)
    elapsed_h=$(( (now - START) / 3600 ))
    disk=$(du -sh --apparent-size /workspace 2>/dev/null | cut -f1)

    echo "[$(date '+%F %T')] ${done_count}/${TOTAL} done, elapsed ${elapsed_h}h, /workspace ${disk}"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "  POSSIBLY STALLED (log exists, no process, no .done): ${failed[*]}"
    fi

    if [[ "$done_count" -eq "$TOTAL" ]]; then
        echo "ALL ${TOTAL} RUNS FINISHED after ${elapsed_h}h."
        exit 0
    fi
    if [[ "$elapsed_h" -ge "$MAX_HOURS" ]]; then
        echo "TIMEOUT: still ${done_count}/${TOTAL} after ${MAX_HOURS}h -- stopping the watch, sweep may still be running."
        exit 1
    fi
    sleep 5400
done
