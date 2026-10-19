#!/bin/bash
# Run one or both Stage 5b configs (configs/stage5b/*.toml), in parallel, each
# with its own checkpoint cleaner. Mirrors run_stage5.sh's conventions.
#
# Usage:
#   ./run_stage5b.sh <run> [run ...] [options]
#
#   run   baseline_tier_spec_vmax_distance | baseline_tier_spec_oracle_exit
#         or "all" for both
#   --keep-rounds N     checkpoints to keep   (default: 50)
#   --clean-interval N  cleaner period, minutes (default: 3)
#   --max-dir-mb N      hard ceiling per checkpoint dir, MB (default: 12000)
#   --no-clean          do not start the cleaner
#   --dry-run           print what would run, touch nothing
#
# Safe to re-run: a run that already finished is skipped via its .done
# marker. Delete the marker (or the run directory) to force a redo.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FEDASYNC_DIR="$(dirname "$HERE")"
REPO_ROOT="$(cd "$FEDASYNC_DIR/../../.." && pwd)"
CLEANER="$FEDASYNC_DIR/gpu_tiers/clean_checkpoints.sh"

PYTHON="${PYTHON:-python3}"
KEEP_ROUNDS=50
CLEAN_INTERVAL=3
MAX_DIR_MB=12000
DO_CLEAN=true
DRY_RUN=false
RUNS=()

ALL_RUNS=(baseline_tier_spec_vmax_distance baseline_tier_spec_oracle_exit)

while [[ $# -gt 0 ]]; do
    case "$1" in
        all) RUNS+=("${ALL_RUNS[@]}"); shift ;;
        baseline_tier_spec_vmax_distance|baseline_tier_spec_oracle_exit)
            RUNS+=("$1"); shift ;;
        --keep-rounds)    KEEP_ROUNDS="$2"; shift 2 ;;
        --clean-interval) CLEAN_INTERVAL="$2"; shift 2 ;;
        --max-dir-mb)     MAX_DIR_MB="$2"; shift 2 ;;
        --no-clean)       DO_CLEAN=false; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        *) echo "unknown argument '$1'" >&2; exit 2 ;;
    esac
done

if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "ERROR: name at least one run, or 'all'." >&2
    exit 2
fi

LOG_DIR="$HERE/logs"
RUNS_DIR="$FEDASYNC_DIR/runs/sumo_close_loop"
CONFIG_DIR="$HERE/configs/stage5b"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] [run_stage5b] $*"; }

CLEANER_PIDS=()
RUN_PIDS=()

cleanup() {
    for pid in "${CLEANER_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
    done
    for pid in "${RUN_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
    done
}
trap 'log "interrupted — stopping runs and cleaners"; cleanup; exit 130' INT TERM

log "runs: ${RUNS[*]}"

start_run() {
    local name="$1"
    local config="$CONFIG_DIR/${name}.toml"
    local run_dir="$RUNS_DIR/stage5b_${name}"
    local log_file="$LOG_DIR/stage5b_${name}.log"
    local done_marker="$LOG_DIR/stage5b_${name}.done"

    if [[ ! -f "$config" ]]; then
        log "$name: config $config is missing — run make_stage5b_configs.py first"
        return 1
    fi
    if [[ -f "$done_marker" ]]; then
        log "$name: already finished (delete $done_marker to redo) — skipping"
        return 0
    fi

    if $DRY_RUN; then
        log "$name: would run with config $config, log $log_file"
        return 0
    fi

    if $DO_CLEAN; then
        mkdir -p "$run_dir/checkpoints"
        "$CLEANER" "$run_dir/checkpoints" "$KEEP_ROUNDS" "$CLEAN_INTERVAL" \
            "$MAX_DIR_MB" >> "$LOG_DIR/stage5b_${name}.cleaner.log" 2>&1 &
        CLEANER_PIDS+=("$!")
        log "$name: cleaner started (pid $!, keep $KEEP_ROUNDS, every ${CLEAN_INTERVAL}m)"
    fi

    (
        cd "$FEDASYNC_DIR" || exit 1
        PYTHONPATH="$REPO_ROOT:$FEDASYNC_DIR/gpu_tiers:$FEDASYNC_DIR:$HERE" \
            "$PYTHON" "$HERE/sumo_fedasync.py" -c "$config" > "$log_file" 2>&1
        status=$?
        if [[ $status -eq 0 ]]; then
            date '+%F %T' > "$done_marker"
        fi
        exit $status
    ) &
    RUN_PIDS+=("$!")
    log "$name: started (pid $!) -> $log_file"
}

for name in "${RUNS[@]}"; do
    start_run "$name"
done

$DRY_RUN && exit 0

failures=0
for pid in "${RUN_PIDS[@]:-}"; do
    wait "$pid" || failures=$((failures + 1))
done
cleanup

log "all runs finished; $failures failure(s)"
for name in "${RUNS[@]}"; do
    line=$(grep -h '\[Stage0\] summary' "$LOG_DIR/stage5b_${name}.log" 2>/dev/null | tail -1)
    log "$name: ${line:-no stage0 summary found}"
done

exit $failures
