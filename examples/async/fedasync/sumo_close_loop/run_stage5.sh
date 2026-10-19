#!/bin/bash
# Run one or more Stage 5 configs (configs/stage5/*.toml), in parallel, each
# with its own checkpoint cleaner. Mirrors run_sumo.sh's conventions (cleaner,
# .done marker, log naming) for the configs/stage5 layout run_sumo.sh's own
# ${arm}_${CELL}${TAG} path pattern does not cover -- see
# make_stage5_configs.py and prompts/STAGE5_stage0_filter.md, Step 2.
#
# Usage:
#   ./run_stage5.sh <run> [run ...] [options]
#
#   run   one of: baseline_fleet_mean baseline_tier_spec baseline_oracle
#                 gpu_tier_spec cpu_tier_spec observed_fleet_mean
#         or "all" for every one of the six
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

ALL_RUNS=(baseline_fleet_mean baseline_tier_spec baseline_oracle gpu_tier_spec cpu_tier_spec observed_fleet_mean)

while [[ $# -gt 0 ]]; do
    case "$1" in
        all) RUNS+=("${ALL_RUNS[@]}"); shift ;;
        baseline_fleet_mean|baseline_tier_spec|baseline_oracle|gpu_tier_spec|cpu_tier_spec|observed_fleet_mean)
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
CONFIG_DIR="$HERE/configs/stage5"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] [run_stage5] $*"; }

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
    local run_dir="$RUNS_DIR/stage5_${name}"
    local log_file="$LOG_DIR/stage5_${name}.log"
    local done_marker="$LOG_DIR/stage5_${name}.done"

    if [[ ! -f "$config" ]]; then
        log "$name: config $config is missing — run make_stage5_configs.py first"
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
            "$MAX_DIR_MB" >> "$LOG_DIR/stage5_${name}.cleaner.log" 2>&1 &
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
    line=$(grep -h '\[Stage0\] summary' "$LOG_DIR/stage5_${name}.log" 2>/dev/null | tail -1)
    log "$name: ${line:-no stage0 summary found}"
done

exit $failures
