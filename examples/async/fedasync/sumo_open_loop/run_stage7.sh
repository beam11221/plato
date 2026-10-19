#!/bin/bash
# Run one or more Stage 7 / 7b configs (configs/stage<X>/*.toml), in parallel, each
# with its own checkpoint cleaner. Mirrors run_stage5b.sh's conventions, but
# runs through sumo_open_fedasync.py (every Stage 6 config needs its seed
# wiring, unlike earlier stages).
#
# Usage:
#   ./run_stage7.sh <run> [run ...] [options]
#
#   run   a config's basename without .toml, e.g. gpu_tier_spec_s4126,
#         or "batch-a" / "batch-b" / "all"
#   --keep-rounds N     checkpoints to keep   (default: 50)
#   --clean-interval N  cleaner period, minutes (default: 3)
#   --max-dir-mb N      hard ceiling per checkpoint dir, MB (default: 12000)
#   --no-clean          do not start the cleaner
#   --dry-run           print what would run, touch nothing
#   --stage X           7 (default), 7b, or 7b_preflight -- selects
#                       configs/stage<X>/ and names logs/runs stage<X>_*
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

STAGE="7"

WAVE_1=(baseline_open gpu_open cpu_open observed_open)
WAVE_2=(baseline_closed gpu_closed cpu_closed observed_closed)

while [[ $# -gt 0 ]]; do
    case "$1" in
        wave-1) RUNS+=("${WAVE_1[@]}"); shift ;;
        wave-2) RUNS+=("${WAVE_2[@]}"); shift ;;
        # No "all": 8 at once exhausted the GPU in Stage 6 and hung four runs
        # silently for 17 hours.  Two waves of four, always.
        --keep-rounds)    KEEP_ROUNDS="$2"; shift 2 ;;
        --clean-interval) CLEAN_INTERVAL="$2"; shift 2 ;;
        --max-dir-mb)     MAX_DIR_MB="$2"; shift 2 ;;
        --no-clean)       DO_CLEAN=false; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --stage)          STAGE="$2"; shift 2 ;;
        *) RUNS+=("$1"); shift ;;
    esac
done

if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "ERROR: name at least one run, or 'wave-1' / 'wave-2'." >&2
    exit 2
fi

PREFIX="stage${STAGE}"
CONFIG_DIR="$HERE/configs/$PREFIX"
if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "ERROR: $CONFIG_DIR missing -- run make_stage7_configs.py" >&2
    exit 2
fi
LOG_DIR="$HERE/logs"
RUNS_DIR="$FEDASYNC_DIR/runs/sumo_open_loop"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] [run_stage7] $*"; }

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
    local run_dir="$RUNS_DIR/${PREFIX}_${name}"
    local log_file="$LOG_DIR/${PREFIX}_${name}.log"
    local done_marker="$LOG_DIR/${PREFIX}_${name}.done"

    if [[ ! -f "$config" ]]; then
        log "$name: config $config is missing — run make_stage7_configs.py first"
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
            "$MAX_DIR_MB" >> "$LOG_DIR/${PREFIX}_${name}.cleaner.log" 2>&1 &
        CLEANER_PIDS+=("$!")
        log "$name: cleaner started (pid $!, keep $KEEP_ROUNDS, every ${CLEAN_INTERVAL}m)"
    fi

    (
        cd "$FEDASYNC_DIR" || exit 1
        PYTHONPATH="$REPO_ROOT:$FEDASYNC_DIR/gpu_tiers:$FEDASYNC_DIR/gpu_tiers_dataset:$FEDASYNC_DIR:$FEDASYNC_DIR/sumo_close_loop:$HERE" \
            "$PYTHON" "$HERE/sumo_open_fedasync.py" -c "$config" > "$log_file" 2>&1
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
    line=$(grep -h '\[Stage7\] summary' "$LOG_DIR/${PREFIX}_${name}.log" 2>/dev/null | tail -1)
    log "$name: ${line:-no stage0 summary found}"
done

exit $failures
