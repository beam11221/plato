#!/bin/bash
# Run one or more arms of the SUMO experiment, in parallel, each with its own
# checkpoint cleaner.
#
# Usage:
#   ./run_sumo.sh <arm> [arm ...] [options]
#
#   arm                 baseline | gpu | cpu | observed
#   --cell n20|n100     fleet size            (default: n20)
#   --tag STR           config/run suffix, for a short test build (default: none)
#   --keep-rounds N     checkpoints to keep   (default: 50)
#   --clean-interval N  cleaner period, minutes (default: 3)
#   --max-dir-mb N      hard ceiling per checkpoint dir, MB (default: 12000)
#   --quota-gb N        this mount's allocation, for the usage warning (default: 200)
#   --no-clean          do not start the cleaner (you will fill the disk)
#   --dry-run           print what would run, touch nothing
#
# Two arms per machine
# --------------------
# The usual split is two machines, two arms each -- this box runs `cpu gpu`, the
# other runs `baseline observed`. Naming both here runs them concurrently, so
# they share the GPU and the disk quota. Naming one runs one.
#
# DISK IS THE BINDING CONSTRAINT: this mount is a 200 GB allocation, and a full
# 10 000-round arm writes ~85 MB per dispatch cycle, i.e. about 1.1 TB if nothing
# deletes it. The cleaner is not optional here the way it is on a bigger volume.
# With the defaults each running arm holds roughly
#
#     (KEEP_ROUNDS + cycles_per_interval) * 85 MB
#
# which is ~10-15 GB per arm at a 3-minute period. --max-dir-mb is the backstop
# if the rate is higher than expected.
#
# Safe to re-run: an arm that already finished is skipped via its .done marker.
# Delete the marker (or the run directory) to force a redo.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FEDASYNC_DIR="$(dirname "$HERE")"
REPO_ROOT="$(cd "$FEDASYNC_DIR/../../.." && pwd)"
CLEANER="$FEDASYNC_DIR/gpu_tiers/clean_checkpoints.sh"

PYTHON="${PYTHON:-python3}"
CELL="n20"
TAG=""
KEEP_ROUNDS=50
CLEAN_INTERVAL=3
MAX_DIR_MB=12000
QUOTA_GB=200
DO_CLEAN=true
DRY_RUN=false
ARMS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        baseline|gpu|cpu|observed) ARMS+=("$1"); shift ;;
        --cell)           CELL="$2"; shift 2 ;;
        --tag)            TAG="$2"; shift 2 ;;
        --keep-rounds)    KEEP_ROUNDS="$2"; shift 2 ;;
        --clean-interval) CLEAN_INTERVAL="$2"; shift 2 ;;
        --max-dir-mb)     MAX_DIR_MB="$2"; shift 2 ;;
        --quota-gb)       QUOTA_GB="$2"; shift 2 ;;
        --no-clean)       DO_CLEAN=false; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        *) echo "unknown argument '$1'" >&2; exit 2 ;;
    esac
done

if [[ ${#ARMS[@]} -eq 0 ]]; then
    echo "ERROR: name at least one arm (baseline gpu cpu observed)." >&2
    exit 2
fi

LOG_DIR="$HERE/logs"
RUNS_DIR="$FEDASYNC_DIR/runs/sumo_close_loop"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] [run_sumo] $*"; }

# Track every cleaner we start so none survives this script.
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

# ---------------------------------------------------------------------------
# Preflight, once, before anything expensive starts
# ---------------------------------------------------------------------------

log "preflight for cell $CELL"
if ! "$PYTHON" "$HERE/preflight.py" --cell "$CELL"; then
    log "preflight FAILED — refusing to run"
    exit 1
fi

# df reports the shared cluster filesystem, not this box's allocation, so it is
# useless here. Track what we actually occupy against the stated quota instead.
USED_GB=$(du -sBG "$FEDASYNC_DIR/runs" 2>/dev/null | cut -f1 | tr -dc '0-9')
USED_GB="${USED_GB:-0}"
log "arms: ${ARMS[*]}   cell: $CELL   tag: '${TAG:-none}'"
log "runs/ currently holds ${USED_GB} GB of the ${QUOTA_GB} GB allocation"
if (( USED_GB > QUOTA_GB * 7 / 10 )); then
    log "WARNING: over 70% of the allocation is already used before starting."
    log "         Old runs under $FEDASYNC_DIR/runs are the usual culprit."
fi

# ---------------------------------------------------------------------------
# One arm: cleaner + run + .done marker
# ---------------------------------------------------------------------------

start_arm() {
    local arm="$1"
    local name="${arm}_${CELL}${TAG}"
    local config="$HERE/configs/${name}.toml"
    local run_dir="$RUNS_DIR/${name}"
    local log_file="$LOG_DIR/${name}.log"
    local done_marker="$LOG_DIR/${name}.done"

    if [[ ! -f "$config" ]]; then
        log "$name: config $config is missing — run build_inputs.sh first"
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
            "$MAX_DIR_MB" >> "$LOG_DIR/${name}.cleaner.log" 2>&1 &
        CLEANER_PIDS+=("$!")
        log "$name: cleaner started (pid $!, keep $KEEP_ROUNDS, every ${CLEAN_INTERVAL}m)"
    fi

    (
        cd "$FEDASYNC_DIR" || exit 1
        # gpu_tiers first: the entry points import gpu_tier_sleep bare.
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

for arm in "${ARMS[@]}"; do
    start_arm "$arm"
done

$DRY_RUN && exit 0

# ---------------------------------------------------------------------------
# Wait, then report
# ---------------------------------------------------------------------------

failures=0
for pid in "${RUN_PIDS[@]:-}"; do
    wait "$pid" || failures=$((failures + 1))
done
cleanup

log "all runs finished; $failures failure(s)"
for arm in "${ARMS[@]}"; do
    name="${arm}_${CELL}${TAG}"
    line=$(grep -h '\[SumoRsu\] waste' "$LOG_DIR/${name}.log" 2>/dev/null | tail -1)
    log "$name: ${line:-no waste summary found}"
done

exit $failures
