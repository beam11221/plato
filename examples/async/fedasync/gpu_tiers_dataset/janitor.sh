#!/bin/bash
# Keep runs/gpu_tiers_dataset/ inside the disk allocation, for as long as the
# Stage 3 sweep runs.
#
#   ./janitor.sh [--interval-min N] [--quota-gb N] [--warn-pct N] [--once] [--dry-run]
#
# Ported from ../separability/janitor.sh, which this project already had a
# working answer for. Same design, same guarantees; only the paths differ.
#
# WHY THIS EXISTS ON TOP OF THE PER-RUN CLEANER
# ---------------------------------------------
# `run_sweep.sh` starts `gpu_tiers/clean_checkpoints.sh` for each run and kills
# it when that run ends, which holds a LIVE run at a bounded size (default:
# newest 50 rounds). It does nothing about runs that have already FINISHED, and
# it deliberately never touches the per-client payload files --
# `comm_simulation = true` hands each dispatched client its model through
# `checkpoints/resnet_18_client_<id>.pkl`, and deleting one under a running
# client would break that round.
#
# So a finished run leaves its last-kept checkpoint window behind (several GB),
# and 23 runs of those add up. This loop reclaims it: once a run's `.done`
# marker exists, NOTHING in its checkpoints/ is needed again, so the directory
# is emptied.
#
# What it never touches: `results/` (the accuracy curves),
# `config_used.toml` (what the run actually ran with -- clock_reconstruction.py
# needs it), `*.log` (the dispatch, merge and staleness record), `models/`
# (the final weights) -- those ARE the analysis data -- and any run directory
# outside runs/gpu_tiers_dataset/.
#
# The one consequence to know: a finished run whose checkpoints have been
# pruned cannot `--resume`. Deleting its `.done` marker to force a redo
# therefore starts it from round 1, which is what a redo should do anyway.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FEDASYNC_DIR="$(dirname "$HERE")"
RUNS_DIR="$FEDASYNC_DIR/runs/gpu_tiers_dataset"
LOG_DIR="$HERE/logs"

INTERVAL_MIN=5
# 2026-09-14: this folder's own runs/ hit ~78 GB (61.5 GB of it dead-weight
# checkpoints from hand-run verification builds with no cleaner attached) while
# the WHOLE runs/ tree -- shared with separability, separability_n100,
# sumo_close_loop, surgical -- was at ~156 GB and the user reported the volume
# at 100%. Confirmed by the user the same day: the pod's real allocation is
# 200 GB, shared with things on /workspace outside examples/async/fedasync/
# entirely -- this script and the disk-usage numbers it prints only ever see
# $FEDASYNC_DIR/runs, so a quota breach that comes from elsewhere on the pod
# will NOT show up as "over Nx% of QUOTA_GB" here even though it eats the same
# 200 GB.
QUOTA_GB=200
WARN_PCT=70
ONCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval-min) INTERVAL_MIN="$2"; shift 2 ;;
        --quota-gb)     QUOTA_GB="$2"; shift 2 ;;
        --warn-pct)     WARN_PCT="$2"; shift 2 ;;
        --once)         ONCE=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        *) echo "unknown argument '$1'" >&2; exit 2 ;;
    esac
done

log() { echo "[$(date '+%F %T')] [janitor] $*"; }

mkdir -p "$RUNS_DIR" "$LOG_DIR"

# Guard: every deletion below is inside this directory, so refuse to start if it
# is not the path this script was written for.
case "$RUNS_DIR" in
    */runs/gpu_tiers_dataset) : ;;
    *) log "refusing to run: '$RUNS_DIR' is not .../runs/gpu_tiers_dataset"; exit 2 ;;
esac

gb_of() {  # directory -> integer GB, 0 if missing
    local size
    size=$(du -sBG "$1" 2>/dev/null | cut -f1 | tr -dc '0-9')
    echo "${size:-0}"
}

prune_finished() {
    local pruned=0
    shopt -s nullglob
    for run_dir in "$RUNS_DIR"/*/; do
        local name checkpoints
        name="$(basename "$run_dir")"
        checkpoints="$run_dir/checkpoints"
        [[ -d "$checkpoints" ]] || continue
        # Only a run with a .done marker is finished. A run still in flight is
        # the per-run cleaner's business, and its payload files are live data.
        [[ -f "$LOG_DIR/${name}.done" ]] || continue
        # -mindepth 1: empty the directory, never remove it.
        if [[ -z "$(find "$checkpoints" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            continue
        fi
        local freed
        freed=$(gb_of "$checkpoints")
        if $DRY_RUN; then
            log "would empty $name/checkpoints (~${freed} GB)"
        else
            find "$checkpoints" -mindepth 1 -delete 2>/dev/null
            log "emptied $name/checkpoints — reclaimed ~${freed} GB (run is done)"
        fi
        pruned=$((pruned + 1))
    done
    shopt -u nullglob
    return 0
}

report_usage() {
    local runs_gb all_gb
    runs_gb=$(gb_of "$RUNS_DIR")
    all_gb=$(gb_of "$FEDASYNC_DIR/runs")
    log "usage: gpu_tiers_dataset ${runs_gb} GB, all runs/ ${all_gb} GB of ${QUOTA_GB} GB"
    if (( all_gb > QUOTA_GB * WARN_PCT / 100 )); then
        log "WARNING: over ${WARN_PCT}% of the allocation is used."
        log "         Largest directories under runs/:"
        du -sh "$FEDASYNC_DIR"/runs/* 2>/dev/null | sort -rh | head -5 | \
            while read -r line; do log "           $line"; done
    fi
}

log "started (interval ${INTERVAL_MIN} min, quota ${QUOTA_GB} GB, dry-run $DRY_RUN)"
while true; do
    prune_finished
    report_usage
    $ONCE && break
    sleep $(( INTERVAL_MIN * 60 ))
done
