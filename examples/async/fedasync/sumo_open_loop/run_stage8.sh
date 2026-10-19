#!/bin/bash
# Stage 8: 12 runs (baseline/gpu/observed x open/closed x seeds 4127/4128),
# never more than four at once, in three waves.  Each wave's checkpoints are
# deleted as soon as the wave finishes (logs and results/ are kept).
# Seed 4126 is Stage 7b and is NOT re-run.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$(dirname "$HERE")/runs/sumo_open_loop"
cd "$HERE"
log() { echo "[$(date '+%F %T')] [run_stage8] $*"; }
clean() {  # clean <stage-key> <run...>
    local key="$1"; shift
    for r in "$@"; do
        if [[ -f "logs/stage${key}_${r}.done" ]]; then
            rm -rf "$RUNS_DIR/stage${key}_${r}/checkpoints"
            log "cleaned checkpoints: stage${key}_${r}"
        else
            log "NOT cleaned (no .done): stage${key}_${r}"
        fi
    done
}

log "wave 1: s4127 baseline_open gpu_open observed_open baseline_closed"
./run_stage7.sh --stage 8_s4127 baseline_open gpu_open observed_open baseline_closed
clean 8_s4127 baseline_open gpu_open observed_open baseline_closed

log "wave 2: s4127 gpu_closed observed_closed + s4128 baseline_open gpu_open"
./run_stage7.sh --stage 8_s4127 gpu_closed observed_closed &
./run_stage7.sh --stage 8_s4128 baseline_open gpu_open &
wait
clean 8_s4127 gpu_closed observed_closed
clean 8_s4128 baseline_open gpu_open

log "wave 3: s4128 observed_open baseline_closed gpu_closed observed_closed"
./run_stage7.sh --stage 8_s4128 observed_open baseline_closed gpu_closed observed_closed
clean 8_s4128 observed_open baseline_closed gpu_closed observed_closed

log "STAGE8_ALL_WAVES_DONE"
