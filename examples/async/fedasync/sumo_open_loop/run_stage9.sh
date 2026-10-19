#!/bin/bash
# Stage 9: 23 runs through a four-slot queue (never more than four at once).
#
#   1 invariance   stage9_s4126 baseline_open           (Step 2; reproduces 7b)
#  18 neutral      stage9_s<seed> {gpu,cpu,observed}_{open,closed}, seeds 4126-4128
#   4 alpha = 10   stage8_s<seed> cpu_{open,closed}, seeds 4127 / 4128 (Stage 8's
#                  missing cpu partners)
#
# The invariance run goes first.  When it finishes, stage9_invariance.py
# compares it with Stage 7b; on ANY difference the queue stops launching (a
# stop condition) and logs STAGE9_STOPPED.  Each run's checkpoints are deleted
# as soon as it finishes; logs and results/ are kept.
set -o pipefail  # no -u: bash treats an empty associative array as unbound
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$(dirname "$HERE")/runs/sumo_open_loop"
cd "$HERE"
log() { echo "[$(date '+%F %T')] [run_stage9] $*"; }

JOBS=("9_s4126 baseline_open")
for seed in 4126 4127 4128; do
    for run in gpu_open cpu_open observed_open gpu_closed cpu_closed observed_closed; do
        JOBS+=("9_s${seed} ${run}")
    done
done
for seed in 4127 4128; do
    JOBS+=("8_s${seed} cpu_open" "8_s${seed} cpu_closed")
done

declare -A RUNNING=()   # pid -> "stage run"
STOP=false
next=0

finish() {  # finish <pid>
    local job="${RUNNING[$1]}"; unset "RUNNING[$1]"
    local key="${job% *}" run="${job#* }"
    if [[ -f "logs/stage${key}_${run}.done" ]]; then
        rm -rf "$RUNS_DIR/stage${key}_${run}/checkpoints"
        log "done + cleaned: stage${key}_${run}"
    else
        log "FAILED (no .done): stage${key}_${run}"
    fi
    if [[ "$job" == "9_s4126 baseline_open" && -f logs/stage9_invariance.accepted ]]; then
        log "baseline invariance: decision on file (logs/stage9_invariance.accepted): $(head -1 logs/stage9_invariance.accepted)"
    elif [[ "$job" == "9_s4126 baseline_open" ]]; then
        if python3 stage9_invariance.py > logs/stage9_invariance.out 2>&1; then
            log "baseline invariance: PASS (identical to Stage 7b)"
        else
            log "baseline invariance: FAIL -- stopping the queue (stop condition)"
            STOP=true
            # Each job runs in its own process group (setsid below), so this
            # reaches the python server and its workers, not just the wrapper.
            # On 2026-09-25 killing only the wrapper left two runs training for
            # 8 h with their cleaners gone: ~130 GB of checkpoints.
            for other in "${!RUNNING[@]}"; do
                kill -TERM -- "-$other" 2>/dev/null && log "stopped ${RUNNING[$other]} (pgid $other)"
            done
        fi
    fi
}

log "queue of ${#JOBS[@]} runs"
while :; do
    while ! $STOP && (( ${#RUNNING[@]} < 4 )) && (( next < ${#JOBS[@]} )); do
        job="${JOBS[$next]}"; next=$((next + 1))
        setsid ./run_stage7.sh --stage "${job% *}" "${job#* }" \
            >> "logs/run_stage9_queue.${job// /_}.log" 2>&1 &
        RUNNING[$!]="$job"
        log "started stage${job% *}_${job#* } (pid $!)"
    done
    (( ${#RUNNING[@]} == 0 )) && break
    wait -n
    for pid in "${!RUNNING[@]}"; do
        kill -0 "$pid" 2>/dev/null || finish "$pid"
    done
done

if $STOP; then log "STAGE9_STOPPED"; else log "STAGE9_ALL_DONE"; fi
