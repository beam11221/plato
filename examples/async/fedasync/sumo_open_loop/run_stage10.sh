#!/bin/bash
# Stage 10: 30 + 18 runs through a four-slot queue (never more than four at once).
#
# Wave A, in the spec's launch order (headline first):
#   A1  open    baseline+F, gpu+F                               (stage10f1)  6
#   A2  open    observed+F, cpu+F (stage10f1), baseline/none (stage10nf)     9
#   A3  closed  the same five arms                                          15
# Wave B (stage10f10, alpha = 10 with the filter): gpu, observed, cpu x 2 x 3  18
#
# After every run, stage10_runcheck.py checks the stop conditions:
#   exit 1 -> stop the stage: kill running jobs (whole process groups), stop.
#   exit 2 -> baseline/none drifted > 3% from 7b/8: Wave B is blocked.
# Wave B starts only when logs/stage10_waveB.go exists -- written once
# STAGE10A_report.md is written (the spec's order) -- and nothing blocks it.
# Every job runs in its own process group (setsid) with its own cleaner
# (run_stage7.sh); a finished run's checkpoints are deleted immediately.
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$(dirname "$HERE")/runs/sumo_open_loop"
cd "$HERE"
log() { echo "[$(date '+%F %T')] [run_stage10] $*"; }

SEEDS=(4126 4127 4128)
WAVE_A=()
for s in "${SEEDS[@]}"; do WAVE_A+=("10f1_s$s baseline_open" "10f1_s$s gpu_open"); done
for s in "${SEEDS[@]}"; do WAVE_A+=("10f1_s$s observed_open" "10f1_s$s cpu_open" "10nf_s$s baseline_open"); done
for s in "${SEEDS[@]}"; do
    for a in baseline gpu observed cpu; do WAVE_A+=("10f1_s$s ${a}_closed"); done
    WAVE_A+=("10nf_s$s baseline_closed")
done
WAVE_B=()
for s in "${SEEDS[@]}"; do
    for r in gpu_open observed_open cpu_open gpu_closed observed_closed cpu_closed; do
        WAVE_B+=("10f10_s$s $r")
    done
done

declare -A RUNNING=()
STOP=false

finish() {  # finish <pgid>
    local job="${RUNNING[$1]}"; unset "RUNNING[$1]"
    local key="${job% *}" run="${job#* }"
    if [[ ! -f "logs/stage${key}_${run}.done" ]]; then
        rm -rf "$RUNS_DIR/stage${key}_${run}/checkpoints"
        # A crash is not one of the spec's stop conditions: record it, keep the
        # queue going (the watcher reports it) -- it can be re-queued later.
        log "FAILED (no .done): stage${key}_${run} (checkpoints removed)"
        echo "stage${key}_${run}" >> logs/stage10_failed_runs
    else
        rm -rf "$RUNS_DIR/stage${key}_${run}/checkpoints"
        python3 stage10_runcheck.py "stage${key}" "$run" > "logs/stage10_runcheck.stage${key}_${run}.out" 2>&1
        case $? in
            0) log "done + cleaned + checks PASS: stage${key}_${run}" ;;
            2) log "done + cleaned; BLOCKS WAVE B: stage${key}_${run} ($(tail -1 logs/stage10_runcheck.stage${key}_${run}.out | cut -c1-200))"
               echo "stage${key}_${run}" >> logs/stage10_waveB.blocked ;;
            *) log "STOP CONDITION: stage${key}_${run} ($(tail -1 logs/stage10_runcheck.stage${key}_${run}.out | cut -c1-300))"
               STOP=true ;;
        esac
    fi
    if $STOP; then
        for other in "${!RUNNING[@]}"; do
            kill -TERM -- "-$other" 2>/dev/null && log "stopped ${RUNNING[$other]} (pgid $other)"
        done
    fi
}

run_queue() {  # run_queue <name> <jobs...>
    local name="$1"; shift
    local jobs=("$@") next=0
    log "$name: ${#jobs[@]} runs"
    while :; do
        while ! $STOP && (( ${#RUNNING[@]} < 4 )) && (( next < ${#jobs[@]} )); do
            local job="${jobs[$next]}"; next=$((next + 1))
            setsid ./run_stage7.sh --stage "${job% *}" "${job#* }" \
                >> "logs/run_stage10_queue.${job// /_}.log" 2>&1 &
            RUNNING[$!]="$job"
            log "started stage${job% *}_${job#* } (pgid $!)"
        done
        (( ${#RUNNING[@]} == 0 )) && break
        wait -n
        for pid in "${!RUNNING[@]}"; do
            kill -0 "$pid" 2>/dev/null || finish "$pid"
        done
    done
}

log "Stage 10: Wave A ${#WAVE_A[@]} runs, Wave B ${#WAVE_B[@]} runs"
run_queue "Wave A" "${WAVE_A[@]}"
if $STOP; then log "STAGE10_STOPPED (Wave A)"; exit 1; fi
log "WAVE_A_DONE"
# The spec's order: write STAGE10A_report.md, then start Wave B without
# waiting.  stage10_waveA_report.py runs the Wave A analysis, writes the report
# and, if no stop condition or block is present, writes logs/stage10_waveB.go.
# If it is missing or fails, the queue waits here for the go-file by hand.
if [[ -f stage10_waveA_report.py ]]; then
    if python3 stage10_waveA_report.py > logs/stage10_waveA_report.out 2>&1; then
        log "STAGE10A_report.md written ($(tail -1 logs/stage10_waveA_report.out | cut -c1-160))"
    else
        log "stage10_waveA_report.py FAILED -- waiting for logs/stage10_waveB.go by hand"
    fi
fi
log "waiting for logs/stage10_waveB.go"

until [[ -f logs/stage10_waveB.go ]]; do
    [[ -f logs/stage10_waveB.blocked ]] && { log "STAGE10_WAVE_B_BLOCKED: $(tr '\n' ' ' < logs/stage10_waveB.blocked)"; exit 2; }
    sleep 60
done
[[ -f logs/stage10_waveB.blocked ]] && { log "STAGE10_WAVE_B_BLOCKED: $(tr '\n' ' ' < logs/stage10_waveB.blocked)"; exit 2; }
log "go-file found: $(head -1 logs/stage10_waveB.go)"
run_queue "Wave B" "${WAVE_B[@]}"
if $STOP; then log "STAGE10_STOPPED (Wave B)"; exit 1; fi
log "STAGE10_ALL_DONE"
