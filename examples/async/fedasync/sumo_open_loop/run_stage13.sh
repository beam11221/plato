#!/bin/bash
# Stage 13 (CIFAR-100 pilot): six runs, seed 4126, queued behind Stage 11 Phase 2.
#
#   ./run_stage13.sh                    Stage 13 (4,000 rounds)
#   ./run_stage13.sh stage13b 10000     Stage 13B: the same six at 10,000 rounds
#                                       (configs from make_stage13_configs.py --stage stage13b)
#
#   0. wait until Stage 11's driver has started its last Phase 2 job (34 started)
#      or has exited (any ending); from then on Stage 11 starts nothing new
#   1. run the six cells, order #2 #3 (first, per spec) #1 #4 #5 #6 (replicate last),
#      never more than FOUR training runs on the machine in total (Stage 11's
#      tail counts against the limit)
#   2. after each run: stage13_runcheck.py (soundness only; plateau rule disabled).
#      A failure is LOGGED and the queue CONTINUES (spec)
#   3. stage13_report.py writes STAGE13_pilot_report.md + the series CSVs
#
# Every job runs in its own process group (setsid) with its own checkpoint cleaner
# (run_stage7.sh); a finished run's checkpoints are deleted immediately.
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$(dirname "$HERE")/runs/sumo_open_loop"
cd "$HERE"
STAGE="${1:-stage13}"; ROUNDS="${2:-4000}"; S="${STAGE#stage}"
log() { echo "[$(date '+%F %T')] [run_${STAGE}] $*"; }
export PYTHONPATH="$(cd ../../../.. && pwd):$(cd .. && pwd):$(cd ../gpu_tiers && pwd):$(cd ../gpu_tiers_dataset && pwd):$(cd ../sumo_close_loop && pwd):$HERE"
LIMIT=4

JOBS=("${S}f_s4126 baseline_open #2" "${S}f_s4126 gpu_open #3" "${S}nf_s4126 baseline_open #1"
      "${S}f_s4126 observed_open #4" "${S}f_s4126 cpu_open #5" "${S}rep_s4126 baseline_open #6")

training_runs() { ps -eo args | grep -c "[s]umo_open_fedasync.py -c"; }
stage11_driver_alive() { ps -eo args | awk '$1=="/bin/bash" && $2 ~ /run_stage11\.sh$/' | grep -q .; }

log "waiting for Stage 11 Phase 2 to have started all 34 runs (or its driver to exit)"
while stage11_driver_alive; do
    n=$(grep -cE "started stage11(f|nf)_p[0-9]{3}_s412[78]_[a-z]+_ret " logs/run_stage11.log)
    (( n >= 34 )) && break
    sleep 300
done
log "Stage 11 starts nothing more ($(grep -oE 'STAGE11_[A-Z_]+|PHASE1_DONE' logs/run_stage11.log | tail -1))"

declare -A RUNNING=()
finish() {
    local pg="$1" job="${RUNNING[$1]}"
    unset "RUNNING[$pg]"
    read -r key run num <<< "$job"
    rm -rf "$RUNS_DIR/stage${key}_${run}/checkpoints"
    if [[ ! -f "logs/stage${key}_${run}.done" ]]; then
        log "FAILED (no .done): stage${key}_${run} $num (checkpoints removed) -- continuing"
        echo "stage${key}_${run}" >> "logs/${STAGE}_failed_runs"
        return
    fi
    if python3 stage13_runcheck.py "stage${key}" "$run" "$ROUNDS" > "logs/stage13_runcheck.stage${key}_${run}.out" 2>&1; then
        log "done + cleaned + soundness PASS: stage${key}_${run} $num"
    else
        log "SOUNDNESS FAIL (reported, continuing): stage${key}_${run} $num ($(tail -1 logs/stage13_runcheck.stage${key}_${run}.out | cut -c1-300))"
    fi
}

next=0
log "${STAGE}: ${#JOBS[@]} runs, ${ROUNDS} rounds each"
while :; do
    while (( next < ${#JOBS[@]} )) && (( $(training_runs) < LIMIT )); do
        read -r key run num <<< "${JOBS[$next]}"; next=$((next + 1))
        setsid ./run_stage7.sh --stage "$key" "$run" >> "logs/run_${STAGE}_queue.${key}_${run}.log" 2>&1 &
        RUNNING[$!]="$key $run $num"
        log "started stage${key}_${run} $num (pgid $!)"
        sleep 20   # let the process appear before counting again
    done
    (( ${#RUNNING[@]} == 0 && next >= ${#JOBS[@]} )) && break
    sleep 60
    for pg in "${!RUNNING[@]}"; do
        kill -0 "$pg" 2>/dev/null || finish "$pg"
    done
done
log "all six finished"
python3 stage13_report.py --stage "$STAGE" --rounds "$ROUNDS" > "logs/${STAGE}_report.out" 2>&1 \
    && log "report written ($(tail -1 logs/${STAGE}_report.out | cut -c1-200))" || log "stage13_report.py FAILED (see logs/${STAGE}_report.out)"
log "${STAGE^^}_ALL_DONE"
