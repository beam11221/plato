#!/bin/bash
# Stage 11 (partial recurrence, p_return): the whole sequence, unattended.
#
#   0. wait until Stage 10 has finished (STAGE10_ALL_DONE in logs/run_stage10.log);
#      any other Stage 10 ending -> do not start
#   1. alpha gate: stage11_alpha_check.py (Wave B: alpha = 10 beating alpha = 1 for
#      gpu+F open in every seed -> stop and report)
#   2. pre-flight: 100-round smoke runs, gpu+F and baseline+F at each p_return,
#      seed 4126; preflight_stage11.py must pass at 100%
#   3. Phase 1: the 17 cells at run seed 4126, in the spec's order
#   4. stage11_phase1_report.py (if present) writes STAGE11_seed4126_report.md
#   5. STOP until logs/stage11_phase2.go exists (written by the user after review)
#   6. Phase 2: the 17 cells at 4127, then 4128 (4127's first three groups finish
#      before 4128 starts: with four slots, 4128's first job starts only after
#      4127's 17th has started, i.e. after its first 13 have finished)
#
# Never more than four runs at once.  Every job runs in its own process group
# (setsid) with its own checkpoint cleaner (run_stage7.sh); a finished run's
# checkpoints are deleted immediately.  After every run stage11_runcheck.py
# checks the stop conditions; a stop kills every running job group.
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$(dirname "$HERE")/runs/sumo_open_loop"
cd "$HERE"
log() { echo "[$(date '+%F %T')] [run_stage11] $*"; }
# The run check and the report import preflight_stage11 -> open_stage0 -> plato.
export PYTHONPATH="$(cd ../../../.. && pwd):$(cd .. && pwd):$(cd ../gpu_tiers && pwd):$(cd ../gpu_tiers_dataset && pwd):$(cd ../sumo_close_loop && pwd):$HERE"

p_key() { printf "p%03d" "$(python3 -c "print(round($1*100))")"; }

cells_for_seed() {  # emits "stage-key run p_return" lines in the spec's order
    local s="$1"
    for p in 0.5 0.25 0.75; do
        for a in gpu baseline observed; do echo "11f_$(p_key $p)_s$s ${a}_ret $p"; done
    done
    for a in gpu baseline; do echo "11f_$(p_key 1.0)_s$s ${a}_ret 1.0"; done
    for p in 0.25 0.5 0.75; do
        echo "11f_$(p_key $p)_s$s cpu_ret $p"
        echo "11nf_$(p_key $p)_s$s baseline_ret $p"
    done
}

declare -A RUNNING=()
declare -A PRET=()
STOP=false

finish() {
    local pg="$1" job="${RUNNING[$1]}" p="${PRET[$1]}"
    unset "RUNNING[$pg]" "PRET[$pg]"
    local key="${job% *}" run="${job#* }"
    rm -rf "$RUNS_DIR/stage${key}_${run}/checkpoints"
    if [[ ! -f "logs/stage${key}_${run}.done" ]]; then
        log "FAILED (no .done): stage${key}_${run} (checkpoints removed)"
        echo "stage${key}_${run}" >> logs/stage11_failed_runs
        return
    fi
    if [[ "$key" == *preflight* ]]; then
        log "done + cleaned (pre-flight): stage${key}_${run}"
        return
    fi
    if python3 stage11_runcheck.py "stage${key}" "$run" "$p" > "logs/stage11_runcheck.stage${key}_${run}.out" 2>&1; then
        log "done + cleaned + checks PASS: stage${key}_${run}"
    else
        log "STOP CONDITION: stage${key}_${run} ($(tail -1 logs/stage11_runcheck.stage${key}_${run}.out | cut -c1-300))"
        STOP=true
        for other in "${!RUNNING[@]}"; do
            kill -TERM -- "-$other" 2>/dev/null && log "stopped ${RUNNING[$other]} (pgid $other)"
        done
    fi
}

run_queue() {  # run_queue <name> <lines "key run p">...
    local name="$1"; shift
    local jobs=("$@") next=0
    log "$name: ${#jobs[@]} runs"
    while :; do
        while ! $STOP && (( ${#RUNNING[@]} < 4 )) && (( next < ${#jobs[@]} )); do
            read -r key run p <<< "${jobs[$next]}"; next=$((next + 1))
            setsid ./run_stage7.sh --stage "$key" "$run" >> "logs/run_stage11_queue.${key}_${run}.log" 2>&1 &
            RUNNING[$!]="$key $run"; PRET[$!]="$p"
            log "started stage${key}_${run} (pgid $!, p_return=$p)"
        done
        (( ${#RUNNING[@]} == 0 )) && break
        wait -n
        for pg in "${!RUNNING[@]}"; do
            kill -0 "$pg" 2>/dev/null || finish "$pg"
        done
    done
}

# -- 0. wait for Stage 10 -------------------------------------------------------
log "waiting for Stage 10 to finish"
until grep -qE "STAGE10_ALL_DONE|STAGE10_STOPPED|STAGE10_WAVE_B_BLOCKED" logs/run_stage10.log 2>/dev/null; do
    if ! ps -eo args | awk '$1=="/bin/bash" && $2 ~ /run_stage10\.sh$/' | grep -q .; then
        log "Stage 10 driver is gone without a final marker -- not starting"; log "STAGE11_NOT_STARTED"; exit 1
    fi
    sleep 300
done
if ! grep -q "STAGE10_ALL_DONE" logs/run_stage10.log; then
    log "Stage 10 did not finish cleanly ($(grep -oE 'STAGE10_[A-Z_]+' logs/run_stage10.log | tail -1)) -- not starting"
    log "STAGE11_NOT_STARTED"; exit 1
fi
while ps -eo args | grep -q "[s]umo_open_fedasync.py -c .*configs/stage10"; do sleep 60; done
log "Stage 10 finished"

# -- 1. alpha gate --------------------------------------------------------------
python3 stage11_alpha_check.py > logs/stage11_alpha_check.out 2>&1
rc=$?
log "alpha check: $(grep -E 'STAGE11_ALPHA_CHECK|comparison I' logs/stage11_alpha_check.out | tr '\n' ' ' | cut -c1-300)"
if (( rc != 0 )); then log "STAGE11_ALPHA_STOP (rc=$rc) -- report before launching"; exit 1; fi

# -- 2. pre-flight ------------------------------------------------------------------
PF=()
for p in 0.25 0.5 0.75 1.0; do
    for a in gpu baseline; do PF+=("11f_$(p_key $p)_s4126_preflight ${a}_ret $p"); done
done
run_queue "pre-flight" "${PF[@]}"
if $STOP; then log "STAGE11_STOPPED (pre-flight)"; exit 1; fi
if PYTHONPATH="$(cd ../../../.. && pwd):..:../gpu_tiers:../gpu_tiers_dataset:../sumo_close_loop:." \
        python3 preflight_stage11.py --seed 4126 --tag preflight > logs/stage11_preflight.out 2>&1; then
    log "pre-flight PASSED ($(tail -1 logs/stage11_preflight.out))"
else
    log "STAGE11_PREFLIGHT_FAILED -- see logs/stage11_preflight.out"; exit 1
fi

# -- 3. Phase 1 ---------------------------------------------------------------------
mapfile -t P1 < <(cells_for_seed 4126)
run_queue "Phase 1 (seed 4126)" "${P1[@]}"
if $STOP; then log "STAGE11_STOPPED (Phase 1)"; exit 1; fi
log "PHASE1_DONE"
if [[ -f stage11_phase1_report.py ]]; then
    python3 stage11_phase1_report.py > logs/stage11_phase1_report.out 2>&1 \
        && log "STAGE11_seed4126_report.md written" || log "stage11_phase1_report.py FAILED"
fi

# -- 4. wait for the user's go-file ---------------------------------------------------
log "waiting for logs/stage11_phase2.go (written by the user after review)"
until [[ -f logs/stage11_phase2.go ]]; do sleep 300; done
log "go-file found: $(head -1 logs/stage11_phase2.go)"

# -- 5. Phase 2 -----------------------------------------------------------------------
mapfile -t P2 < <(cells_for_seed 4127; cells_for_seed 4128)
run_queue "Phase 2 (seeds 4127, 4128)" "${P2[@]}"
if $STOP; then log "STAGE11_STOPPED (Phase 2)"; exit 1; fi
log "STAGE11_ALL_DONE"
