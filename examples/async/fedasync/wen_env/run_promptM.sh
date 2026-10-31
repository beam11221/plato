#!/usr/bin/env bash
# Prompt M driver: Part A full runs, then Part B (tier-mix sweep, seed 4126).
# Survives the Claude session: start it in tmux.
#
#   cd examples/async/fedasync
#   tmux new-session -d -s pm_driver "./wen_env/run_promptM.sh > wen_env/logs/promptM/driver.out 2>&1"
#
# Order (Prompt M, as amended by the author on 31 Oct: the Part A runs replace
# the old seed-4126 runs and are compared with them in overview only):
#   1. Part A: gpu, polaris, random, wen_fresh (M0, full runs), in parallel.
#   2. Part B starts once Part A's gpu, polaris and random have finished (their
#      overview vs the originals is written to logs/promptM/partA_compare.txt);
#      Part A's wen_fresh keeps running. Cheaper arms first, wen_fresh last.
# run_cells.sh holds the disk guard (stop launching below 20 GB for 15 min) and
# the global cap on live servers (MAX_PARALLEL, counted across both batches).
# The watchdog (tools/watchdog.sh) stops everything on a clock mismatch.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-/workspace/plato/.venv/bin/python}
export MAX_PARALLEL=${MAX_PARALLEL:-5}
LOG=wen_env/logs/promptM
CFG=wen_env/configs/promptM
mkdir -p "$LOG"

echo "$(date -Is) Part A launch"
./wen_env/run_cells.sh $CFG/S1_gpu_s4126_pm.toml $CFG/S1_polaris_s4126_pm.toml \
    $CFG/S1_random_s4126_pm.toml $CFG/S1_wen_fresh_s4126_pm.toml > "$LOG/partA.out" 2>&1 &
partA=$!

cheap_done() {   # Part A's gpu / polaris / random servers have all exited
    sleep 120
    for n in gpu polaris random; do
        pgrep -f "wen_fedasync.py -c $CFG/S1_${n}_s4126_pm.toml" > /dev/null && return 1
        [ -f "wen_env/logs/S1_${n}_s4126_pm.log" ] || return 1
    done
    return 0
}
until cheap_done; do sleep 300; done
echo "$(date -Is) Part A gpu / polaris / random finished"
PYTHONPATH=/workspace/plato "$PY" wen_env/tools/promptm_compare.py \
    runs/wen_env/S1_gpu_s4126_pm runs/wen_env/_phaseA/S1_gpu_s4126 \
    runs/wen_env/S1_polaris_s4126_pm runs/wen_env/S1_polaris_s4126 \
    runs/wen_env/S1_random_s4126_pm runs/wen_env/_phaseA/S1_random_s4126 \
    > "$LOG/partA_compare.txt" 2>&1
if grep -q "ClockMismatchError" wen_env/logs/S1_*_s4126_pm.log; then
    echo "$(date -Is) STOP: clock mismatch in Part A"; exit 2
fi

echo "$(date -Is) Part B launch"
cfgs=()
for arm in random gpu polaris wen_fresh; do
    for mix in M1 M2 M3 T1 T2 T3; do cfgs+=("$CFG/S1_${mix}_${arm}_s4126.toml"); done
done
./wen_env/run_cells.sh "${cfgs[@]}" > "$LOG/partB.out" 2>&1
statusB=$?
wait $partA
echo "$(date -Is) all done: Part A status $?, Part B status $statusB"
