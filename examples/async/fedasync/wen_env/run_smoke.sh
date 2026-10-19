#!/usr/bin/env bash
# Prompt B, Step 9: run the smoke cells in parallel, plus two short repeats of
# the random arm for the determinism test (Step 8, test 6).
#
#   cd examples/async/fedasync && ./wen_env/run_smoke.sh
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-/workspace/plato/.venv/bin/python}
REPO=$(cd ../../.. && pwd)
mkdir -p wen_env/logs

"$PY" wen_env/wen_make_configs.py --smoke > /dev/null
"$PY" wen_env/wen_make_configs.py --cells S1:random --rounds 30 --tag _det1 --port-base 20180 > /dev/null
"$PY" wen_env/wen_make_configs.py --cells S1:random --rounds 30 --tag _det2 --port-base 20181 > /dev/null

configs=(wen_env/configs/smoke/*_smoke.toml
         wen_env/configs/runs/S1_random_s4126_det1.toml
         wen_env/configs/runs/S1_random_s4126_det2.toml)
pids=()
for cfg in "${configs[@]}"; do
    name=$(basename "$cfg" .toml)
    PYTHONPATH="$REPO" "$PY" wen_env/wen_fedasync.py -c "$cfg" > "wen_env/logs/$name.log" 2>&1 &
    pids+=($!)
    echo "started $name (pid $!)"
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
echo "all done, status $status"
exit $status
