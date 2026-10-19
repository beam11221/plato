#!/usr/bin/env bash
# Launch Wen-environment runs in parallel, one process per config, with the
# Prompt C disk guard before each launch.
#
#   cd examples/async/fedasync
#   nohup ./wen_env/run_cells.sh wen_env/configs/main/*.toml > wen_env/logs/main.out 2>&1 &
#
# Disk guard: `df` on this pod reports the whole shared MooseFS (hundreds of TB
# free), not the 200 GB volume, so free space is QUOTA_GB - du(/workspace),
# taken as the smaller of that and df's own figure. Below MIN_FREE_GB, wait for
# the checkpoint cleaner; still below after WAIT_S, stop launching and exit 3.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-/workspace/plato/.venv/bin/python}
REPO=$(cd ../../.. && pwd)
QUOTA_GB=${QUOTA_GB:-200}
MIN_FREE_GB=${MIN_FREE_GB:-20}
WAIT_S=${WAIT_S:-900}
# The container's cgroup memory limit is 50 GB and one run (server + C = 3
# clients) holds about 7 GB RSS; 9 parallel runs were OOM-killed on 16 Oct.
# Count every live wen_fedasync server, from any batch, before launching.
MAX_PARALLEL=${MAX_PARALLEL:-5}
mkdir -p wen_env/logs

free_gb() {
    local used_b df_free_b quota_free_b
    used_b=$(du -sb /workspace 2>/dev/null | tail -1 | cut -f1)
    quota_free_b=$(( QUOTA_GB * 1024 * 1024 * 1024 - used_b ))
    df_free_b=$(( $(df -B1 --output=avail /workspace | tail -1) ))
    (( df_free_b < quota_free_b )) && quota_free_b=$df_free_b
    echo $(( quota_free_b / 1024 / 1024 / 1024 ))
}

guard() {
    local waited=0 free
    free=$(free_gb)
    while (( free < MIN_FREE_GB )); do
        if (( waited >= WAIT_S )); then
            echo "$(date -Is) STOP: free ${free} GB < ${MIN_FREE_GB} GB for ${WAIT_S} s"
            return 1
        fi
        echo "$(date -Is) free ${free} GB < ${MIN_FREE_GB} GB; waiting for the cleaner"
        sleep 60; waited=$(( waited + 60 )); free=$(free_gb)
    done
    echo "$(date -Is) disk guard ok: ${free} GB free"
}

live_servers() {
    # Servers carry "-c <cfg>" on their command line; spawned clients do not.
    pgrep -f "wen_fedasync.py -c" | wc -l
}

wait_for_slot() {
    while (( $(live_servers) >= MAX_PARALLEL )); do sleep 30; done
}

pids=()
for cfg in "$@"; do
    name=$(basename "$cfg" .toml)
    wait_for_slot
    guard || { status=3; break; }
    PYTHONPATH="$REPO" "$PY" wen_env/wen_fedasync.py -c "$cfg" > "wen_env/logs/$name.log" 2>&1 &
    pids+=($!)
    echo "$(date -Is) started $name (pid $!)"
done
status=${status:-0}
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
echo "$(date -Is) all done, status $status"
exit $status
