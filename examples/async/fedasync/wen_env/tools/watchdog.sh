#!/usr/bin/env bash
# Run watchdog (Prompt K Step 0.6). Reports only; never restarts a run.
#
#   cd examples/async/fedasync
#   nohup wen_env/tools/watchdog.sh > wen_env/logs/watchdog.out 2>&1 &
#
# Every CHECK_S seconds, for each live wen_fedasync server:
#   * no new merge (merge_log.csv unchanged) for STALL_S -> one alert per stall,
#     appended to the run's log and to wen_env/logs/watchdog_alerts.log;
#   * ClockMismatchError in the run's log -> alert, then stop every wen run
#     (Prompt K stop rule: online != replayed clock -> stop all runs).
set -u
cd "$(dirname "$0")/../.."
STALL_S=${STALL_S:-1800}
CHECK_S=${CHECK_S:-60}
ALERTS=wen_env/logs/watchdog_alerts.log
declare -A alerted

alert() {   # $1 run log, $2 message
    local line="$(date -Is) WATCHDOG $2"
    echo "$line" | tee -a "$ALERTS" >> "$1"
}

while true; do
    for pid in $(pgrep -f "wen_fedasync.py -c"); do
        cfg=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | sed -n 's/.* -c \([^ ]*\).*/\1/p')
        [ -n "$cfg" ] || continue
        name=$(basename "$cfg" .toml)
        log=wen_env/logs/$name.log
        base=$(sed -n 's/^base_path = "\(.*\)"/\1/p' "$cfg")
        merges=$base/merge_log.csv
        if grep -q ClockMismatchError "$log" 2>/dev/null; then
            alert "$log" "CLOCK MISMATCH in $name -- stopping all wen runs"
            pkill -f "run_cells.sh"
            pkill -f "wen_fedasync.py -c"
            exit 2
        fi
        [ -f "$merges" ] || continue
        m=$(stat -c %Y "$merges"); now=$(date +%s)
        if (( now - m > STALL_S )) && [ "${alerted[$name]:-}" != "$m" ]; then
            alert "$log" "STALL: $name has no new merge for $(( (now - m) / 60 )) min (pid $pid); not restarted"
            alerted[$name]=$m
        fi
    done
    sleep "$CHECK_S"
done
