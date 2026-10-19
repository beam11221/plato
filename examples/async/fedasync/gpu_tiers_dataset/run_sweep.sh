#!/bin/bash
# Run gpu_tiers_dataset Stage 3 configs, each with its own checkpoint cleaner.
#
# Usage:
#   ./run_sweep.sh <name> [name ...] [options]
#   ./run_sweep.sh --from-file batch1.txt [options]
#
#   name                 a config stem under configs/, e.g. gpu_a0p8_N20_C4_s4126
#                        (no .toml, no directory)
#   --from-file FILE     one name per line (blank lines and #-comments ignored),
#                        instead of / in addition to positional names
#   --max-parallel N     runs in flight at once             (default: 4)
#   --keep-rounds N      checkpoints to keep                (default: 50)
#   --clean-interval N   cleaner period, minutes            (default: 3)
#   --max-dir-mb N       hard ceiling per checkpoint dir    (default: 12000)
#   --no-clean           do not start the cleaner (you will fill the disk)
#   --dry-run            print what would run, touch nothing
#
# No default name list: `configs/` accumulates Stage 1/2/Step-0 throwaway
# builds (det1-6, probe, smoke) alongside the real sweep, and "run everything
# in the directory" would eventually re-launch all of them. Every invocation
# names its runs explicitly, or points at a file that does.
#
# WHY RUNS OF DIFFERENT CELLS MAY SHARE ONE WINDOW HERE
# ------------------------------------------------------
# `../separability/run.sh` runs every arm of every alpha together because
# PLATO's `processing_time` (a real host measurement) rides in the simulated
# clock, so two runs sharing a machine window share that residue and stay
# mutually comparable -- but only runs sharing a window with EACH OTHER.
#
# Stage 3's Step 0 (`stage3_evidence/step0_verification.md`) removes the
# mechanism itself: with `[server] exclude_processing_time_from_clock = true`
# (the default `make_configs.py` now writes), `processing_time` never enters
# `finish_time`, so two runs of one config produce byte-identical dispatch
# sequences, merge orders and reconstructed clocks -- verified, not assumed.
# Contention no longer contaminates a comparison, so cells may run in ANY
# arrangement, per the brief: "If Step 0 succeeded, this constraint is gone."
#
# A config rendered WITHOUT the patch (`exclude_processing_time_from_clock =
# false`) is still run by this script if asked -- but the one-cell-at-a-time
# discipline for THAT config is then on the caller, not enforced here.
#
# Safe to re-run: a run that already finished is skipped via its .done marker.
# Delete the marker (or the run directory) to force a redo.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FEDASYNC_DIR="$(dirname "$HERE")"
REPO_ROOT="$(cd "$FEDASYNC_DIR/../../.." && pwd)"
CLEANER="$FEDASYNC_DIR/gpu_tiers/clean_checkpoints.sh"
RUNS_DIR="$FEDASYNC_DIR/runs/gpu_tiers_dataset"
LOG_DIR="$HERE/logs"

PYTHON="${PYTHON:-python3}"
MAX_PARALLEL=4
KEEP_ROUNDS=50
CLEAN_INTERVAL=3
MAX_DIR_MB=12000
DO_CLEAN=true
DRY_RUN=false
FROM_FILE=""
NAMES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-file)       FROM_FILE="$2"; shift 2 ;;
        --max-parallel)    MAX_PARALLEL="$2"; shift 2 ;;
        --keep-rounds)     KEEP_ROUNDS="$2"; shift 2 ;;
        --clean-interval)  CLEAN_INTERVAL="$2"; shift 2 ;;
        --max-dir-mb)      MAX_DIR_MB="$2"; shift 2 ;;
        --no-clean)        DO_CLEAN=false; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --*) echo "unknown option '$1'" >&2; exit 2 ;;
        *)   NAMES+=("$1"); shift ;;
    esac
done

if [[ -n "$FROM_FILE" ]]; then
    if [[ ! -f "$FROM_FILE" ]]; then
        echo "--from-file '$FROM_FILE' does not exist" >&2
        exit 2
    fi
    while IFS= read -r line; do
        line="${line%%#*}"                  # strip trailing comments
        line="$(echo "$line" | xargs)"       # trim whitespace
        [[ -n "$line" ]] && NAMES+=("$line")
    done < "$FROM_FILE"
fi

if [[ ${#NAMES[@]} -eq 0 ]]; then
    echo "no run names given -- pass them positionally or via --from-file" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%F %T')] [run_sweep] $*"; }

arm_of() {
    "$PYTHON" -c "
import sys
sys.path.insert(0, '$HERE')
import make_configs
print(make_configs.arm_of(__import__('pathlib').Path(sys.argv[1] + '.toml')))
" "$1"
}

CLEANER_PIDS=()
cleanup() {
    for pid in "${CLEANER_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null
    done
    pkill -P $$ 2>/dev/null
}
trap 'log "interrupted — stopping runs and cleaners"; cleanup; exit 130' INT TERM

# ---------------------------------------------------------------------------
# Preflight once, on exactly the configs about to run -- not the whole
# directory, which also holds throwaway builds this invocation was not asked
# to launch.
# ---------------------------------------------------------------------------
PREFLIGHT_CONFIGS=()
for name in "${NAMES[@]}"; do
    PREFLIGHT_CONFIGS+=("$HERE/configs/${name}.toml")
done
log "preflight (${#NAMES[@]} config(s))"
if ! "$PYTHON" "$HERE/preflight.py" --skip-env --configs "${PREFLIGHT_CONFIGS[@]}"; then
    log "preflight FAILED — refusing to run"
    exit 1
fi

log "${#NAMES[@]} run(s): ${NAMES[*]}"
log "at most $MAX_PARALLEL in flight, cleaner $( $DO_CLEAN && echo on || echo OFF )"

start_run() {
    local name="$1"
    local config="$HERE/configs/${name}.toml"
    local run_dir="$RUNS_DIR/${name}"
    local log_file="$LOG_DIR/${name}.log"
    local done_marker="$LOG_DIR/${name}.done"

    if [[ ! -f "$config" ]]; then
        log "$name: config is missing — run make_configs.py first"
        return 1
    fi
    if [[ -f "$done_marker" ]]; then
        log "$name: already finished (delete $done_marker to redo) — skipping"
        return 0
    fi

    local arm
    arm="$(arm_of "$name")" || { log "$name: could not resolve its arm"; return 1; }

    if $DRY_RUN; then
        log "$name: would run --arm $arm with $config -> $log_file"
        return 0
    fi

    if $DO_CLEAN; then
        mkdir -p "$run_dir/checkpoints"
        "$CLEANER" "$run_dir/checkpoints" "$KEEP_ROUNDS" "$CLEAN_INTERVAL" \
            "$MAX_DIR_MB" >> "$LOG_DIR/${name}.cleaner.log" 2>&1 &
        CLEANER_PIDS+=("$!")
    fi

    (
        cd "$FEDASYNC_DIR" || exit 1
        # gpu_tiers first: fedasync_seeded.py imports gpu_tier_sleep bare, and
        # the repo root must be on PYTHONPATH -- PLATO is imported from the
        # source tree, not installed.
        PYTHONPATH="$REPO_ROOT:$FEDASYNC_DIR/gpu_tiers:$FEDASYNC_DIR" \
            "$PYTHON" "$HERE/fedasync_seeded.py" --arm "$arm" -c "$config" \
            > "$log_file" 2>&1
        status=$?
        if [[ $status -eq 0 ]]; then
            date '+%F %T' > "$done_marker"
        fi
        exit $status
    ) &
    log "$name: started (pid $!, arm $arm) -> $log_file"
}

# ---------------------------------------------------------------------------
# Launch, respecting the cap, then wait
# ---------------------------------------------------------------------------
failures=0
for name in "${NAMES[@]}"; do
    while [[ $MAX_PARALLEL -gt 0 ]] && \
          [[ $(jobs -rp | wc -l) -ge $(( MAX_PARALLEL + ${#CLEANER_PIDS[@]} )) ]]; do
        wait -n 2>/dev/null || failures=$((failures + 1))
    done
    start_run "$name" || failures=$((failures + 1))
    # PLATO binds its port and unpacks CIFAR-10 at startup; a stagger keeps
    # several simultaneous first-round downloads from fighting over the data dir.
    $DRY_RUN || sleep 15
done

while [[ $(jobs -rp | wc -l) -gt ${#CLEANER_PIDS[@]} ]]; do
    wait -n 2>/dev/null || failures=$((failures + 1))
done
cleanup

log "done. failures: $failures"
exit $(( failures > 0 ? 1 : 0 ))
