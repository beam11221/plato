#!/bin/bash
# Watch the Stage 3 sweep and repair the two things that are safe to repair.
#
#   ./watchdog.sh [--interval-min N] [--stall-passes N] [--quota-gb N] [--once]
#                 [--names-file FILE] [--no-janitor] [--log-tag TAG]
#
# --names-file, --no-janitor and --log-tag exist for a two-machine split that
# shares ONE mounted filesystem (both hosts see the same configs/, runs/ and
# logs/ at the same path in real time -- no git sync needed for code or
# results). See the comment above live_runs() for why --names-file is not
# optional in that setup, and check_janitor() for why only one host's
# watchdog should run without --no-janitor.
#
# Ported from ../separability/watchdog.sh, which exists because of a prior
# incident there: "the cleaner died, the volume filled, and the trainer hung
# with no error." On 2026-09-14 this folder had a related but distinct one --
# a bulk `du`/`rm` over runs/gpu_tiers_dataset (cleaning up dead-weight
# checkpoints from unmanaged verification builds) appears to have stalled every
# live run's blocking checkpoint I/O at once; all five processes went idle in
# `ep_poll` and their logs stopped growing for 20+ minutes with no error raised
# anywhere. `check_progress` below is exactly the mechanism that would have
# caught that within one interval instead of the user noticing "disk 100%"
# first -- log-growth stall detection does not care WHY a process stopped
# advancing, only THAT it did.
#
# WHAT IT WATCHES
#   * every live run (a config with no .done marker): is its round count rising?
#   * the janitor
#   * each live run's checkpoint cleaner
#   * disk against the allocation
#   * Traceback / CUDA OOM / OOM-kill in the live logs
#
# WHAT IT REPAIRS, and nothing else
#   * a dead janitor -> restarted. It only empties the checkpoints of runs that
#     have already finished.
#   * a dead per-run cleaner -> restarted for that run.
#
# WHAT IT WILL NOT DO
#   Restart a trainer, delete a .done marker, touch results/, *.log,
#   config_used.toml or models/, or change any config. A stalled or crashed run
#   is REPORTED, not restarted: deciding whether a half-finished cell is still
#   comparable is a judgement call about what the experiment measures, and that
#   belongs to a person -- this is the same discipline
#   stage3_evidence/step0_verification.md and everything before it in this
#   project has followed for every other judgement call.
#
# Everything goes to logs/watchdog.log. Run it in its own screen:
#   screen -dmS gpuwatch bash -c '/path/to/gpu_tiers_dataset/watchdog.sh'

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FEDASYNC_DIR="$(dirname "$HERE")"
CLEANER="$FEDASYNC_DIR/gpu_tiers/clean_checkpoints.sh"
RUNS_DIR="$FEDASYNC_DIR/runs/gpu_tiers_dataset"
LOG_DIR="$HERE/logs"
STATE="$LOG_DIR/.watchdog_rounds"
PYTHON="${PYTHON:-python3}"

INTERVAL_MIN=15
STALL_PASSES=2          # 2 x 15 min = 30 min of no progress before it shouts --
                        # tighter than separability's 45 min, deliberately: this
                        # folder has already lost 20+ min to an undetected stall
                        # once, and Stage 3's runs are individually more
                        # expensive (5-9 concurrent at a time) to leave stuck.
QUOTA_GB=200             # confirmed by the user 2026-09-14; see janitor.sh's own
                        # note that this quota is POD-WIDE, not scoped to
                        # $FEDASYNC_DIR/runs, which is the only thing check_disk()
                        # below actually measures.
KEEP_ROUNDS=50
CLEAN_INTERVAL=3
MAX_DIR_MB=12000
ONCE=false
NAMES_FILE=""
NO_JANITOR=false
LOG_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval-min) INTERVAL_MIN="$2"; shift 2 ;;
        --stall-passes) STALL_PASSES="$2"; shift 2 ;;
        --quota-gb)     QUOTA_GB="$2"; shift 2 ;;
        --once)         ONCE=true; shift ;;
        --names-file)   NAMES_FILE="$2"; shift 2 ;;
        --no-janitor)   NO_JANITOR=true; shift ;;
        --log-tag)      LOG_TAG="$2"; shift 2 ;;
        *) echo "unknown argument '$1'" >&2; exit 2 ;;
    esac
done

# --names-file exists for a multi-host, ONE shared filesystem setup (two
# machines with the same runs/gpu_tiers_dataset/, configs/ and logs/ mounted
# at the same path). Without it, live_runs() globs every config in configs/,
# including the OTHER host's runs -- and pgrep in check_cleaner()/check_progress()
# only sees processes on ITS OWN host, so a watchdog on host A would see host
# B's runs, conclude their cleaners are dead (it cannot see B's process table),
# and start a second cleaner against the same checkpoints/ directory B is
# already cleaning. That is the same class of concurrent-write hazard that
# stalled every live run once already on 2026-09-14 (see janitor.sh's own
# note) -- just triggered by two hosts instead of one bulk command. Pass a
# file (one run name per line, blank/#-comment lines ignored, same format as
# run_sweep.sh --from-file) listing only the runs THIS host is responsible
# for. Every other host must do the same with its own disjoint name list.
mkdir -p "$LOG_DIR"
declare -A OWNED_NAMES=()
if [[ -n "$NAMES_FILE" ]]; then
    if [[ ! -f "$NAMES_FILE" ]]; then
        echo "--names-file '$NAMES_FILE' does not exist" >&2
        exit 2
    fi
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | xargs)"
        [[ -n "$line" ]] && OWNED_NAMES["$line"]=1
    done < "$NAMES_FILE"
fi

LOG_SUFFIX=""
[[ -n "$LOG_TAG" ]] && LOG_SUFFIX="_${LOG_TAG}"
log() { echo "[$(date '+%F %T')] [watchdog] $*" | tee -a "$LOG_DIR/watchdog${LOG_SUFFIX}.log"; }

declare -A STALLED=()

is_throwaway() {
    # make_configs.THROWAWAY_TAGS is the one registry (see preflight.py /
    # collect.py, which both used to keep a second hand-maintained copy of
    # this list that silently drifted). A bare "*_probe*"-style glob here
    # would have exactly that same failure mode.
    "$PYTHON" -c "
import sys
sys.path.insert(0, '$HERE')
import make_configs
sys.exit(0 if any(sys.argv[1].endswith(t) for t in make_configs.THROWAWAY_TAGS) else 1)
" "$1"
}

live_runs() {
    shopt -s nullglob
    for config in "$HERE"/configs/*.toml; do
        local name
        name="$(basename "$config" .toml)"
        is_throwaway "$name" && continue
        # See the --names-file note above: with no file, every host-visible
        # config counts (the original, single-host behaviour); with a file,
        # only names in it do -- a run outside this host's assignment is
        # someone else's to watch, not silently skipped by accident.
        if [[ -n "$NAMES_FILE" ]] && [[ -z "${OWNED_NAMES[$name]:-}" ]]; then
            continue
        fi
        [[ -f "$LOG_DIR/${name}.done" ]] && continue
        [[ -f "$LOG_DIR/${name}.log" ]] || continue   # not started yet
        # A run whose trainer is gone but which has no .done marker either died
        # or was interrupted; it is still reported below, once.
        echo "$name"
    done
    shopt -u nullglob
}

check_progress() {
    local name="$1"
    local rounds previous
    rounds=$(grep -c 'Starting round' "$LOG_DIR/${name}.log" 2>/dev/null)
    rounds="${rounds:-0}"
    previous=$(grep "^${name} " "$STATE" 2>/dev/null | awk '{print $2}')
    previous="${previous:-0}"

    if pgrep -f -- "fedasync_seeded.py .*-c .*${name}\.toml" > /dev/null; then
        if (( rounds > previous )); then
            STALLED["$name"]=0
        else
            STALLED["$name"]=$(( ${STALLED[$name]:-0} + 1 ))
            if (( STALLED[$name] >= STALL_PASSES )); then
                log "STALL: $name is alive but stuck at round $rounds for "
                log "       $(( STALLED[$name] * INTERVAL_MIN )) min — a log that stops growing while the"
                log "       process lives is the failure mode this exists to catch. NOT restarting it."
            fi
        fi
    else
        log "DOWN: $name has no process and no .done marker (last round $rounds)."
        log "      Re-running run_sweep.sh with this name resumes it from PLATO's checkpoint,"
        log "      if its checkpoints/ has not since been pruned by the janitor."
    fi

    grep -v "^${name} " "$STATE" 2>/dev/null > "$STATE.tmp"
    echo "$name $rounds" >> "$STATE.tmp"
    mv "$STATE.tmp" "$STATE"
}

check_errors() {
    local name="$1"
    local hits
    hits=$(grep -c -E 'Traceback|CUDA out of memory|Killed process|torch\.cuda\.OutOfMemoryError' \
        "$LOG_DIR/${name}.log" 2>/dev/null)
    hits="${hits:-0}"
    (( hits > 0 )) && log "ERROR: $name has $hits error line(s) — grep Traceback in $LOG_DIR/${name}.log"
    return 0
}

check_cleaner() {
    local name="$1"
    local checkpoints="$RUNS_DIR/${name}/checkpoints"
    [[ -d "$checkpoints" ]] || return 0
    if ! pgrep -f "clean_checkpoints.sh $checkpoints" > /dev/null; then
        log "REPAIR: cleaner for $name was not running — restarting it"
        "$CLEANER" "$checkpoints" "$KEEP_ROUNDS" "$CLEAN_INTERVAL" "$MAX_DIR_MB" \
            >> "$LOG_DIR/${name}.cleaner.log" 2>&1 &
    fi
}

check_janitor() {
    $NO_JANITOR && return 0
    if ! pgrep -f "gpu_tiers_dataset/janitor.sh" > /dev/null; then
        log "REPAIR: janitor was not running — restarting it"
        "$HERE/janitor.sh" --interval-min 5 --quota-gb "$QUOTA_GB" \
            >> "$LOG_DIR/janitor.log" 2>&1 &
    fi
}

check_disk() {
    local used
    used=$(du -sBG "$FEDASYNC_DIR/runs" 2>/dev/null | cut -f1 | tr -dc '0-9')
    used="${used:-0}"
    if (( used > QUOTA_GB * 90 / 100 )); then
        log "DISK: ${used} GB of ${QUOTA_GB} GB — over 90%, running the janitor now"
        "$HERE/janitor.sh" --once --quota-gb "$QUOTA_GB" >> "$LOG_DIR/janitor.log" 2>&1
    elif (( used > QUOTA_GB * 70 / 100 )); then
        log "DISK: ${used} GB of ${QUOTA_GB} GB — over 70%"
    fi
}

arm_of() {
    "$PYTHON" -c "
import sys
sys.path.insert(0, '$HERE')
import make_configs
from pathlib import Path
print(make_configs.arm_of(Path(sys.argv[1] + '.toml')))
" "$1" 2>/dev/null || echo "?"
}

log "started (every ${INTERVAL_MIN} min, stall after ${STALL_PASSES} passes, quota ${QUOTA_GB} GB)"
while true; do
    check_janitor
    check_disk
    mapfile -t names < <(live_runs)
    heartbeat=""
    for name in "${names[@]:-}"; do
        [[ -z "$name" ]] && continue
        check_progress "$name"
        check_errors "$name"
        check_cleaner "$name"
        heartbeat+=" $(arm_of "$name"):$(grep -c 'Starting round' "$LOG_DIR/${name}.log" 2>/dev/null)"
    done
    # One line per pass even when nothing is wrong. Without it an empty log is
    # ambiguous between "all well" and "the watchdog died an hour ago".
    log "ok:${heartbeat:- no live runs} | done $(ls "$LOG_DIR"/*.done 2>/dev/null | wc -l) | disk $(du -sBG "$FEDASYNC_DIR/runs" 2>/dev/null | cut -f1)"
    $ONCE && break
    sleep $(( INTERVAL_MIN * 60 ))
done
