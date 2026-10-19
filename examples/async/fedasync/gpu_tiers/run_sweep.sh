#!/bin/bash
# Run the heterogeneity sweep: every gap of every arm you name, one at a time.
#
# Each run gets its own checkpoint cleaner, started with the run and killed when
# it ends, so disk usage stays flat instead of growing to ~440 GB per run.
#
# Usage:
#   ./run_sweep.sh <arm> [arm ...] [options]
#
#   arm                gpu | cpu | observed | baseline  (run in the order given)
#   --gaps "01 05 10"  gaps to run       (default: 01 10 20 30 40)
#   --mixes "m70_20_10 m34_33_33"  fleet compositions to run (default: none, i.e.
#                      the historical 50/35/15 fleet, whose names carry no label)
#   --anchor N         anchor seconds    (default: 20 — must match configs/)
#   --keep-rounds N    checkpoints kept  (default: 100)
#   --clean-interval N cleaner period, minutes (default: 5 — see "On disk")
#   --max-dir-mb N     hard ceiling per checkpoint dir, MB (default: 20000; 0 off)
#   --mem-per-proc N   MB per client process for the guard (default: 1600)
#   --probe-mb N       MB written to test the disk QUOTA before each run
#                      (default: 4096; 0 disables — see "disk guard")
#   --no-clean         do not start the cleaner (watch your disk)
#   --skip-mem-check   run even if the memory guard objects
#   --dry-run          print what would run, touch nothing
#
# ONE ARM AT A TIME — do not run two of these in parallel
# ------------------------------------------------------
# Measured on this box: every client process holds ~1.6 GB RSS, so one run
# (1 server + max_concurrency clients) needs ~17.6 GB.  The container is capped
# at ~29 GB by its cgroup, NOT by the 251 GB that `free` reports for the host.
# Two arms in parallel therefore need ~35 GB and the cgroup OOM killer takes the
# trainers out with SIGKILL (exit 137) roughly 70 s in.  That is what happened
# on the first attempt: 7 runs killed, 0 completed.
#
# So name the arms here and let them queue:
#   screen -S sweep -dm ./run_sweep.sh gpu cpu baseline
#
# The guard below re-checks the cgroup limit before every run and refuses rather
# than letting the OOM killer decide.  GPU memory was never the constraint —
# the whole sweep sat at 2.9 GB of 16 GB.
#
# On disk
# -------
# A checkpoint is ~34 MB and, with sleep_simulation = true, rounds arrive at
# roughly 30/min.  Peak usage is set by the cleaner period, not --keep-rounds:
#
#     peak ~ (KEEP_ROUNDS + 30 * INTERVAL_MINUTES) * 34 MB
#
# 5 minutes keeps one run near 8 GB.
#
# Safe to re-run: a run that already finished is skipped via its .done marker.
# Delete the marker (or the run directory) to force a redo.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$HERE")"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../../.." && pwd)"

# Overridable so a multi-seed sweep (configs_ms/, written by
# `make_configs.py --seeds ...`) can be driven by the same runner.
CONFIG_DIR="${CONFIG_DIR:-$HERE/configs}"
# Overridable so a sweep that lives in its own directory (surgical/) keeps its
# logs and .done markers beside its own configs instead of in gpu_tiers/logs.
LOG_DIR="${LOG_DIR:-$HERE/logs}"
CLEANER="$HERE/clean_checkpoints.sh"

# The project venv's torch install is broken (torch/lib missing), so the default
# is the system interpreter.  Override with PYTHON=... once the venv is fixed.
PYTHON="${PYTHON:-python3}"

# Leading positional arguments are arm names; everything from the first --flag on
# is options.
ARMS=()
while [[ $# -gt 0 && "$1" != --* ]]; do
    ARMS+=("$1")
    shift
done

# Five gaps, matching make_configs.py's default — 01/10/20/30/40 spans the whole
# effective range (1.0 -> 8.8).  Nine gaps x 3 arms x ~3 h does not fit in a day.
GAPS="01 10 20 30 40"
# Empty = the single-seed layout, whose config and run names carry no seed
# suffix.  Set --seeds to run the multi-seed set; the suffix is part of the run
# name, so .done markers, logs and run directories stay separate per seed.
SEEDS=""
# Empty = the historical 50/35/15 fleet, whose names carry no mix label.  Set
# --mixes to run the fleet-composition sweep (make_mix_sweep.sh); like the seed
# suffix, the label is part of the run name.
MIXES=""
ANCHOR=20
KEEP_ROUNDS=100
CLEAN_INTERVAL=5
MAX_DIR_MB=20000
MEM_PER_PROC=1600
PROBE_MB=4096
DO_CLEAN=true
DRY_RUN=false
SKIP_MEM_CHECK=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gaps)           GAPS="$2"; shift 2 ;;
        --seeds)          SEEDS="$2"; shift 2 ;;
        --mixes)          MIXES="$2"; shift 2 ;;
        --config-dir)     CONFIG_DIR="$2"; shift 2 ;;
        --anchor)         ANCHOR="$2"; shift 2 ;;
        --keep-rounds)    KEEP_ROUNDS="$2"; shift 2 ;;
        --clean-interval) CLEAN_INTERVAL="$2"; shift 2 ;;
        --max-dir-mb)     MAX_DIR_MB="$2"; shift 2 ;;
        --mem-per-proc)   MEM_PER_PROC="$2"; shift 2 ;;
        --probe-mb)       PROBE_MB="$2"; shift 2 ;;
        --no-clean)       DO_CLEAN=false; shift ;;
        --skip-mem-check) SKIP_MEM_CHECK=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        *) echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
    esac
done

if [[ ${#ARMS[@]} -eq 0 ]]; then
    echo "Usage: $0 <gpu|cpu|observed|baseline> [arm ...] [options]" >&2
    exit 2
fi

entry_for() {
    case "$1" in
        gpu|cpu|observed) echo "fedasync_gpu_tier_polaris_vi.py" ;;
        baseline)         echo "fedasync_gpu_tier.py" ;;
        *) return 1 ;;
    esac
}

for arm in "${ARMS[@]}"; do
    if ! entry_for "$arm" >/dev/null; then
        echo "ERROR: unknown arm '$arm' (expected gpu, cpu, observed or baseline)" >&2
        exit 2
    fi
done

ARM="${ARMS[0]}"   # updated per arm; read by log()
log() { echo "[$(date '+%F %T')] [$ARM] $*"; }

# ── memory guard ─────────────────────────────────────────────────────────────
# `free` reports the host, which on a container is meaningless — read the cgroup
# limit that actually applies.  Returns MB, or 0 when there is no limit.
cgroup_limit_mb() {
    local raw=""
    if [[ -r /sys/fs/cgroup/memory.max ]]; then
        raw=$(cat /sys/fs/cgroup/memory.max)
        [[ "$raw" == "max" ]] && { echo 0; return; }
    elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
        raw=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    else
        echo 0; return
    fi
    # A "limit" larger than physical RAM means unlimited in practice.
    local phys_mb
    phys_mb=$(( $(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE) / 1024 / 1024 ))
    local mb=$(( raw / 1024 / 1024 ))
    (( mb >= phys_mb )) && { echo 0; return; }
    echo "$mb"
}

# Usage, minus the page cache the kernel will drop the moment a run needs the
# memory.  cgroup v1 counts cache in usage_in_bytes, and analysing the sweep's
# logs leaves ~10 GB of it behind, so charging it against the next run refuses
# runs that would have fitted with room to spare — which is how a queued sweep
# silently stops after its first few runs.  Only inactive_file is discounted;
# active file cache stays counted, which errs toward refusing.
cgroup_usage_mb() {
    local usage=0 reclaimable=0
    if [[ -r /sys/fs/cgroup/memory.current ]]; then
        usage=$(cat /sys/fs/cgroup/memory.current)
        reclaimable=$(awk '/^inactive_file /{print $2; exit}' /sys/fs/cgroup/memory.stat 2>/dev/null)
    elif [[ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
        usage=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)
        reclaimable=$(awk '/^total_inactive_file /{print $2; exit}' /sys/fs/cgroup/memory/memory.stat 2>/dev/null)
    else
        echo 0; return
    fi
    reclaimable="${reclaimable:-0}"
    (( usage > reclaimable )) && usage=$(( usage - reclaimable ))
    echo $(( usage / 1024 / 1024 ))
}

# Refuse a run whose estimated footprint does not fit, so the failure is a clear
# message here instead of SIGKILL 70 s later with nothing to show for it.
check_memory() {
    local cfg="$1" limit used concurrency need
    limit=$(cgroup_limit_mb)
    [[ "$limit" -eq 0 ]] && return 0

    concurrency=$(grep -E '^max_concurrency' "$cfg" | head -1 | grep -oE '[0-9]+')
    concurrency="${concurrency:-10}"
    need=$(( (concurrency + 1) * MEM_PER_PROC ))
    used=$(cgroup_usage_mb)

    if (( used + need > limit )); then
        log "MEMORY GUARD: need ~${need} MB ($((concurrency + 1)) processes x ${MEM_PER_PROC} MB)"
        log "              in use ${used} MB of ${limit} MB cgroup limit -> would not fit"
        if $SKIP_MEM_CHECK; then
            log "              --skip-mem-check given, going ahead anyway"
            return 0
        fi
        log "              refusing.  Free memory, lower max_concurrency in"
        log "              templates/ and rerun make_configs.py, or pass --skip-mem-check."
        return 1
    fi
    return 0
}

# ── disk guard ───────────────────────────────────────────────────────────────
# `df` is useless here: /workspace is a MooseFS share whose QUOTA is invisible to
# it — it cheerfully reports hundreds of terabytes free while a write returns
# ENOSPC/EDQUOT.  That has now cost this project three separate incidents:
# 2026-08-13 (cleaner died, 94 GB piled up, volume filled, trainer hung with no
# error in its log), 2026-08-14 and 2026-08-15 (two runs in parallel doubled the
# checkpoint rate and hit the quota; every process parked in ep_poll forever,
# and the only trace was an `OSError: [Errno 122] Disk quota exceeded` inside a
# logging handler, thousands of lines up in a client's stdout).
#
# The only reliable probe is to actually write.  A run needs roughly
# KEEP_ROUNDS * 34 MB plus whatever accumulates between cleaner passes, so the
# default probe is deliberately larger than one run's steady state.
check_disk() {
    # Separate statements on purpose: bash expands every word of a `local`
    # command before performing any of its assignments, so "$dir" inside the
    # same `local` is still unset — and under `set -u` that aborts the runner.
    local dir="$1"
    local mb="$PROBE_MB"
    local probe="$dir/.disk_probe.$$"
    [[ "$mb" -le 0 ]] && return 0
    mkdir -p "$dir" 2>/dev/null
    if ! dd if=/dev/zero of="$probe" bs=1M count="$mb" status=none 2>/dev/null; then
        rm -f "$probe"
        log "DISK GUARD: cannot write ${mb} MB under $dir"
        log "            the volume or its quota is full — free space before running."
        log "            (checkpoints of FINISHED runs are the usual culprit:"
        log "             ./clean_checkpoints.sh <run>/checkpoints 5 --once 20000)"
        return 1
    fi
    rm -f "$probe"
    return 0
}

# ── cleanup on exit ──────────────────────────────────────────────────────────
CLEANER_PID=""
RUN_PID=""

# PLATO forks one client subprocess per max_concurrency slot, and those are
# children of the trainer.  Killing only the trainer leaves them running and
# still holding GPU memory, so take the children down first.
kill_tree() {
    local pid="$1" sig="${2:-TERM}"
    [[ -z "$pid" ]] && return
    pkill -"$sig" -P "$pid" 2>/dev/null
    kill -"$sig" "$pid" 2>/dev/null
}

cleanup() {
    [[ -n "$CLEANER_PID" ]] && kill "$CLEANER_PID" 2>/dev/null
    if [[ -n "$RUN_PID" ]]; then
        kill_tree "$RUN_PID" TERM
        for _ in 1 2 3 4 5; do
            kill -0 "$RUN_PID" 2>/dev/null || break
            sleep 1
        done
        kill_tree "$RUN_PID" KILL
    fi
    wait 2>/dev/null
}
trap 'log "interrupted — stopping current run and cleaner"; cleanup; exit 130' INT TERM
trap cleanup EXIT

mkdir -p "$LOG_DIR"

mem_limit=$(cgroup_limit_mb)
echo "[$(date '+%F %T')] arms=[${ARMS[*]}] gaps=[$GAPS] anchor=${ANCHOR}s python=$PYTHON"
echo "[$(date '+%F %T')] keep_rounds=$KEEP_ROUNDS ceiling=${MAX_DIR_MB}MB cleaner=$($DO_CLEAN && echo "every ${CLEAN_INTERVAL}m, supervised" || echo off)"
if [[ "$mem_limit" -gt 0 ]]; then
    echo "[$(date '+%F %T')] cgroup memory limit ${mem_limit} MB, currently $(cgroup_usage_mb) MB in use"
else
    echo "[$(date '+%F %T')] no cgroup memory limit detected"
fi

# Expand $MIXES and $SEEDS into the filename suffixes the configs actually carry.
# Empty on both axes yields a single empty suffix, which is the original layout.
# The order is mix-then-seed because that is the order make_configs.py renders
# into the filename: <arm>_xa<A>_gap<G>[_m<mix>][_s<seed>].toml
#
# Fills the SUFFIXES array instead of echoing.  `for sfx in $(seed_suffixes)`
# word-splits the output, and the single-seed branch echoes an empty line, which
# splits to *zero* words — so every single-seed sweep started after multi-seed
# support landed did nothing at all and exited "0 done, 0 skipped, 0 failed".
# The sweep running at the time survived only because bash had already read the
# old body off disk.
run_suffixes() {
    local mix_parts=("") seed_parts=("")
    if [[ -n "$MIXES" ]]; then
        mix_parts=()
        for m in $MIXES; do mix_parts+=("_${m#_}"); done
    fi
    if [[ -n "$SEEDS" ]]; then
        seed_parts=()
        for s in $SEEDS; do seed_parts+=("_s${s}"); done
    fi
    SUFFIXES=()
    for m in "${mix_parts[@]}"; do
        for s in "${seed_parts[@]}"; do SUFFIXES+=("${m}${s}"); done
    done
}
run_suffixes

# The run directory has to come from the config, not be guessed: a multi-seed
# sweep writes into runs/gap_sweep_ms, and a cleaner pointed at the wrong
# directory would prune nothing while the right one filled the disk.
run_dir_for() {
    "$PYTHON" - "$1" <<'PYEOF'
import sys, tomllib, pathlib
cfg = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
print(cfg["general"]["base_path"])
PYEOF
}

# Fail before doing any work if a config is missing, rather than halfway through.
missing=0
for arm in "${ARMS[@]}"; do
    for gap in $GAPS; do
      for sfx in "${SUFFIXES[@]}"; do
        cfg="$CONFIG_DIR/${arm}_xa${ANCHOR}_gap${gap}${sfx}.toml"
        [[ -f "$cfg" ]] || { echo "ERROR: missing $cfg" >&2; missing=1; }
      done
    done
done
if [[ "$missing" -eq 1 ]]; then
    echo "Run 'python3 make_configs.py --anchor $ANCHOR' first." >&2
    exit 1
fi

started_at=$(date +%s)
completed=0
skipped=0
failed=0

for ARM in "${ARMS[@]}"; do
    ENTRY="$(entry_for "$ARM")"
    log "=== arm start: $ARM ($ENTRY) ==="

    for gap in $GAPS; do
      for sfx in "${SUFFIXES[@]}"; do
        name="${ARM}_xa${ANCHOR}_gap${gap}${sfx}"
        cfg="$CONFIG_DIR/${name}.toml"
        run_dir="$EXAMPLE_DIR/$(run_dir_for "$cfg")"
        log_file="$LOG_DIR/${name}.log"
        done_marker="$LOG_DIR/${name}.done"

        if [[ -f "$done_marker" ]]; then
            log "SKIP  $name (already finished — delete $done_marker to redo)"
            (( skipped++ ))
            continue
        fi

        if $DRY_RUN; then
            log "DRY   PYTHONPATH=$HERE:$REPO_ROOT $PYTHON $ENTRY -c $cfg"
            $DO_CLEAN && log "DRY   $CLEANER $run_dir/checkpoints $KEEP_ROUNDS $CLEAN_INTERVAL $MAX_DIR_MB"
            continue
        fi

        if ! check_memory "$cfg"; then
            log "FAIL  $name (memory guard) — skipping"
            (( failed++ ))
            continue
        fi

        if ! check_disk "$(dirname "$run_dir")"; then
            log "FAIL  $name (disk guard) — skipping"
            (( failed++ ))
            continue
        fi

        log "START $name  -> $log_file"
        run_started=$(date +%s)

        # An earlier attempt at this run leaves its checkpoints in the same
        # directory, and that residue is poison twice over: the cleaner keys off
        # the newest round it can see, and residue numbered *above* the live run
        # sits past every cutoff it will ever compute, so it is never reclaimed.
        # The attempt starts from round 1 regardless, so none of it is resumable.
        if [[ -d "$run_dir/checkpoints" ]]; then
            stale=$(find "$run_dir/checkpoints" -maxdepth 1 -type f 2>/dev/null | wc -l)
            if [[ "$stale" -gt 0 ]]; then
                log "      clearing $stale stale checkpoint file(s) from a previous attempt"
                rm -rf "${run_dir:?}/checkpoints"
            fi
        fi

        if $DO_CLEAN; then
            # Supervised, not bare.  On 2026-08-13 the gap40 cleaner died without
            # a word; nothing noticed, 94 GB accumulated, the volume filled and
            # the trainer hung mid-round with no error in its log.  A cleaner that
            # can stop protecting the run while the run keeps going is the whole
            # failure mode, so respawn it for as long as the run lives.
            (
                trap 'kill "${child:-}" 2>/dev/null; exit 0' TERM INT
                while true; do
                    "$CLEANER" "$run_dir/checkpoints" "$KEEP_ROUNDS" \
                        "$CLEAN_INTERVAL" "$MAX_DIR_MB" &
                    child=$!
                    wait "$child"
                    echo "[$(date '+%F %T')] [supervisor] cleaner exited rc=$? — restarting in 30s"
                    sleep 30
                done
            ) >> "$LOG_DIR/${name}.cleaner.log" 2>&1 &
            CLEANER_PID=$!
        fi

        # `exec` matters: without it $! is the subshell's PID, and killing that
        # leaves the trainer reparented to init and still writing checkpoints.
        ( cd "$EXAMPLE_DIR" && exec env PYTHONPATH="$HERE:$REPO_ROOT" \
            "$PYTHON" "$ENTRY" -c "$cfg" ) > "$log_file" 2>&1 &
        RUN_PID=$!
        wait "$RUN_PID"
        status=$?
        RUN_PID=""

        if [[ -n "$CLEANER_PID" ]]; then
            kill "$CLEANER_PID" 2>/dev/null
            wait "$CLEANER_PID" 2>/dev/null
            CLEANER_PID=""
            "$CLEANER" "$run_dir/checkpoints" "$KEEP_ROUNDS" --once "$MAX_DIR_MB" \
                >> "$LOG_DIR/${name}.cleaner.log" 2>&1
        fi

        elapsed=$(( $(date +%s) - run_started ))
        if [[ "$status" -eq 0 ]]; then
            date '+%F %T' > "$done_marker"
            log "DONE  $name in $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
            (( completed++ ))
        else
            log "FAIL  $name exit=$status after $(( elapsed / 60 ))m — see $log_file"
            if [[ "$status" -eq 137 ]]; then
                log "      exit 137 = SIGKILL, almost always the cgroup OOM killer."
                log "      Check: cat /sys/fs/cgroup/memory/memory.failcnt"
            fi
            log "      continuing with the remaining gaps; no .done marker written"
            (( failed++ ))
        fi
      done
    done
done

total=$(( $(date +%s) - started_at ))
echo "[$(date '+%F %T')] finished: $completed done, $skipped skipped, $failed failed," \
     "in $(( total / 3600 ))h $(( (total % 3600) / 60 ))m"
[[ "$failed" -eq 0 ]]
