#!/bin/bash
# Delete old PLATO checkpoints so a 10 000-round run does not fill the disk.
#
# Each round writes ~44 MB (ResNet-18 safetensors + PRNG state), so one full run
# would leave ~440 GB behind.  This keeps the most recent KEEP_ROUNDS and deletes
# everything older.
#
# Generalised from clean_old_checkpoints.sh / _ver2 / _ver3, which were three
# copies of the same file differing only in a hard-coded CHECKPOINT_DIR.  With 27
# runs in the sweep that pattern does not scale, so the directory is a required
# argument here and run_sweep.sh supplies it per run.
#
# Usage:
#   ./clean_checkpoints.sh <checkpoint_dir> [keep_rounds] [interval_minutes]
#   ./clean_checkpoints.sh <checkpoint_dir> 100 --once      # single pass, then exit
#
#   $1  CHECKPOINT_DIR    required; must be a .../checkpoints directory
#   $2  KEEP_ROUNDS       recent rounds to keep (default: 100)
#   $3  INTERVAL_MINUTES  sleep between passes (default: 20), or --once
#   $4  MAX_DIR_MB        hard ceiling in MB; 0 disables (default: 20000)
#
# run_sweep.sh starts one of these per run and kills it when that run finishes,
# so it is normally not launched by hand.

set -uo pipefail

CHECKPOINT_DIR="${1:-}"
KEEP_ROUNDS="${2:-100}"
INTERVAL_ARG="${3:-20}"
MAX_DIR_MB="${4:-20000}"

# Floor for the emergency prune in enforce_budget().  PLATO only ever resumes
# from the newest checkpoint, so a handful of rounds is enough to stay safe.
MIN_KEEP_ROUNDS=20

if [[ -z "$CHECKPOINT_DIR" ]]; then
    echo "ERROR: checkpoint directory is required." >&2
    echo "Usage: $0 <checkpoint_dir> [keep_rounds] [interval_minutes|--once]" >&2
    exit 2
fi

# Guard: this script deletes files, so refuse anything that is not a checkpoints
# directory.  A mistyped path would otherwise be deleted from silently.
if [[ "$(basename "$CHECKPOINT_DIR")" != "checkpoints" ]]; then
    echo "ERROR: refusing to clean '$CHECKPOINT_DIR' — basename is not 'checkpoints'." >&2
    exit 2
fi

ONCE=false
if [[ "$INTERVAL_ARG" == "--once" ]]; then
    ONCE=true
    INTERVAL_SECONDS=0
else
    INTERVAL_SECONDS=$(( INTERVAL_ARG * 60 ))
fi

log() { echo "[$(date '+%F %T')] [cleaner] $*"; }

get_latest_round() {
    # prng_state_<N>.pkl is written every round, so it is the reliable round marker.
    #
    # Take the round of the *most recently written* file, not the highest number
    # on disk.  A re-run writes into the same directory, so if an earlier attempt
    # got to round 4570 and died, the new attempt starting at round 1 would see
    # latest=4570 and a cutoff of 4470 — and delete every checkpoint the live run
    # had just written, including the one it resumes from.  That is exactly what
    # was sitting in baseline_xa20_gap40 on 2026-08-14.  Newest-by-mtime tracks
    # the running attempt and degrades to the same answer when there is no residue.
    # latest must be initialised: under `set -u` an empty directory would leave it
    # unset and the regex test below would abort the function.
    local newest latest=""
    newest=$(ls -t "$CHECKPOINT_DIR"/prng_state_*.pkl 2>/dev/null | head -1)
    if [[ -n "$newest" ]]; then
        latest=$(basename "$newest" .pkl)
        latest=${latest#prng_state_}
    fi
    [[ "$latest" =~ ^[0-9]+$ ]] || latest=0
    echo "$latest"
}

# Round number encoded in a checkpoint filename, or "" if it carries none.
# Pure parameter expansion on purpose: this runs once per file and a `sed` here
# would fork ~10 000 processes per pass.  current_round.pkl matches nothing and
# is therefore never a deletion candidate.
round_of() {
    local base="${1##*/}" r=""
    case "$base" in
        prng_state_*.pkl)       r=${base#prng_state_};       r=${r%.pkl} ;;
        numpy_prng_state_*.pkl) r=${base#numpy_prng_state_}; r=${r%.pkl} ;;
        *.safetensors.pkl)      r=${base%.safetensors.pkl};  r=${r##*_} ;;
        *.safetensors)          r=${base%.safetensors};      r=${r##*_} ;;
    esac
    [[ "$r" =~ ^[0-9]+$ ]] && echo "$r"
}

# Delete every round-numbered checkpoint file at or below $1.  Echoes the count.
#
# Walks the directory once instead of looping `seq 1 $cutoff` and stat'ing four
# candidate names per round.  The old form issued ~40 000 stat calls per pass by
# round 10 000; on this network mount a pass then took longer than the 5-minute
# interval it was scheduled at, so checkpoints outran the cleaner.
#
# current_round.pkl carries no round number in its name and is what PLATO resumes
# from — round_of() returns "" for it, so it is never a target.
prune_to_cutoff() {
    local cutoff="$1" deleted=0 f round
    while IFS= read -r f; do
        round=$(round_of "$f")
        [[ -n "$round" ]] || continue
        (( round <= cutoff )) && rm -f "$f" && (( deleted++ ))
    done < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f \
                  \( -name 'prng_state_*.pkl' -o -name 'numpy_prng_state_*.pkl' \
                     -o -name '*.safetensors' -o -name '*.safetensors.pkl' \) 2>/dev/null)
    echo "$deleted"
}

clean_once() {
    local latest_round cutoff deleted freed_before freed_after
    latest_round=$(get_latest_round)

    if [[ "$latest_round" -eq 0 ]]; then
        log "no checkpoints yet in $CHECKPOINT_DIR — skipping."
        return
    fi

    cutoff=$(( latest_round - KEEP_ROUNDS ))
    if [[ "$cutoff" -le 0 ]]; then
        log "latest=$latest_round, keeping last $KEEP_ROUNDS — nothing to delete yet."
        return
    fi

    freed_before=$(du -sm "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
    deleted=$(prune_to_cutoff "$cutoff")
    freed_after=$(du -sm "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
    log "latest=$latest_round cutoff<=$cutoff kept=$KEEP_ROUNDS deleted=$deleted files" \
        "(${freed_before}MB -> ${freed_after}MB)"
}

# Backstop for when the KEEP_ROUNDS policy is not enough.  That policy assumes
# rounds arrive at a known rate; if they arrive faster, or a pass is delayed, or
# an attempt's residue inflates the directory, it can still outgrow the volume
# between passes.  On 2026-08-13 baseline_xa20_gap40 reached 94 GB with a dead
# cleaner, filled the 200 GB volume, and the trainer hung mid-round with no error
# in its log.  Losing old checkpoints is recoverable; wedging the run is not.
enforce_budget() {
    [[ "$MAX_DIR_MB" -gt 0 ]] || return
    local size latest cutoff deleted
    size=$(du -sm "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
    [[ "${size:-0}" -gt "$MAX_DIR_MB" ]] || return

    latest=$(get_latest_round)
    cutoff=$(( latest - MIN_KEEP_ROUNDS ))
    log "WARN ${size}MB over ceiling ${MAX_DIR_MB}MB — emergency prune to last $MIN_KEEP_ROUNDS rounds"
    if [[ "$cutoff" -le 0 ]]; then
        log "WARN latest=$latest, nothing safe left to prune — the ceiling cannot be met"
        return
    fi
    deleted=$(prune_to_cutoff "$cutoff")
    log "emergency prune deleted=$deleted files (now $(du -sm "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)MB)"
}

log "started  dir=$CHECKPOINT_DIR  keep=$KEEP_ROUNDS  ceiling=${MAX_DIR_MB}MB  pid=$$"

if $ONCE; then
    clean_once
    enforce_budget
    exit 0
fi

# The run creates the directory at startup; wait rather than exiting, since
# run_sweep.sh launches the cleaner and the run at the same moment.
for _ in $(seq 1 60); do
    [[ -d "$CHECKPOINT_DIR" ]] && break
    sleep 10
done
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    log "ERROR: $CHECKPOINT_DIR never appeared after 10 min — exiting."
    exit 1
fi

# Exit promptly when run_sweep.sh signals us.  Bash defers signal handling until
# the current foreground child finishes, so a bare `sleep "$INTERVAL_SECONDS"`
# here would keep the cleaner alive for up to INTERVAL_MINUTES after the kill —
# long enough for the runner to move on, leaving this loop orphaned and pruning a
# finished run's directory forever.  Backgrounding the sleep and waiting on it
# makes the trap fire immediately.
trap 'log "stopping on signal"; kill "${nap:-}" 2>/dev/null; exit 0' TERM INT

while true; do
    clean_once
    enforce_budget
    sleep "$INTERVAL_SECONDS" & nap=$!
    wait "$nap"
done
