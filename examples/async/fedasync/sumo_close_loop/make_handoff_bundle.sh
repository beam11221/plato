#!/bin/bash
# Pack everything the second machine needs that is NOT in git.
#
# The fedasync_*.py entry points are tracked, so a `git pull` brings those. These
# three directories are not tracked and must be copied:
#
#   sumo_close_loop/   this experiment
#   gpu_tiers/         gpu_tier_sleep.py, generate_sleep_times.py, the cleaner
#   surgical/input/    the tier specs both machines must agree on, byte for byte
#   surgical/make_fleet.py
#   docs/decisions/    the ADRs HANDOFF.md refers to, ADR-0004 in particular
#
# `trace.csv` is deliberately excluded: it is 228 MB and nothing at runtime reads
# it. build_inputs.sh regenerates every input from `vehicles.csv` and
# `coverage_windows.csv`, which are small and are included. Keep trace.csv on the
# machine that produced it.
#
# Usage:  ./make_handoff_bundle.sh [output.tar.gz]

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FEDASYNC_DIR="$(dirname "$HERE")"
OUT="${1:-$HOME/sumo_handoff.tar.gz}"

cd "$FEDASYNC_DIR"

tar czf "$OUT" \
    --exclude='*/trace.csv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='sumo_close_loop/logs' \
    --exclude='sumo_close_loop/configs/*_t*.toml' \
    sumo_close_loop \
    gpu_tiers/gpu_tier_sleep.py \
    gpu_tiers/generate_sleep_times.py \
    gpu_tiers/clean_checkpoints.sh \
    gpu_tiers/HANDOFF.md \
    surgical/make_fleet.py \
    surgical/input \
    surgical/BRIEF_surgical.md \
    surgical/CODE_AUDIT.md \
    docs/decisions

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo
echo "On the other machine, from the same place in its plato checkout:"
echo "    cd <plato>/examples/async/fedasync && tar xzf $(basename "$OUT")"
echo "    cat sumo_close_loop/HANDOFF.md"
