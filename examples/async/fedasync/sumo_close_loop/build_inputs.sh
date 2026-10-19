#!/bin/bash
# Build every input file the SUMO experiment needs, for one fleet size.
#
#   input/clients_<cell>.csv    client_id -> tier   (the fleet composition)
#   input/coverage_<cell>.csv   client_id -> RSU coverage windows
#   input/mapping_<cell>.csv    client_id -> veh_id (audit trail)
#   output/client_sleep_times_<cell>.csv   what each client actually sleeps
#   configs/{baseline,gpu,cpu,observed}_<cell>.toml
#
# The three input files describe one fleet from three sides and must be
# regenerated together; crossing files from different cells is silent.
#
# Usage:  ./build_inputs.sh n20
#         ./build_inputs.sh n100

set -euo pipefail

CELL="${1:-n20}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FEDASYNC_DIR="$(dirname "$HERE")"
GPU_TIERS="$FEDASYNC_DIR/gpu_tiers"

case "$CELL" in
    n20)  TOTAL=20;  TRACE="$HERE/sumo_trace/e1_closed_n20_seed1"  ;;
    n100) TOTAL=100; TRACE="$HERE/sumo_trace/e1_closed_n100_seed1" ;;
    *) echo "unknown cell '$CELL' (expected n20 or n100)" >&2; exit 1 ;;
esac

ANCHOR_SLEEP=15
PARTITION_SIZE=2500
TIERS_CSV="$FEDASYNC_DIR/surgical/input/tiers_estimated_15x.csv"

echo "== 1/4  fleet composition (50/35/15 over $TOTAL clients)"
python3 "$HERE/../surgical/make_fleet.py" \
    --total "$TOTAL" --mix 50 35 15 \
    -o "$HERE/input/clients_$CELL.csv"

echo "== 2/4  RSU coverage windows, remapped onto client IDs"
python3 "$HERE/remap_fleet.py" \
    --trace-dir "$TRACE" \
    --clients-csv "$HERE/input/clients_$CELL.csv" \
    --out-coverage "$HERE/input/coverage_$CELL.csv" \
    --out-mapping "$HERE/input/mapping_$CELL.csv"

echo "== 3/4  sleep times, anchored at xavier = ${ANCHOR_SLEEP}s"
python3 "$GPU_TIERS/generate_sleep_times.py" \
    --dataset-size "$PARTITION_SIZE" \
    --tiers-csv "$TIERS_CSV" \
    --clients-csv "$HERE/input/clients_$CELL.csv" \
    --anchor-tier xavier --anchor-sleep "$ANCHOR_SLEEP" \
    -o "$HERE/output/client_sleep_times_$CELL.csv"

echo "== 4/4  configs for the four arms"
python3 "$HERE/make_sumo_configs.py" --cell "$CELL"

echo
echo "Done. Next: python3 $HERE/preflight.py --cell $CELL"
