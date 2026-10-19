#!/usr/bin/env bash
# Regenerate the heterogeneity sweep (gap G = 1..40) used for the
# "same straggler, faster fast-tiers" comparison.
#
# Design decisions baked in here (see README_gpu_tier.md "Heterogeneity sweep"):
#   * tiers are log-spaced       -> --mid-mode geometric, so Orin stays a real
#                                   middle tier instead of collapsing onto Thor
#   * the STRAGGLER is anchored  -> --anchor-tier xavier, so Xavier sleeps the
#                                   same 20 s in every setting and the real
#                                   run time per setting is identical
#   * the anchor sits well above -> --anchor-sleep 20, ~10x the ~2 s additive
#     the framework's floor         floor (comm + real training + processing)
#                                   that the server also charges to wall time
#
# It also regenerates the CPU-predictor arm (staleness_predictor = "cpu"):
# tiers_cpu.csv is what the predictor believes (GHz only, 1.30x spread), and
# client_predicted_times_cpu_xa<N>.csv is that belief written per client on the
# same Xavier anchor as the GPU files, for side-by-side reporting.  One CPU file
# serves all nine gaps — it is hardware spec, so it does not move with the gap.
#
# Usage:  ./make_gap_sweep.sh [dataset_size] [epochs] [anchor_seconds]
set -euo pipefail

cd "$(dirname "$0")"

DATASET_SIZE="${1:-500}"
EPOCHS="${2:-1}"
ANCHOR_SLEEP="${3:-20}"
GAPS=(01 05 10 15 20 25 30 35 40)

echo "Generating heterogeneity sweep for dataset size ${DATASET_SIZE}, ${EPOCHS} epochs, anchor sleep ${ANCHOR_SLEEP} s"

python3 make_tier_gaps.py --mid-mode geometric --prefix tiers_geo

for gap in "${GAPS[@]}"; do
    python3 generate_sleep_times.py \
        --dataset-size "${DATASET_SIZE}" \
        --epochs "${EPOCHS}" \
        --tiers-csv "input/tiers_geo_gap${gap}.csv" \
        --anchor-tier xavier \
        --anchor-sleep "${ANCHOR_SLEEP}" \
        -o "output/client_sleep_times_xa${ANCHOR_SLEEP}_gap${gap}.csv"
done

# CPU-predictor arm — belief only; clients still sleep by the GPU files above.
python3 make_cpu_tiers.py

python3 generate_sleep_times.py \
    --dataset-size "${DATASET_SIZE}" \
    --epochs "${EPOCHS}" \
    --tiers-csv "input/tiers_cpu.csv" \
    --anchor-tier xavier \
    --anchor-sleep "${ANCHOR_SLEEP}" \
    -o "output/client_predicted_times_cpu_xa${ANCHOR_SLEEP}.csv"

python3 summarize_sweep.py \
    --anchor-sleep "${ANCHOR_SLEEP}" \
    --predictor-tiers input/tiers_cpu.csv
