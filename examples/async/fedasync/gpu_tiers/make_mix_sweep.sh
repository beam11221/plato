#!/usr/bin/env bash
# Regenerate the FLEET-MIX sweep: one gap G, four compositions.
#
# The gap sweep moves the tier speeds and holds the fleet at 50/35/15.  This one
# does the opposite — G is pinned (default 30: xavier 20 s, orin 3.65 s,
# thor 0.667 s, effective gap 8.25x) and what moves is how many clients sit in
# each tier.  Design decisions baked in here:
#
#   * SAME tier file for every cell   -> input/tiers_geo_gap<G>.csv untouched, so
#                                        "fix the GPU speeds, change the ratio"
#                                        is literally true: a xavier client
#                                        sleeps 20 s in all four cells.
#   * SAME anchor (xavier = 20 s)     -> anchor_scale() reads the tier spec only,
#                                        never the fleet, so the anchor does not
#                                        drift when the composition changes.
#   * SAME seed, same partition       -> client i keeps its 500 images in every
#                                        cell (make_mixes.py); only the speed
#                                        label attached to it moves.
#   * t_round_model = "queue"         -> the "min" model's error depends on the
#                                        POOL's composition, i.e. on this sweep's
#                                        own x-axis: ~0 % at 50/35/15, +194 % at
#                                        85/10/5, -21 % at 15/35/50.  Running the
#                                        mix sweep under "min" would confound the
#                                        selector's steering strength with the
#                                        variable being swept (HANDOFF § 4.14).
#                                        This is why the 50/35/15 cell is re-run
#                                        rather than reused from runs/gap_sweep,
#                                        which is a "min" sweep.
#
# Usage:  ./make_mix_sweep.sh [gap] [dataset_size] [epochs] [anchor_seconds]
set -euo pipefail

cd "$(dirname "$0")"

GAP="${1:-30}"
DATASET_SIZE="${2:-500}"
EPOCHS="${3:-1}"
ANCHOR_SLEEP="${4:-20}"
MIXES=(m50_35_15 m70_20_10 m34_33_33 m15_35_50)

echo "Fleet-mix sweep at G=${GAP}: ${#MIXES[@]} compositions, dataset ${DATASET_SIZE}, ${EPOCHS} epoch(s), anchor ${ANCHOR_SLEEP}s"
echo

# 1. The compositions themselves (client -> tier).
python3 make_mixes.py
echo

# 2. Reality: one sleep CSV per composition, from the SAME tier spec.
for mix in "${MIXES[@]}"; do
    python3 generate_sleep_times.py \
        --dataset-size "${DATASET_SIZE}" \
        --epochs "${EPOCHS}" \
        --tiers-csv "input/tiers_geo_gap${GAP}.csv" \
        --clients-csv "input/clients_${mix}.csv" \
        --anchor-tier xavier \
        --anchor-sleep "${ANCHOR_SLEEP}" \
        -o "output/client_sleep_times_xa${ANCHOR_SLEEP}_gap${GAP}_${mix}.csv"
done
echo

# 3. Configs.  Separate directory and separate runs/ subtree from the gap sweep:
#    these are "queue" runs and must never be mixed in with the "min" ones.
python3 make_configs.py \
    --gaps "${GAP}" \
    --anchor "${ANCHOR_SLEEP}" \
    --mixes "${MIXES[@]}" \
    --t-round-model queue \
    --target-accuracy 1.01 \
    --rounds 5000 \
    --runs-dir mix_sweep \
    --outdir configs_mix

echo
echo "Then:  CONFIG_DIR=\$PWD/configs_mix ./run_sweep.sh gpu cpu baseline \\"
echo "           --gaps ${GAP} --mixes \"${MIXES[*]}\""
