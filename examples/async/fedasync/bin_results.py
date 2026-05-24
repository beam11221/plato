import pandas as pd
import numpy as np
import os
import sys


def bin_by_accuracy(input_path, acc_step=5, output_path=None):
    """For each accuracy threshold (multiples of acc_step %), find the first
    round and elapsed_time at which that accuracy was reached or exceeded."""
    df = pd.read_csv(input_path)
    df = df.dropna(subset=["accuracy"])
    df["round"] = df["round"].astype(int)
    df = df.sort_values("round").reset_index(drop=True)

    max_acc = df["accuracy"].max()

    # Accuracy may be stored as a fraction [0, 1] or already as a percentage [0, 100].
    # Normalise to percentage for consistent threshold stepping.
    is_fraction = max_acc <= 1.0
    acc_col = df["accuracy"] * 100 if is_fraction else df["accuracy"]
    max_acc_pct = acc_col.max()

    thresholds = np.arange(acc_step, max_acc_pct + acc_step, acc_step)

    records = []
    for thresh in thresholds:
        reached = df[acc_col >= thresh]
        if reached.empty:
            continue
        first = reached.iloc[0]
        records.append({
            "accuracy_pct": round(thresh, 2),
            "first_round":  int(first["round"]),
            "elapsed_time": first["elapsed_time"],
        })

    result = pd.DataFrame(records)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_accuracy_bins{ext}"

    result.to_csv(output_path, index=False)
    print(f"Saved accuracy bins ({len(result)} thresholds) to: {output_path}")
    print(result.to_string(index=False))
    return result


def bin_results(input_path, bin_size=200, output_path=None):
    df = pd.read_csv(input_path)

    # Drop rows where valid_loss is missing
    df = df.dropna(subset=["valid_loss"])
    df["round"] = df["round"].astype(int)

    # Assign bin label: rounds 1-200 -> bin 1, 201-400 -> bin 2, etc.
    df["bin"] = ((df["round"] - 1) // bin_size) + 1

    records = []
    for bin_id, group in df.groupby("bin"):
        # Representative row: the one with lowest valid_loss in this bin
        best_idx = group["valid_loss"].idxmin()
        best = group.loc[best_idx]

        round_start = (bin_id - 1) * bin_size + 1
        round_end = bin_id * bin_size

        records.append({
            "bin": bin_id,
            "round_range": f"{round_start}-{round_end}",
            "best_round": int(best["round"]),
            "valid_loss": best["valid_loss"],
            "accuracy": best["accuracy"],
            "train_loss": best["train_loss"],
        })

    result = pd.DataFrame(records)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_binned{ext}"

    result.to_csv(output_path, index=False)
    print(f"Saved binned results ({len(result)} bins) to: {output_path}")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the example file
        input_path = (
            "plato/examples/async/fedasync/runs/"
            "fedasync_CIFAR10_vgg11_100client_noniid_lite_GN_ver2_alpha03/"
            "results/90627.csv"
        )
    else:
        input_path = sys.argv[1]

    bin_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 200
    output_path = sys.argv[3] if len(sys.argv) >= 4 else None

    bin_results(input_path, bin_size=bin_size, output_path=output_path)

    acc_step = float(sys.argv[4]) if len(sys.argv) >= 5 else 5
    bin_by_accuracy(input_path, acc_step=acc_step)
