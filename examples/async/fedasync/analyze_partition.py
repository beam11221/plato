"""
Standalone script to analyze the Dirichlet non-IID partition distribution
for Plato's 'noniid' sampler, matching the logic in plato/samplers/dirichlet.py.

Usage:
    python analyze_partition.py [--config path/to/config.toml] [--seed 42] [--plot]

If --seed is not given, a fixed seed of 42 is used so results are reproducible.
To match an actual training run you must set `random_seed` in the TOML and pass
the same value here.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms


# ── defaults matching the config ─────────────────────────────────────────────
DEFAULT_TOTAL_CLIENTS = 100
DEFAULT_PARTITION_SIZE = 500
DEFAULT_CONCENTRATION = 0.9
DEFAULT_DATA_PATH = str(Path(__file__).parent / "data")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-clients", type=int, default=DEFAULT_TOTAL_CLIENTS)
    p.add_argument("--partition-size", type=int, default=DEFAULT_PARTITION_SIZE)
    p.add_argument("--concentration", type=float, default=DEFAULT_CONCENTRATION)
    p.add_argument("--seed", type=int, default=8888,
                   help="random_seed value (set the same in your TOML for reproducibility)")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--plot", action="store_true", help="show matplotlib heatmap")
    p.add_argument("--top", type=int, default=10,
                   help="print label counts only for the first N clients")
    return p.parse_args()


def replicate_dirichlet_sampler(targets: np.ndarray, num_classes: int,
                                 concentration: float, partition_size: int,
                                 random_seed: int, client_id: int) -> np.ndarray:
    """Mirrors plato/samplers/dirichlet.py exactly."""
    np.random.seed(random_seed * client_id)

    proportions = np.random.dirichlet(np.repeat(concentration, num_classes))
    if np.isnan(proportions.sum()):
        proportions = np.zeros(num_classes)
        proportions[np.random.randint(0, num_classes)] = 1.0

    weights = proportions[targets].tolist()

    gen = torch.Generator()
    gen.manual_seed(random_seed)

    subset_indices = list(
        WeightedRandomSampler(
            weights=weights,
            num_samples=partition_size,
            replacement=False,
            generator=gen,
        )
    )
    return np.array(subset_indices)


def main():
    args = parse_args()

    print(f"Loading CIFAR-10 from {args.data_path} …")
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True,
        transform=transforms.ToTensor(),
    )
    targets = np.array(trainset.targets)
    classes = trainset.classes          # 10 class names
    num_classes = len(classes)
    dataset_size = len(targets)

    print(f"\nSettings: {args.total_clients} clients | "
          f"partition_size={args.partition_size} | "
          f"concentration={args.concentration} | seed={args.seed}\n")

    label_matrix = np.zeros((args.total_clients, num_classes), dtype=int)
    all_indices_list = []   # NEW: keep raw indices for overlap/coverage analysis

    for cid in range(1, args.total_clients + 1):
        indices = replicate_dirichlet_sampler(
            targets, num_classes,
            args.concentration, args.partition_size,
            args.seed, cid,
        )
        all_indices_list.append(indices)
        label_matrix[cid - 1] = np.bincount(targets[indices], minlength=num_classes)

    # ── per-client summary ────────────────────────────────────────────────────
    header = f"{'Client':>7}  {'#samples':>8}  " + "  ".join(f"{c:>6}" for c in classes)
    print(header)
    print("-" * len(header))

    n_print = min(args.top, args.total_clients)
    for cid in range(1, n_print + 1):
        counts = label_matrix[cid - 1]
        row = f"{cid:>7}  {counts.sum():>8}  " + "  ".join(f"{v:>6}" for v in counts)
        print(row)

    if n_print < args.total_clients:
        print(f"  … (showing first {n_print} of {args.total_clients}; "
              "use --top N to see more)")

    # ── aggregate stats ───────────────────────────────────────────────────────
    totals = label_matrix.sum(axis=0)
    per_client_totals = label_matrix.sum(axis=1)
    print("\n── Aggregate label counts across all clients ──")
    for i, c in enumerate(classes):
        print(f"  {c:>10}: {totals[i]:6d}")

    print(f"\n── Per-client sample count ──")
    print(f"  min={per_client_totals.min()}  max={per_client_totals.max()}  "
          f"mean={per_client_totals.mean():.1f}  std={per_client_totals.std():.1f}")

    dominant = label_matrix.argmax(axis=1)
    dom_counts = np.bincount(dominant, minlength=num_classes)
    print("\n── Dominant class distribution across clients ──")
    for i, c in enumerate(classes):
        bar = "█" * dom_counts[i]
        print(f"  {c:>10}: {dom_counts[i]:3d} clients  {bar}")

    # ── NEW: realized global label distribution ───────────────────────────────
    # The aggregate p_global is NOT necessarily uniform, because Plato's
    # Dirichlet sampler runs independently per client (overlapping draws).
    # If your Δ_div term uses KL(p_i || p_global), use THIS as p_global,
    # not the assumed 1/K uniform.
    p_global_realized = totals / totals.sum()
    p_uniform = np.ones(num_classes) / num_classes
    print("\n── Realized global label distribution (use this as p_global for Δ_div) ──")
    print(f"  {'class':>10}  {'p_realized':>11}  {'p_uniform':>11}  {'Δ':>9}")
    for i, c in enumerate(classes):
        delta = p_global_realized[i] - p_uniform[i]
        print(f"  {c:>10}  {p_global_realized[i]:>11.4f}  {p_uniform[i]:>11.4f}  {delta:>+9.4f}")
    print(f"\n  Max |Δ| from uniform: {np.abs(p_global_realized - p_uniform).max():.4f}")
    print(f"  L1 distance from uniform: {np.abs(p_global_realized - p_uniform).sum():.4f}")

    # ── NEW: sample overlap & coverage diagnostics ───────────────────────────
    # WeightedRandomSampler(replacement=False) only forbids duplicates WITHIN
    # one client's partition. Across clients, the same training image can
    # appear in many partitions, which is why the per-class totals can exceed
    # CIFAR-10's true class size of 5000.
    all_indices = np.concatenate(all_indices_list)
    total_draws = len(all_indices)
    uniq, counts = np.unique(all_indices, return_counts=True)
    coverage = len(uniq) / dataset_size

    print("\n── Sample overlap across clients ──")
    print(f"  (replacement=False is per-client only; cross-client overlap is expected)")
    print(f"  Total draws (Σ over clients):    {total_draws}")
    print(f"  CIFAR-10 train set size:         {dataset_size}")
    print(f"  Unique samples ever drawn:       {len(uniq)} ({coverage:.1%} of dataset)")
    print(f"  Samples never drawn:             {dataset_size - len(uniq)} "
          f"({(dataset_size - len(uniq))/dataset_size:.1%})")
    print(f"  Mean draws per *drawn* sample:   {counts.mean():.2f}")
    print(f"  Median / p95 / max multiplicity: "
          f"{int(np.median(counts))} / {int(np.percentile(counts, 95))} / {counts.max()}")

    mult_dist = Counter(counts.tolist())
    print(f"\n  Multiplicity histogram (samples drawn N times):")
    max_bar_count = max(mult_dist.values())
    for n in sorted(mult_dist.keys()):
        bar_len = int(40 * mult_dist[n] / max_bar_count)
        bar = "█" * bar_len
        print(f"    drawn {n:>2}× : {mult_dist[n]:6d} samples  {bar}")

    # ── optional heatmap ──────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Heatmap: clients × classes (proportion)
            prop_matrix = label_matrix / label_matrix.sum(axis=1, keepdims=True)
            im = axes[0].imshow(prop_matrix, aspect="auto", cmap="YlOrRd")
            axes[0].set_title(f"Label proportion per client\n"
                              f"(Dirichlet α={args.concentration}, {args.total_clients} clients)")
            axes[0].set_xlabel("Class")
            axes[0].set_ylabel("Client ID")
            axes[0].set_xticks(range(num_classes))
            axes[0].set_xticklabels(classes, rotation=45, ha="right")
            plt.colorbar(im, ax=axes[0], label="Proportion")

            # Bar chart: total label counts (with reference line for uniform)
            axes[1].bar(classes, totals)
            axes[1].axhline(totals.sum() / num_classes, color="k", linestyle="--",
                            linewidth=1, label=f"uniform = {totals.sum()/num_classes:.0f}")
            axes[1].set_title("Total samples per class across all clients\n"
                              "(may exceed 5000 due to cross-client overlap)")
            axes[1].set_xlabel("Class")
            axes[1].set_ylabel("Count")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].legend()

            plt.tight_layout()
            out = Path("./partition_distribution.png")
            plt.savefig(out, dpi=150)
            print(f"\nPlot saved to {out}")
            plt.show()
        except ImportError:
            print("\nmatplotlib not available — skipping plot. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()