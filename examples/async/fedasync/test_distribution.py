"""
Compare three partition strategies for federated learning on CIFAR-10:
    1. IID baseline (uniform class weights, same Plato-style sampler)
    2. Dirichlet α=0.5  (moderate non-IID — current thesis setting)
    3. Dirichlet α=0.9  (mild non-IID — "is this enough non-IID?" check)

The thesis-relevant question this answers:
    V_i = s(τ̂) × Δ_div. If Δ_div is nearly constant across clients,
    the diversity term contributes no signal and V_i collapses to pure
    speed selection. Use the CV (coefficient of variation) of
    KL(p_i || uniform) across clients as the diagnostic.

Per-client metrics computed:
    • KL(p_i || uniform)        — how far this client's label dist is from IID
    • max class proportion      — peak class dominance
    • effective # classes (e^H) — soft count of represented classes

Usage:
    python compare_partitions.py [--seed 42] [--plot]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms


# ── partition generators ─────────────────────────────────────────────────────
def _sample_indices(weights, partition_size, random_seed):
    """Run Plato-style WeightedRandomSampler(replacement=False)."""
    gen = torch.Generator()
    gen.manual_seed(random_seed)
    return np.array(list(WeightedRandomSampler(
        weights=weights,
        num_samples=partition_size,
        replacement=False,
        generator=gen,
    )))


def generate_iid_partition(targets, num_classes, partition_size,
                           random_seed, client_id):
    """IID: uniform class weights (so each sample equally likely)."""
    np.random.seed(random_seed * client_id)
    # Uniform class proportions — identical mechanism, different weights only.
    proportions = np.ones(num_classes) / num_classes
    weights = proportions[targets].tolist()
    return _sample_indices(weights, partition_size, random_seed)


def generate_dirichlet_partition(targets, num_classes, concentration,
                                  partition_size, random_seed, client_id):
    """Mirrors plato/samplers/dirichlet.py exactly."""
    np.random.seed(random_seed * client_id)
    proportions = np.random.dirichlet(np.repeat(concentration, num_classes))
    if np.isnan(proportions.sum()):
        proportions = np.zeros(num_classes)
        proportions[np.random.randint(0, num_classes)] = 1.0
    weights = proportions[targets].tolist()
    return _sample_indices(weights, partition_size, random_seed)


# ── metric computations ──────────────────────────────────────────────────────
def build_label_matrix(partition_fn, targets, num_classes, total_clients):
    """Run partition_fn(client_id) for each client. Returns [N, K] int matrix."""
    label_matrix = np.zeros((total_clients, num_classes), dtype=int)
    for cid in range(1, total_clients + 1):
        idx = partition_fn(cid)
        label_matrix[cid - 1] = np.bincount(targets[idx], minlength=num_classes)
    return label_matrix


def per_client_metrics(label_matrix):
    """Per-client non-IID severity metrics."""
    num_classes = label_matrix.shape[1]
    p = label_matrix / label_matrix.sum(axis=1, keepdims=True)
    p_safe = np.where(p > 0, p, 1e-12)
    uniform = 1.0 / num_classes

    # KL(p_i || uniform)
    kl = np.sum(np.where(p > 0, p * np.log(p_safe / uniform), 0.0), axis=1)

    # Peak class proportion
    max_prop = p.max(axis=1)

    # Effective number of classes = exp(H(p))  (Hill number of order 1)
    entropy = -np.sum(np.where(p > 0, p * np.log(p_safe), 0.0), axis=1)
    eff_classes = np.exp(entropy)

    return {
        "kl_from_uniform": kl,
        "max_class_prop": max_prop,
        "effective_classes": eff_classes,
    }


def summarize(name, metrics, label_matrix, num_classes):
    print(f"\n── {name} ──")
    for key, vals in metrics.items():
        cv = vals.std() / vals.mean() if vals.mean() > 1e-9 else 0.0
        print(f"  {key:>22}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}  CV={cv:.3f}")

    totals = label_matrix.sum(axis=0)
    p_global = totals / totals.sum()
    l1_uniform = np.abs(p_global - 1.0 / num_classes).sum()
    print(f"  {'p_global L1 from unif':>22}: {l1_uniform:.4f}")


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_comparison(settings, classes, out_path):
    import matplotlib.pyplot as plt

    num_classes = len(classes)
    fig = plt.figure(figsize=(18, 10))

    # Top row: heatmaps with shared color scale so non-IID streaks are obvious
    vmax = 0.5
    for i, (name, lm, _) in enumerate(settings):
        ax = fig.add_subplot(2, 3, i + 1)
        p = lm / lm.sum(axis=1, keepdims=True)
        im = ax.imshow(p, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(f"{name}\nLabel proportion per client", fontsize=11)
        ax.set_xlabel("Class")
        ax.set_ylabel("Client ID" if i == 0 else "")
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, label="proportion")

    # Bottom row: distributions of three metrics
    metric_specs = [
        ("kl_from_uniform",  "KL(p_i || uniform)",       "lower = more IID"),
        ("max_class_prop",   "max class proportion",     "0.1 = perfectly balanced"),
        ("effective_classes","effective # classes (e^H)","10 = uniform; 1 = single-class"),
    ]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # green, orange, red

    for i, (key, label, sublabel) in enumerate(metric_specs):
        ax = fig.add_subplot(2, 3, 4 + i)
        # find shared bin range
        all_vals = np.concatenate([m[key] for _, _, m in settings])
        bins = np.linspace(all_vals.min(), all_vals.max(), 30)
        for (name, _, metrics), color in zip(settings, colors):
            vals = metrics[key]
            ax.hist(vals, bins=bins, alpha=0.55, label=name, color=color,
                    edgecolor="black", linewidth=0.4)
        ax.set_xlabel(f"{label}\n({sublabel})", fontsize=10)
        ax.set_ylabel("# clients")
        ax.set_title(f"Per-client distribution: {label}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--total-clients", type=int, default=100)
    parser.add_argument("--partition-size", type=int, default=500)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.9],
                        help="Dirichlet concentrations to compare against IID")
    parser.add_argument("--data-path", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print(f"Loading CIFAR-10 from {args.data_path} …")
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True,
        transform=transforms.ToTensor(),
    )
    targets = np.array(trainset.targets)
    classes = trainset.classes
    num_classes = len(classes)

    print(f"\nComparing partitions | {args.total_clients} clients × "
          f"{args.partition_size} samples | seed={args.seed}")

    settings = []

    # IID baseline
    iid_lm = build_label_matrix(
        partition_fn=lambda cid: generate_iid_partition(
            targets, num_classes, args.partition_size, args.seed, cid),
        targets=targets, num_classes=num_classes,
        total_clients=args.total_clients,
    )
    iid_metrics = per_client_metrics(iid_lm)
    settings.append(("IID", iid_lm, iid_metrics))
    summarize("IID baseline", iid_metrics, iid_lm, num_classes)

    # Dirichlet variants
    for alpha in args.alphas:
        lm = build_label_matrix(
            partition_fn=lambda cid, a=alpha: generate_dirichlet_partition(
                targets, num_classes, a, args.partition_size, args.seed, cid),
            targets=targets, num_classes=num_classes,
            total_clients=args.total_clients,
        )
        metrics = per_client_metrics(lm)
        settings.append((f"Dirichlet α={alpha}", lm, metrics))
        summarize(f"Dirichlet α={alpha}", metrics, lm, num_classes)

    # ── the thesis-relevant headline ──────────────────────────────────────────
    print("\n" + "=" * 84)
    print("THESIS-RELEVANT QUESTION: how much signal does Δ_div have?")
    print("=" * 84)
    print("  V_i = s(τ̂) × Δ_div.  For Δ_div to change selection ordering relative")
    print("  to pure-speed selection, it needs ABSOLUTE spread across clients.")
    print("  std(KL) and range(KL) are the right metrics. CV (= std/mean) can")
    print("  invert misleadingly when the mean shifts faster than the std — as")
    print("  happens here between α=0.5 and α=0.9. Always report absolute spread.")
    print(f"\n  {'Setting':>20}  {'mean KL':>9}  {'std KL':>9}  "
          f"{'range KL':>10}  {'p95 KL':>8}  {'CV':>6}")
    print(f"  {'-'*20}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*6}")
    for name, _, m in settings:
        kl = m["kl_from_uniform"]
        cv = kl.std() / kl.mean() if kl.mean() > 1e-9 else 0.0
        rng = kl.max() - kl.min()
        p95 = float(np.percentile(kl, 95))
        print(f"  {name:>20}  {kl.mean():>9.4f}  {kl.std():>9.4f}  "
              f"{rng:>10.4f}  {p95:>8.4f}  {cv:>6.3f}")

    print("\n  How to read the table:")
    print("  • mean(KL)   → typical client's distance from uniform (problem severity)")
    print("  • std(KL)    → absolute spread Δ_div has to work with (signal magnitude)")
    print("  • range(KL)  → how diverse the most and least skewed clients are")
    print("  • p95(KL)    → how skewed the worst-case clients get (drift risk)")
    print("  • CV         → normalized ratio; report for transparency but DO NOT")
    print("                 use as headline metric (can invert when mean shifts)\n")
    print("  Defense soundbite: cite mean & std & range of KL together. Avoid")
    print("  framing the choice of α purely in terms of CV.\n")

    if args.plot:
        try:
            out = Path(__file__).parent / "compare_partitions.png"
            plot_comparison(settings, classes, out)
        except ImportError:
            print("\nmatplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()