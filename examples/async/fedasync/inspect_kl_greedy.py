"""
Diagnostic script for KLGreedyClientSelectionStrategy.

Answers: "does the selector actually pick the most balanced clients?"

What this script does
---------------------
1. Loads CIFAR-10 and replicates PLATO's Dirichlet partition (same seed,
   same numpy call sequence as plato/samplers/dirichlet.py) for every client.
2. Computes KL(P || Q̃_i) for every client using the exact same function
   as the live selector (_kl_divergence, α=0.5, ε=1e-300).
3. Prints:
   • Full ranked table: every client, sorted by KL ascending.
   • Label histograms for the top-5 lowest and top-5 highest KL clients
     so you can visually confirm balanced vs. skewed.
   • Per-round simulation: K simulated async rounds where each round a
     random subset of clients is "available" (busy clients excluded), and
     we show which 10 are selected, their mean KL vs. pool mean KL.
   • Summary statistics confirming selected pool KL < full pool KL.
4. Optionally saves a PNG with three panels (run with --plot).

This replicates exactly the logic in fedasync_kl_greedy_selection.py so
any discrepancy between this output and the live experiment log is a bug.

Usage
-----
    cd plato/examples/async/fedasync

    # Quick check (no plot, uses existing CIFAR-10 data)
    python inspect_kl_greedy.py

    # Full run with plots
    python inspect_kl_greedy.py --plot --rounds 20

    # Different config
    python inspect_kl_greedy.py --total-clients 100 --per-round 10 \\
        --concentration 0.5 --seed 8888 --partition-size 500 \\
        --partition-dist-low 0 --partition-dist-high 30
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler

# ---------------------------------------------------------------------------
# Constants matching fedasync_kl_greedy_selection.py exactly
# ---------------------------------------------------------------------------
_ALPHA: float = 0.5
_EPSILON: float = 1e-300

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# KL computation — identical copy of the live selector's static method
# ---------------------------------------------------------------------------

def kl_divergence(p: np.ndarray, counts_i: np.ndarray, alpha: float = _ALPHA) -> float:
    """KL(P || Q̃_i) with Jeffreys smoothing on Q_i only.

    Identical to KLGreedyClientSelectionStrategy._kl_divergence.
    If you change the selector's formula, update this too.
    """
    K = len(p)
    n_i = float(counts_i.sum())
    if n_i == 0:
        return math.inf
    denom = n_i + alpha * K
    kl = 0.0
    for k in range(K):
        if p[k] <= 0.0:
            continue
        q_tilde_k = (float(counts_i[k]) + alpha) / denom
        kl += p[k] * math.log(p[k] / (q_tilde_k + _EPSILON))
    return max(kl, 0.0)


# ---------------------------------------------------------------------------
# Partition generator — mirrors plato/samplers/dirichlet.py exactly
# ---------------------------------------------------------------------------

def _build_partitions(
    targets: np.ndarray,
    num_classes: int,
    total_clients: int,
    partition_size: int,
    concentration: float,
    random_seed: int,
    partition_dist_low: float | None,
    partition_dist_high: float | None,
) -> dict[int, np.ndarray]:
    """Return {client_id: counts_array} using PLATO's exact sampler logic."""
    label_counts: dict[int, np.ndarray] = {}

    for cid in range(1, total_clients + 1):
        # ── mirrors dirichlet.py __init__ ──────────────────────────────────
        np.random.seed(random_seed * cid)
        props = np.random.dirichlet(np.repeat(concentration, num_classes))
        if np.isnan(props.sum()):
            props = np.zeros(num_classes)
            props[np.random.randint(0, num_classes)] = 1.0
        weights = props[targets].tolist()

        # ── mirrors dirichlet.py num_samples() called inside get() ─────────
        if partition_dist_low is not None and partition_dist_high is not None:
            actual_size = int(
                partition_size * np.random.uniform(partition_dist_low, partition_dist_high)
            )
        else:
            actual_size = partition_size

        # ── mirrors dirichlet.py get() ─────────────────────────────────────
        if actual_size == 0:
            label_counts[cid] = np.zeros(num_classes, dtype=np.int64)
            continue

        gen = torch.Generator()
        gen.manual_seed(random_seed)
        indices = np.array(list(WeightedRandomSampler(
            weights=weights,
            num_samples=actual_size,
            replacement=False,
            generator=gen,
        )))
        label_counts[cid] = np.bincount(targets[indices], minlength=num_classes).astype(np.int64)

    return label_counts


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bar(counts: np.ndarray, width: int = 30) -> str:
    """Normalised ASCII bar chart for one client's label histogram."""
    total = counts.sum()
    if total == 0:
        return "(no data)"
    props = counts / total
    bar_max = props.max()
    lines = []
    for k, (c, name) in enumerate(zip(counts, CIFAR10_CLASSES)):
        filled = int(props[k] / bar_max * width) if bar_max > 0 else 0
        lines.append(f"  {name:>12} [{k}]: {'█' * filled:<{width}} {c:4d} ({props[k]*100:5.1f}%)")
    return "\n".join(lines)


def _stats(vals: list[float]) -> str:
    a = np.array([v for v in vals if math.isfinite(v)])
    if len(a) == 0:
        return "N/A"
    return (f"mean={a.mean():.4f}  median={np.median(a):.4f}  "
            f"p95={np.percentile(a, 95):.4f}  min={a.min():.4f}  max={a.max():.4f}")


# ---------------------------------------------------------------------------
# Main inspection logic
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # ── 1. Load CIFAR-10 test set to compute global P ──────────────────────
    print(f"Loading CIFAR-10 from {args.data_path} …")
    testset = torchvision.datasets.CIFAR10(
        root=str(args.data_path), train=False, download=True,
        transform=transforms.ToTensor(),
    )
    test_targets = np.array(testset.targets)
    test_counts = np.bincount(test_targets, minlength=10)
    P = test_counts.astype(float) / test_counts.sum()
    num_classes = len(P)

    print(f"\nGlobal test distribution P ({num_classes} classes):")
    print(f"  {np.round(P, 4).tolist()}")
    print(f"  (uniform check: max-min = {P.max()-P.min():.2e})\n")

    # ── 2. Load CIFAR-10 training set for partition generation ─────────────
    trainset = torchvision.datasets.CIFAR10(
        root=str(args.data_path), train=True, download=True,
        transform=transforms.ToTensor(),
    )
    train_targets = np.array(trainset.targets)

    pd_low = args.partition_dist_low
    pd_high = args.partition_dist_high

    print(f"Generating partitions for {args.total_clients} clients "
          f"(concentration={args.concentration}, seed={args.seed}, "
          f"partition_size={args.partition_size}"
          + (f", Uniform[{pd_low},{pd_high}]" if pd_low is not None else "")
          + ") …")

    label_counts = _build_partitions(
        train_targets, num_classes, args.total_clients,
        args.partition_size, args.concentration, args.seed,
        pd_low, pd_high,
    )

    # ── 3. Score every client ──────────────────────────────────────────────
    kl_scores = {
        cid: kl_divergence(P, label_counts[cid])
        for cid in range(1, args.total_clients + 1)
    }
    sorted_clients = sorted(kl_scores, key=kl_scores.__getitem__)

    # ── 4. Full ranked table ───────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("FULL RANKED TABLE  (sorted by KL ascending — top = most similar to global)")
    print(f"{'═'*80}")
    hdr = f"{'Rank':>4}  {'Client':>6}  {'#Samples':>8}  {'DomClass':>10}  {'DomProp':>7}  {'KL':>9}"
    print(hdr)
    print("-" * len(hdr))
    for rank, cid in enumerate(sorted_clients, 1):
        counts = label_counts[cid]
        n = counts.sum()
        dom = counts.argmax()
        dom_prop = counts[dom] / n if n > 0 else 0.0
        kl = kl_scores[cid]
        kl_str = f"{kl:.6f}" if math.isfinite(kl) else "    inf"
        print(f"{rank:>4}  {cid:>6}  {n:>8}  {CIFAR10_CLASSES[dom]:>10}  {dom_prop:>7.3f}  {kl_str:>9}")
        if args.top_n and rank >= args.top_n:
            remaining = args.total_clients - rank
            if remaining > 0:
                print(f"       … {remaining} more clients (use --top-n 0 to show all) …")
            break

    # ── 5. Label histograms: top-5 balanced and top-5 skewed ──────────────
    n_show = 5
    print(f"\n{'═'*80}")
    print(f"TOP-{n_show} LOWEST KL  (most similar to global — should look balanced)")
    print(f"{'═'*80}")
    for rank, cid in enumerate(sorted_clients[:n_show], 1):
        counts = label_counts[cid]
        kl = kl_scores[cid]
        print(f"\nRank {rank} | client={cid} | KL={kl:.6f} | total={counts.sum()}")
        print(_bar(counts))

    print(f"\n{'═'*80}")
    print(f"TOP-{n_show} HIGHEST KL  (most skewed — should look concentrated)")
    print(f"{'═'*80}")
    for rank, cid in enumerate(reversed(sorted_clients[-n_show:]), 1):
        counts = label_counts[cid]
        kl = kl_scores[cid]
        print(f"\nRank {args.total_clients - n_show + rank} | client={cid} | KL={kl:.6f} | total={counts.sum()}")
        print(_bar(counts))

    # ── 6. Async FL simulation ─────────────────────────────────────────────
    if args.rounds > 0:
        print(f"\n{'═'*80}")
        print(f"ASYNC FL SIMULATION  ({args.rounds} aggregation events, "
              f"concurrent_slots={args.per_round})")
        print(f"{'═'*80}")
        print(f"  How PLATO async FL actually works:")
        print(f"  • Server maintains exactly {args.per_round} concurrently-training clients (busy_set).")
        print(f"  • Initial startup: selects {args.per_round} clients at once (all slots empty).")
        print(f"  • Each event: 1 client finishes → aggregated immediately → server selects")
        print(f"    exactly 1 replacement (open_slots=1, effective_count=min(10,1,pool)=1).")
        print(f"  • select_clients() always receives clients_count=10 but open_slots caps it to 1.")

        rng = np.random.default_rng(args.seed)
        all_clients = list(range(1, args.total_clients + 1))
        selection_counts = {cid: 0 for cid in all_clients}

        # ── Initial selection: all 10 slots empty ─────────────────────────
        # Mirrors the first call to select_clients() after server starts:
        # clients_count=10, open_slots=10, effective_count=min(10,10,100)=10
        scored_all = sorted(all_clients, key=lambda c: kl_scores[c])
        busy_set: set[int] = set(scored_all[:args.per_round])
        for cid in busy_set:
            selection_counts[cid] += 1

        init_kl = [kl_scores[c] for c in busy_set if math.isfinite(kl_scores[c])]
        print(f"\n  ── Initial selection (all {args.per_round} slots open) ──")
        print(f"  Selected: {sorted(busy_set)}")
        print(f"  Active pool KL: mean={np.mean(init_kl):.4f}  "
              f"median={np.median(init_kl):.4f}  "
              f"p95={np.percentile(init_kl, 95):.4f}")

        # ── Per-event loop ─────────────────────────────────────────────────
        print(f"\n  ── Aggregation events (1 client finishes → 1 replacement selected) ──")
        print(f"  {'Event':>5}  {'Finished':>8}  {'KL_fin':>8}  "
              f"{'Selected':>8}  {'KL_sel':>8}  {'ActiveMeanKL':>13}  {'AvailMeanKL':>12}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*13}  {'-'*12}")

        event_rows = []

        for event in range(1, args.rounds + 1):
            # One client finishes — in real PLATO this is the first to return
            # its update. Here we sample randomly from busy_set.
            finished = int(rng.choice(sorted(busy_set)))
            busy_set.discard(finished)

            # Available = everyone not currently training
            available = [c for c in all_clients if c not in busy_set]

            # open_slots = per_round - len(busy_set) = 1 (one just finished)
            open_slots = args.per_round - len(busy_set)          # == 1
            effective_count = min(args.per_round, open_slots, len(available))  # == 1

            if effective_count > 0:
                sorted_avail = sorted(available, key=lambda c: kl_scores[c])
                new_client = sorted_avail[0]
                busy_set.add(new_client)
                selection_counts[new_client] += 1
            else:
                new_client = None  # no eligible clients (shouldn't happen here)

            active_kl = [kl_scores[c] for c in busy_set if math.isfinite(kl_scores[c])]
            avail_kl  = [kl_scores[c] for c in available if math.isfinite(kl_scores[c])]

            kl_fin = kl_scores[finished]
            kl_sel = kl_scores[new_client] if new_client is not None else float("nan")
            active_mean = np.mean(active_kl) if active_kl else float("nan")
            avail_mean  = np.mean(avail_kl)  if avail_kl  else float("nan")

            event_rows.append({
                "event": event, "finished": finished, "selected": new_client,
                "kl_fin": kl_fin, "kl_sel": kl_sel,
                "active_mean_kl": active_mean, "avail_mean_kl": avail_mean,
            })

            show = event <= args.show_first or event > args.rounds - args.show_last
            if show:
                fin_str = f"client_{finished}"
                sel_str = f"client_{new_client}" if new_client else "none"
                print(f"  {event:>5}  {fin_str:>8}  {kl_fin:>8.4f}  "
                      f"{sel_str:>8}  {kl_sel:>8.4f}  {active_mean:>13.4f}  {avail_mean:>12.4f}")
            elif event == args.show_first + 1:
                print(f"  … ({args.rounds - args.show_first - args.show_last} events hidden) …")

        # ── Summary ────────────────────────────────────────────────────────
        active_means = [r["active_mean_kl"] for r in event_rows]
        kl_improvements = [
            r["kl_fin"] - r["kl_sel"]
            for r in event_rows
            if r["selected"] is not None and math.isfinite(r["kl_fin"])
            and math.isfinite(r["kl_sel"])
        ]

        print(f"\n  ── Simulation summary over {args.rounds} aggregation events ──")
        print(f"  Active pool mean KL: start={active_means[0]:.4f}  "
              f"end={active_means[-1]:.4f}  "
              f"min={min(active_means):.4f}  max={max(active_means):.4f}")
        print(f"  Per-event KL swap (finished_KL - selected_KL): "
              f"mean={np.mean(kl_improvements):+.4f}  "
              f"(positive = we replaced a higher-KL client with a lower-KL one)")

        n_improved = sum(1 for d in kl_improvements if d > 0)
        n_worsened = sum(1 for d in kl_improvements if d < 0)
        print(f"  Events where KL improved (replaced high with low): {n_improved}/{len(kl_improvements)}")
        print(f"  Events where KL worsened (high-KL client happened to finish first): {n_worsened}/{len(kl_improvements)}")
        print(f"  Note: 'worsened' events happen when a low-KL client finishes first —")
        print(f"        the replacement is still the best available, so this is correct.")

        # Selection frequency
        freq_sorted = sorted(selection_counts, key=selection_counts.__getitem__, reverse=True)
        print(f"\n  ── Selection frequency (top-10 most-selected clients) ──")
        for cid in freq_sorted[:10]:
            freq = selection_counts[cid]
            kl = kl_scores[cid]
            bar = "█" * min(freq, 40)
            print(f"  client={cid:3d}  selected={freq:3d}x  KL={kl:.4f}  {bar}")

        print(f"\n  ── Selection frequency (top-10 LEAST-selected clients) ──")
        for cid in freq_sorted[-10:]:
            freq = selection_counts[cid]
            kl = kl_scores[cid]
            bar = "█" * freq if freq > 0 else "(never)"
            print(f"  client={cid:3d}  selected={freq:3d}x  KL={kl:.4f}  {bar}")

        top10_kl = [kl_scores[cid] for cid in freq_sorted[:10]]
        bot10_kl  = [kl_scores[cid] for cid in freq_sorted[-10:]]
        ok = np.mean(top10_kl) < np.mean(bot10_kl)
        print()
        print(f"  SANITY CHECK:")
        print(f"  mean KL (top-10 most selected):  {np.mean(top10_kl):.4f}")
        print(f"  mean KL (top-10 least selected): {np.mean(bot10_kl):.4f}")
        print(f"  RESULT: {'✓ PASS — selector correctly favours low-KL clients' if ok else '✗ FAIL — investigate!'}")

    # ── 7. Global distribution stats ──────────────────────────────────────
    all_kl_finite = [v for v in kl_scores.values() if math.isfinite(v)]
    n_empty = sum(1 for v in kl_scores.values() if math.isinf(v))

    print(f"\n{'═'*80}")
    print("GLOBAL KL DISTRIBUTION ACROSS ALL CLIENTS")
    print(f"{'═'*80}")
    print(f"  Total clients: {args.total_clients}  "
          f"(empty/no-data: {n_empty})")
    print(f"  KL stats: {_stats(all_kl_finite)}")

    # Histogram of KL scores
    bins = np.linspace(0, max(all_kl_finite) * 1.05, 11)
    hist, edges = np.histogram(all_kl_finite, bins=bins)
    print("\n  KL score distribution (ASCII histogram):")
    max_count = hist.max()
    for i, count in enumerate(hist):
        bar_len = int(count / max_count * 40) if max_count > 0 else 0
        print(f"  [{edges[i]:5.2f}, {edges[i+1]:5.2f}): "
              f"{'█' * bar_len:<40} {count:3d}")

    # ── 8. Optional plot ───────────────────────────────────────────────────
    if args.plot:
        _make_plot(sorted_clients, label_counts, kl_scores, P, args)

    print(f"\n{'═'*80}")
    print("INSPECTION COMPLETE")
    print(f"{'═'*80}")


# ---------------------------------------------------------------------------
# Optional matplotlib plot
# ---------------------------------------------------------------------------

def _make_plot(
    sorted_clients: list[int],
    label_counts: dict[int, np.ndarray],
    kl_scores: dict[int, float],
    P: np.ndarray,
    args: argparse.Namespace,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("\n[Plot] matplotlib not available — skipping plot.")
        return

    num_classes = len(P)
    n = len(sorted_clients)
    label_matrix = np.zeros((n, num_classes))
    for i, cid in enumerate(sorted_clients):
        counts = label_counts[cid]
        total = counts.sum()
        if total > 0:
            label_matrix[i] = counts / total

    kl_vals = [kl_scores[cid] for cid in sorted_clients]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, n // 8)))

    # Panel 1: label proportion heatmap (sorted by KL)
    ax = axes[0]
    im = ax.imshow(label_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.6)
    ax.set_title("Label proportion per client\n(sorted by KL ascending ↓)", fontsize=11)
    ax.set_xlabel("Class")
    ax.set_ylabel("Client rank (0 = most similar to global)")
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(CIFAR10_CLASSES[:num_classes], rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, label="class proportion")

    # Panel 2: KL score per client (rank order)
    ax = axes[1]
    finite_kl = [v if math.isfinite(v) else 10.0 for v in kl_vals]
    ax.barh(range(n), finite_kl, color="steelblue", alpha=0.7)
    ax.axvline(x=np.mean([v for v in finite_kl]), color="red",
               linestyle="--", label=f"mean={np.mean(finite_kl):.3f}")
    # Shade the top-per_round selection zone
    ax.axhspan(-0.5, args.per_round - 0.5, alpha=0.12, color="green",
               label=f"top {args.per_round} selected (if all available)")
    ax.set_title("KL score per client\n(rank order)", fontsize=11)
    ax.set_xlabel("KL(P || Q̃_i)")
    ax.set_ylabel("Client rank")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # Panel 3: histogram of KL scores
    ax = axes[2]
    finite_kl_arr = np.array([v for v in kl_vals if math.isfinite(v)])
    ax.hist(finite_kl_arr, bins=20, color="steelblue", edgecolor="black",
            alpha=0.8, linewidth=0.5)
    ax.axvline(x=finite_kl_arr[:args.per_round].mean(), color="green",
               linestyle="--", label=f"selected mean (top {args.per_round})")
    ax.axvline(x=finite_kl_arr.mean(), color="red",
               linestyle="--", label="full pool mean")
    ax.set_title("Distribution of KL scores\nacross all clients", fontsize=11)
    ax.set_xlabel("KL(P || Q̃_i)")
    ax.set_ylabel("# clients")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = Path(args.plot_out)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved to {out}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect KLGreedy client selection behaviour offline."
    )
    p.add_argument("--total-clients", type=int, default=100)
    p.add_argument("--per-round", type=int, default=10,
                   help="Number of clients selected per round.")
    p.add_argument("--concentration", type=float, default=0.5,
                   help="Dirichlet concentration (matches TOML data.concentration).")
    p.add_argument("--seed", type=int, default=8888,
                   help="Partition random seed (matches TOML data.random_seed).")
    p.add_argument("--partition-size", type=int, default=500,
                   help="Base partition size (matches TOML data.partition_size).")
    p.add_argument("--partition-dist-low", type=float, default=0.0,
                   help="Uniform multiplier lower bound. Set to None to disable.")
    p.add_argument("--partition-dist-high", type=float, default=30.0,
                   help="Uniform multiplier upper bound. Set to None to disable.")
    p.add_argument("--no-partition-dist", action="store_true",
                   help="Use fixed partition size (disable Uniform multiplier).")
    p.add_argument("--data-path", type=str,
                   default=str(Path(__file__).parent / "data"),
                   help="Path to CIFAR-10 data directory.")
    p.add_argument("--rounds", type=int, default=30,
                   help="Number of aggregation events to simulate. 0 = skip.")
    p.add_argument("--show-first", type=int, default=10,
                   help="Show this many events from the start of the simulation.")
    p.add_argument("--show-last", type=int, default=3,
                   help="Show this many events from the end of the simulation.")
    p.add_argument("--top-n", type=int, default=30,
                   help="Rows to show in ranked table (0 = all).")
    p.add_argument("--plot", action="store_true",
                   help="Save a three-panel matplotlib figure.")
    p.add_argument("--plot-out", type=str, default="kl_greedy_inspection.png",
                   help="Output path for the plot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_partition_dist:
        args.partition_dist_low = None
        args.partition_dist_high = None
    run(args)


if __name__ == "__main__":
    main()
