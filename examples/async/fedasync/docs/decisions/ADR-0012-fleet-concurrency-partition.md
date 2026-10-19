# ADR-0012: N = 30, C = 3, and a disjoint 50,000-image partition

Status: accepted 14–15 Oct 2026, implemented 16 Oct 2026.

## Decision

**Fleet.**
- N = 30: 15 Xavier, 10 Orin and 5 Thor, in contiguous ID blocks (Wen Table 2: K = 30).
- Client i is mapped to `veh{i−1:02d}`.
- The mean speed per tier comes out as 26.8 / 24.9 / 28.1 m/s. That is reported, not rebalanced.

**Concurrency.** C = 3, i.e. C/N = 10% (Pisces, Polaris). The staleness bound is ceil(6.25 × C) = 19.

**Data.**
- CIFAR-10 with ResNet-18.
- Dirichlet α = 0.8.
- 20 clients get 1,667 images and 10 clients get 1,666 (clients 21–30), totalling exactly 50,000, with no image reused.

## Why a new sampler

PLATO's `noniid` sampler draws every client's subset independently from the whole set. Images therefore repeat across clients, and the total is not tied to the dataset size.

`wen_env/wen_partition.py` builds the split once, deterministically from `[data] random_seed`:
1. Draw the Dirichlet proportions.
2. Run Sinkhorn so the row sums are |S_i| and every class's column sum is exactly 5,000.
3. Round with both marginals preserved.
4. Deal out shuffled per-class indices.

`wen_client.WenLifecycle` installs it on every client process and logs a fingerprint of each partition.

## Cost

Sinkhorn moves each client's label distribution away from its own Dirichlet draw. For seed 4126 on balanced CIFAR-10 labels, the mean total-variation distance is 0.063.

The split is still clearly non-IID. The mean largest-class share per client stays above 0.25, where an IID split would give 0.10.
