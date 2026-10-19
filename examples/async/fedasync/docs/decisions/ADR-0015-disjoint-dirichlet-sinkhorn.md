# ADR-0015: Disjoint Dirichlet partition with Sinkhorn balancing

Status: implemented 16 Oct 2026 (Prompt B), documented 16 Oct 2026 (Prompt C). ADR-0012 states the split; this ADR records how it is built and what it costs. That cost was flagged as undocumented in the Prompt B report.

## Context

The locked split is 20 clients × 1,667 images plus 10 × 1,666 = 50,000 CIFAR-10 training images, with Dirichlet α = 0.8 and no image reused.

PLATO's `noniid` sampler (`plato/samplers/dirichlet.py`) cannot produce this. Each client draws its subset independently from the whole training set, using `WeightedRandomSampler` without replacement within the client only. Images therefore repeat across clients, and the total is not tied to the dataset size.

A disjoint split that uses every image exactly once has to satisfy two constraints at the same time:
- row sums: each client's |S_i|;
- column sums: 5,000 images per class.

Independent Dirichlet draws almost never satisfy both.

## Decision

`wen_env/wen_partition.py` builds the split deterministically from `[data] random_seed`:

1. Draw label proportions for each client: p_i ~ Dir(0.8 · 1₁₀). The RNG is `default_rng([seed, 0x5EED])`.
2. Scale the 30 × 10 matrix of desired counts |S_i| · p_i with iterative proportional fitting (Sinkhorn) until its rows sum to |S_i| and its columns to 5,000. This is the minimum-KL adjustment of the drawn proportions that makes a disjoint split possible.
3. Round to integers with both marginals preserved:
   - floor every cell;
   - give the leftover units to the largest fractional parts, at most one per cell;
   - then give any remainder to any cell whose column still has stock.

   Because the row and column totals are equal, step 3 always completes.
4. Shuffle each class's image indices once and deal them out in client order.

**Installation.** `wen_client.WenLifecycle` installs the resulting sampler on every client process after PLATO's own `configure`. It logs a fingerprint of each client's partition, so the split can be compared across processes and runs.

## Cost (seed 4126, balanced CIFAR-10 labels)

- Mean total-variation distance between each client's realised label distribution and its own Dirichlet draw: **0.063**. The maximum is reported by `Partition.summary()`.
- The mean largest-class share per client stays above 0.25, against 0.10 for an IID split, so the split remains clearly non-IID.

## Checks (`wen_env/tests/test_wen_truth_and_selection.py`)

- The split is disjoint and exhaustive (50,000 unique indices), with exactly the locked sizes.
- It is deterministic for a given seed and changes with the seed.
- It stays non-IID.
- The rounding was found to need step 3 on some seeds, so the test covers 20 seeds.
