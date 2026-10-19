# Finding: PLATO's `noniid` sampler trains every client on a small, shared slice of CIFAR-10

**Date:** 16 Oct 2026. Found while asking why the Wen environment reaches 90% when Stage 6 stopped at about 83%.

**Status:**
- The mechanism is confirmed from the code.
- The coverage numbers come from exact replays of the sampler.
- The effect on accuracy is **inferred, not yet isolated**; section 6 describes the check that would isolate it.

## 1. Summary

Every experiment before the Wen environment used PLATO's `sampler = "noniid"`: `surgical/`, `separability/`, `separability_n100/`, `gpu_tiers_dataset/`, `sumo_close_loop/` and `sumo_open_loop/`.

That sampler gives every client the **same** torch random stream. Clients whose Dirichlet label weights resemble each other therefore pick largely the same images. In the Stage 6 setting (N = 20 × 2,500 images, α = 0.8, seed 4126), the fleet received 50,000 samples but only **8,453 distinct images, 16.9% of CIFAR-10**. Each class was represented by 552–1,487 distinct images.

The Wen environment uses its own disjoint sampler (ADR-0012, ADR-0015). Every one of the 50,000 images is used exactly once, 5,000 per class. This is the most likely reason the Wen runs reach about 90% while the earlier stages plateaued at 81–87%.

## 2. The code

`plato/samplers/dirichlet.py`:

```python
21    np.random.seed(self.random_seed * int(client_id))          # per client
      ...
      target_proportions = np.random.dirichlet(...)              # the client's label mix
      weights = target_proportions[target_list]                  # one weight per image
      ...
80    gen = torch.Generator()
81    gen.manual_seed(self.random_seed)                          # SAME seed for every client
      ...
85    WeightedRandomSampler(weights=self.sample_weights,
                            num_samples=partition_size,
88                          replacement=False, generator=gen)
```

Two separate properties combine here.

1. **Shared generator seed.** Line 81 seeds every client's generator with `random_seed` alone, not with `random_seed × client_id` as line 21 does. `WeightedRandomSampler` draws through `torch.multinomial`, so with the same uniform stream, two clients whose weight vectors are similar pick the same images. Weights here are per class, so clients with similar label mixes are similar enough.
2. **Independent draws.** Each client samples without replacement *within itself*, from the whole training set. Nothing stops two clients from holding the same image. So even with independent seeds, N × partition = 50,000 does not cover 50,000 distinct images.

## 3. Numbers

These are exact replays of the sampler above on the CIFAR-10 training labels, with seed 4126 (the `[data] random_seed` of the seed-4126 runs). The last column changes only line 81, giving each client its own seed, to separate the two properties.

| N | Partition | α | Samples handed out | **Distinct images (PLATO `noniid`)** | Distinct per class | Distinct if each client had its own generator seed |
|---|---|---|---|---|---|---|
| 20 | 2,500 | 0.1 | 50,000 | 17,254 (34.5%) | 961–2,451 | 32,927 (65.9%) |
| 20 | 2,500 | 0.8 | 50,000 | **8,453 (16.9%)** | 552–1,487 | 31,652 (63.3%) |
| 20 | 2,500 | 1.0 | 50,000 | 8,354 (16.7%) | 484–1,052 | 31,912 (63.8%) |
| 40 | 1,250 | 0.8 | 50,000 | 5,276 (10.6%) | 381–761 | 31,913 (63.8%) |
| 100 | 2,500 | 0.6 | 250,000 | 13,417 (26.8%) | 1,060–1,636 | 49,612 (99.2%) |
| 100 | 2,500 | 0.8 | 250,000 | 12,646 (25.3%) | 1,017–1,487 | 49,678 (99.4%) |
| 100 | 2,500 | 1.0 | 250,000 | 11,005 (22.0%) | 772–1,561 | 49,678 (99.4%) |

**Wen environment (`wen_env/wen_partition.py`):** N = 30, partition 1,667 / 1,666, α = 0.8. It hands out 50,000 samples, all of them distinct (100%), and 5,000 per class.

**Other measured effects (N = 20 × 2,500, α = 0.8).**
- Two clients share on average **1,227 of their 2,500 images** (maximum 2,068).
- The number of samples per class handed out, counting repeats, ranges from 3,023 to 7,080, so the fleet's effective class mix is unbalanced as well.

**The per-client label skew itself is similar in both samplers.**

| | PLATO `noniid` (N = 20) | Wen sampler (N = 30) |
|---|---|---|
| Mean largest-class share per client | 0.32 | 0.29 |
| Classes with at least 5% per client | 5.8 | 6.3 |

So the Wen runs are still clearly non-IID. What changed is how much distinct data the fleet holds, not how skewed each client is.

## 4. Which earlier results are affected

Every config in the folders below uses `sampler = "noniid"`, so every result they produced was trained on the reduced data shown above.

| Folder | Settings |
|---|---|
| `surgical/` | N = 20 × 2,500, α = 1.0 |
| `separability/` | N = 20 × 2,500, α = 0.1 and 1.0 |
| `separability_n100/` | N = 100 × 2,500, α = 0.6 and 0.8 |
| `gpu_tiers_dataset/` | N = 20 × 2,500 and N = 40 × 1,250, α = 0.8 |
| `sumo_close_loop/` | N = 20 and N = 100 × 2,500, α = 1.0; Stage 6 at α = 0.8 |
| `sumo_open_loop/` | trace-identity partitions use the same sampler class |

**What this does NOT invalidate.** Within one stage, every arm shared the same sampler and seed. Comparisons *between arms of one stage* therefore stand on equal footing.

**What it does affect.**
- Absolute accuracy levels and plateaus.
- Any comparison of accuracy *across* stages or fleet sizes. The fraction of distinct data changes with N, partition size, α and seed.
- Any statement about how hard the task is at a given N.

## 5. What the Wen environment does instead

`wen_env/wen_partition.py` is our own sampler, written in Prompt B because the locked decision required "50,000 images split with no image reused" (ADR-0012). ADR-0015 records the method:

1. Draw label proportions per client: Dir(0.8).
2. Scale the desired counts by Sinkhorn so that each client's total is its |S_i| and each class's total is exactly 5,000.
3. Round to integers while keeping both totals exact.
4. Deal out each class's shuffled images.

**What it costs.** The mean total-variation distance from each client's own Dirichlet draw is 0.063.

**How it is installed.** `wen_env/wen_client.py` replaces PLATO's sampler after PLATO builds it. The config still says `sampler = "noniid"`, because PLATO needs a valid value to start, but that sampler is never used for training.

## 6. Checks not yet done

1. **Isolate the effect on accuracy.** Run one Wen-environment cell (for example S1 random) with PLATO's `noniid` sampler at N = 20 × 2,500, and compare its plateau with the disjoint sampler. Then also compare at N = 30 with the same sampler, to separate data coverage from fleet size.
2. **Seed dependence.** The coverage numbers are for seed 4126. Other seeds change the Dirichlet draws and therefore the overlap.
3. **Report upstream.** Line 81 is a one-token bug: `manual_seed(self.random_seed * int(client_id))`, or a separate per-client generator. Even fixed, independent sampling would still reuse images (section 3, last column).
