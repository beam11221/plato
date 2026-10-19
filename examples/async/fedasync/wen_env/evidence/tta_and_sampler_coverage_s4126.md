# S1 TTA so far, and why this environment reaches 90% (16 Oct 2026)

This is an interim note, not the Step 6 report.
- **Single seed (4126).** Every number below comes from one seed.
- **Where the gpu, cpu and ε = 0 numbers come from.** They are taken from the archived runs that hung when the disk filled (`runs/wen_env/_failed/*_hung_diskfull_0740`). Their logs up to the hang are intact: the replayed clock matches the online clock exactly.
- **Rounding to 90%.** At the user's request, gpu and cpu count as having reached 90% at their maximum accuracy (89.75% and 89.68%).
- **How TTA is read.** The time axis is the online simulated clock. Accuracy is evaluated every 10 rounds.

## TTA, S1

| Arm | TTA@80% | TTA@85% | TTA@89% | TTA@90% | Max accuracy (round) |
|---|---|---|---|---|---|
| random (ε = 1) | 1,419 s (r380) | 2,568 s (r680) | 6,202 s (r1650) | 6,645 s (r1770, ended run) | 90.29% (r1770) |
| cpu (ε = 0.2) | 1,535 s (r410) | 2,837 s (r750) | 4,878 s (r1300) | ≈ 6,866 s (rounded, r1840) | 89.68% (r1840) |
| gpu (ε = 0.2) | 1,158 s (r390) | 2,089 s (r680) | 5,586 s (r1720) | ≈ 6,326 s (rounded, r1950) | 89.75% (r1950) |
| gpu ε = 0 (ablation) | 1,117 s (r400) | 2,100 s (r720) | 4,880 s (r1590) | — | 89.32% (r1590) |

**Against random:**
- gpu is 18.4% faster to 80% and 18.6% faster to 85%.
- cpu is 8.2% slower to 80% and 10.5% slower to 85%.

**Caution on the high thresholds.** Above 89% the curves are near their plateau, and accuracy evaluated every 10 rounds moves by about ±0.5 points. A single noisy evaluation can therefore move TTA@89% and TTA@90% by hundreds of seconds. Example: cpu crosses 89% before gpu, yet reaches a lower maximum. The 80% and 85% crossings are the robust ones.

## Why 90% now, when Stage 6 stopped at about 83% after 4,000 rounds

**The comparison.** The earlier experiment is `sumo_close_loop` Stage 6 (N = 20, C = 4, partition 2,500, Dirichlet α = 0.8, seed 4126). It ended at about 81–83%, with a maximum of about 85%.

**The cause is the sampler, not the fleet size.**
- PLATO's `noniid` sampler (`plato/samplers/dirichlet.py`) draws each client's 2,500 images independently from the whole set.
- It also seeds its `torch.Generator` with the **same** `random_seed` for every client. Clients whose label weights are similar therefore draw largely the same images.

Replaying that sampler exactly on the CIFAR-10 labels (N = 20 × 2,500, seed 4126):

| | Stage 6 (PLATO `noniid`) | Wen environment (disjoint, ADR-0015) |
|---|---|---|
| Samples handed out | 50,000 | 50,000 |
| **Unique images** | **8,453 (16.9%)** | **50,000 (100%)** |
| Unique images per class | 552–1,487 | 5,000 each |
| Overlap between two clients | mean 1,227 of 2,500 (max 2,068) | 0 |
| Samples per class, with repeats | 3,023–7,080 | 5,000 each |

So Stage 6 trained on about 17% of CIFAR-10, with an uneven class mix. The Wen environment trains on all of it, with every class exactly balanced across the fleet. That alone is enough to explain a plateau about 7 points higher.

## Smaller differences, which are not tested here

- **Concurrency.** It is C = 3 now and was C = 4 then, so staleness is lower now.
- **Task durations.** They are now nearly uniform (about 9.5–12 s, dominated by upload). The anchored fleet ran Xavier at 15 s against Thor at 1 s, so Xavier updates arrived much staler.
- **Coverage.** Stage 6 used coverage gating. S1 here accepts every upload.

## Fleet size

A bigger per-client partition (2,500 against 1,667) would help *if* the data were distinct. Under the old sampler it was mostly the same images repeated across clients, so the larger partitions did not mean more information.

A controlled check would rerun this environment with PLATO's sampler and N = 20 × 2,500. It has not been run.
