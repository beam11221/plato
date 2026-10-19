# Accuracy dips — where do they come from? Log-only, no training runs.

No GPU needed; run alongside Stage 13B. Do not touch anything under `plato/`. Read run
folders only; do not modify them. **The model stays as it is (BatchNorm ResNet-18).**
This analysis does not propose changing it.

---

## Why

Late in training, single evaluations drop far below the level the model sits at, then
recover. Stage 13 (CIFAR-100), rounds 2,000–4,000: the level is ~46%, but readings fall to
19.5–31.7% in every run. The evaluation itself has a binomial SE of ~0.5 points on 10,000
test images. So these are real, transient drops in the global model. They are the most
likely source of the ±25% TTA noise that no estimator removed.

The candidate cause to test: **a single FedAsync merge pulling the global model toward one
client's skewed update.** The merge is `x ← (1 − w)·x + w·x_client`, with `w` set by the
FedAsync mixing parameter and the staleness discount `s(τ)`.

## Data

- Stage 13: all 6 runs (CIFAR-100, seed 4126)
- Stage 10 open, filtered, α = 1: `baseline + F` and `gpu + F`, seeds 4126 / 4127 / 4128
  (CIFAR-10)

## Step 0 — report what is available

1. The FedAsync mixing parameter and staleness function in the config, and the code that
   computes the applied weight `w`
2. Whether each merge's `w`, staleness, client identity, tier and partition are logged.
   If `w` is not logged, recompute it from staleness and the config formula, and say so
3. Whether any global-model checkpoints survive (most were deleted). If some do, list them

## Step 1 — define dips (fixed now)

For each run, compute the residual of each evaluation from a **centred 21-evaluation
rolling median**. A **dip** is an evaluation whose residual is below **−5 points
(CIFAR-100)** or **−3 points (CIFAR-10)**. Use only rounds ≥ 500. Report per run:
- number of dips and the share of evaluations that are dips
- depth (median, max)
- recovery: evaluations until the residual is back above −1 point

## Step 2 — what was merged just before each dip

For every evaluation interval (the 10 rounds before each evaluation), collect the merges in
it. Compare **dip intervals** with **all other intervals** (rounds ≥ 500) on:

| quantity | per interval |
|---|---|
| applied mixing weight `w` | max and sum |
| staleness `τ` | max and mean |
| partition skew of the merged clients | min label entropy, and max share of the largest class |
| tier | count of xavier / orin / thor merges |
| number of merges | count |

Report medians for the two groups and the rank-biserial correlation (or Mann–Whitney AUC)
for each quantity. Also: for the single merge with the largest `w` in each dip interval,
its partition, tier and `τ`.

## Step 3 — two direct checks

1. **Lag.** Is the dip evaluation the first one after a high-`w` merge? Report the
   distribution of rounds between the highest-`w` merge in the interval and the dip
2. **Arms.** Dip frequency per arm. If dips are about as frequent in every arm, they come
   from training dynamics, not from any selector

## Output — `DIP_analysis_report.md`

Step 0, the dip table, the dip-vs-other comparison, the two checks, and one paragraph:
which quantity (if any) separates dip intervals from the rest, and how strongly. **Report
only. Do not propose or test any fix in this task.**

## Do not

- launch any training run
- change the model, normalisation, FedAsync parameters or any config
- change the dip definition after seeing the results
