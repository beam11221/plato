# GPU-Tier Staleness Sweep — Results Summary

Data-only handoff of the Stage 3 sweep results (accuracy, time-to-accuracy, staleness, dispatch share). **Training loss / valid loss are not included** — this covers accuracy-side metrics only, matching what the interactive results dashboard shows.

Source: `results/run_summary.csv` and `results/accuracy_curves_reconstructed.csv`, both regenerated after all 23 runs completed. Full per-round learning curves (400 evaluation points x 23 runs) live in `results/accuracy_curves_reconstructed.csv` and are not reproduced here row-by-row — this file has round/time-budget checkpoints instead.

**All times below are simulated (the reconstructed clock), not the wall-clock time of the machine that ran the sweep.** The reconstructed clock is a queue-model replay of each client's simulated per-tier sleep time (`sleep = T_tier(tier) * exp(eps)`); real host compute time (`processing_time`) is explicitly excluded from it. See `clock_reconstruction.py` and `stage2_evidence/clock_reconstruction.md`.

## Setup

| | |
|---|---|
| alpha | 0.8 |
| rounds | 4,000 |
| seed | 4126 |
| sigma (jitter) | 0.149166 (nominal CV 15%) |
| fleet mix | 50/35/15 (Xavier/Orin/Thor) |
| arms | baseline, gpu, cpu, observed, gpu_oracle |
| runs completed | 23 / 23 |
| gate spread (gpu vs rep2, stop >10%) | 0.00% |
| stop conditions triggered | none |

## 1. Determinism gate — gpu vs its own repeat (`_rep2`)

Byte-identical `gpu` config at C=4, run twice under different output tags. This is the campaign's noise floor at full length.

| metric | gpu | gpu (rep2) | difference |
|---|---|---|---|
| plateau | 82.55% | 82.55% | +0.00% |
| TTA @ 70% | 5.8m | 5.8m | +0.00% |
| TTA @ 75% | 8.1m | 8.1m | +0.00% |
| TTA @ 78% | 15.8m | 15.8m | +0.00% |
| TTA @ 80% | 22.9m | 22.9m | +0.00% |

## 2. Trends across the sweep

### 2.1 C sweep (N=20 fixed)

C = concurrency (staleness bound = 6.25xC). Prediction: selector advantage grows toward C~4, then flattens/falls by C=8.

| arm | C=1 (@75%) | C=2 (@75%) | C=4 (@75%) | C=8 (@75%) |
|---|---|---|---|---|
| Baseline (random) (reference) | TTA 29.3m, vs baseline +0.0%, plateau 81.81%, staleness 0.00 | TTA 21.2m, vs baseline +0.0%, plateau 82.37%, staleness 1.00 | TTA 10.4m, vs baseline +0.0%, plateau 82.97%, staleness 3.00 | TTA 5.6m, vs baseline +0.0%, plateau 82.12%, staleness 6.98 |
| GPU (spec mean) | TTA 36.6m, vs baseline -24.6%, plateau 81.53%, staleness 0.00 | TTA 16.1m, vs baseline +23.8%, plateau 82.03%, staleness 1.00 | TTA 8.1m, vs baseline +22.3%, plateau 82.55%, staleness 3.00 | TTA 4.4m, vs baseline +21.5%, plateau 82.64%, staleness 6.99 |
| CPU (wrong spec) | TTA 41.1m, vs baseline -39.9%, plateau 81.28%, staleness 0.00 | TTA 18.9m, vs baseline +10.9%, plateau 81.68%, staleness 1.00 | TTA 10.3m, vs baseline +0.7%, plateau 82.28%, staleness 3.00 | TTA 7.1m, vs baseline -25.3%, plateau 82.45%, staleness 6.99 |
| Observed (reactive) | TTA 27.6m, vs baseline +5.9%, plateau 81.86%, staleness 0.00 | TTA 18.7m, vs baseline +12.0%, plateau 82.65%, staleness 1.00 | TTA 7.4m, vs baseline +28.8%, plateau 82.31%, staleness 3.00 | TTA 4.5m, vs baseline +20.7%, plateau 82.26%, staleness 6.98 |

### 2.2 Fleet size (C=4 fixed)

Partition size scales so N x partition = 50,000. Prediction: if staleness is set by C alone, N shouldn't matter.

| arm | N=20 (@75%) | N=40 (@70%) |
|---|---|---|
| Baseline (random) (reference) | TTA 10.4m, vs baseline +0.0%, plateau 82.97%, staleness 3.00 | TTA 32.8m, vs baseline +0.0%, plateau 76.29%, staleness 2.99 |
| GPU (spec mean) | TTA 8.1m, vs baseline +22.3%, plateau 82.55%, staleness 3.00 | TTA 22.2m, vs baseline +32.3%, plateau 76.48%, staleness 3.00 |
| CPU (wrong spec) | TTA 10.3m, vs baseline +0.7%, plateau 82.28%, staleness 3.00 | TTA 30.8m, vs baseline +6.2%, plateau 76.29%, staleness 3.00 |
| Observed (reactive) | TTA 7.4m, vs baseline +28.8%, plateau 82.31%, staleness 3.00 | TTA 23.0m, vs baseline +29.9%, plateau 76.50%, staleness 3.00 |

### 2.3 gpu vs gpu_oracle (upper bound)

Deployable gpu arm vs non-deployable per-client-truth upper bound. A tie would mean jitter didn't make estimation matter.

| arm | C=4 (@75%) | C=8 (@75%) |
|---|---|---|
| GPU (spec mean) | TTA 8.1m, vs oracle -9.0%, plateau 82.55%, staleness 3.00 | TTA 4.4m, vs oracle +12.0%, plateau 82.64%, staleness 6.99 |
| GPU Oracle (per-client truth) (reference) | TTA 7.4m, vs oracle +0.0%, plateau 82.82%, staleness 2.99 | TTA 5.0m, vs oracle +0.0%, plateau 82.59%, staleness 6.98 |

## 3. Time-to-accuracy (all cells, all thresholds)

Each cell's *usable* threshold — the highest of 70/75/78/80% at least 5 points below that cell's lowest plateau — is marked with `*`.

| cell | arm | TTA@70% | TTA@75% | TTA@78% | TTA@80% |
|---|---|---|---|---|---|
| N=20, C=1 | Baseline (random) | 17.9m | 29.3m* | 53.1m | 1.93h |
| N=20, C=1 | GPU (spec mean) | 17.4m | 36.6m* | 55.3m | 1.77h |
| N=20, C=1 | CPU (wrong spec) | 20.5m | 41.1m* | 57.6m | 2.22h |
| N=20, C=1 | Observed (reactive) | 14.9m | 27.6m* | 50.2m | 1.48h |
| N=20, C=2 | Baseline (random) | 9.7m | 21.2m* | 35.5m | 56.4m |
| N=20, C=2 | GPU (spec mean) | 8.2m | 16.1m* | 29.6m | 52.6m |
| N=20, C=2 | CPU (wrong spec) | 13.0m | 18.9m* | 32.4m | 1.00h |
| N=20, C=2 | Observed (reactive) | 9.8m | 18.7m* | 34.6m | 58.3m |
| N=20, C=4 | Baseline (random) | 5.1m | 10.4m* | 18.0m | 27.0m |
| N=20, C=4 | GPU (spec mean) | 5.8m | 8.1m* | 15.8m | 22.9m |
| N=20, C=4 | CPU (wrong spec) | 7.5m | 10.3m* | 21.1m | 45.2m |
| N=20, C=4 | Observed (reactive) | 5.0m | 7.4m* | 13.7m | 25.0m |
| N=20, C=4 | GPU Oracle (per-client truth) | 6.2m | 7.4m* | 14.7m | 22.9m |
| N=20, C=8 | Baseline (random) | 3.7m | 5.6m* | 10.0m | 15.0m |
| N=20, C=8 | GPU (spec mean) | 2.4m | 4.4m* | 8.9m | 11.9m |
| N=20, C=8 | CPU (wrong spec) | 3.7m | 7.1m* | 11.4m | 18.7m |
| N=20, C=8 | Observed (reactive) | 2.9m | 4.5m* | 7.7m | 13.5m |
| N=20, C=8 | GPU Oracle (per-client truth) | 2.2m | 5.0m* | 7.5m | 13.1m |
| N=40, C=4 | Baseline (random) | 32.8m* | 1.09h | — | — |
| N=40, C=4 | GPU (spec mean) | 22.2m* | 1.11h | — | — |
| N=40, C=4 | CPU (wrong spec) | 30.8m* | 1.50h | — | — |
| N=40, C=4 | Observed (reactive) | 23.0m* | 57.1m | — | — |

## 4. Round-budget accuracy

Accuracy reached by a fixed round count — the axis that carries none of the reconstructed clock's noise.

| cell | arm | round 1000 | round 2000 | round 3000 | round 4000 |
|---|---|---|---|---|---|
| N=20, C=1 | Baseline (random) | 79.6% | 82.0% | 81.7% | 82.1% |
| N=20, C=1 | GPU (spec mean) | 81.0% | 79.6% | 82.8% | 82.0% |
| N=20, C=1 | CPU (wrong spec) | 81.5% | 81.2% | 81.2% | 81.8% |
| N=20, C=1 | Observed (reactive) | 81.0% | 82.0% | 82.8% | 82.6% |
| N=20, C=2 | Baseline (random) | 80.8% | 80.3% | 81.5% | 84.0% |
| N=20, C=2 | GPU (spec mean) | 78.6% | 82.0% | 83.2% | 83.5% |
| N=20, C=2 | CPU (wrong spec) | 77.0% | 80.8% | 78.9% | 80.2% |
| N=20, C=2 | Observed (reactive) | 81.2% | 79.5% | 77.9% | 83.0% |
| N=20, C=4 | Baseline (random) | 80.4% | 82.7% | 80.5% | 84.2% |
| N=20, C=4 | GPU (spec mean) | 79.9% | 81.3% | 82.2% | 80.2% |
| N=20, C=4 | CPU (wrong spec) | 79.1% | 80.6% | 81.8% | 82.8% |
| N=20, C=4 | Observed (reactive) | 80.5% | 82.0% | 82.5% | 82.8% |
| N=20, C=4 | GPU Oracle (per-client truth) | 81.7% | 81.9% | 82.9% | 82.7% |
| N=20, C=8 | Baseline (random) | 77.1% | 81.7% | 82.7% | 81.3% |
| N=20, C=8 | GPU (spec mean) | 79.0% | 82.0% | 82.4% | 82.7% |
| N=20, C=8 | CPU (wrong spec) | 77.6% | 81.3% | 78.6% | 82.8% |
| N=20, C=8 | Observed (reactive) | 77.8% | 81.6% | 81.8% | 82.2% |
| N=20, C=8 | GPU Oracle (per-client truth) | 79.1% | 82.1% | 81.8% | 78.0% |
| N=40, C=4 | Baseline (random) | 70.0% | 73.7% | 76.2% | 75.3% |
| N=40, C=4 | GPU (spec mean) | 67.9% | 74.8% | 76.0% | 76.3% |
| N=40, C=4 | CPU (wrong spec) | 72.3% | 72.8% | 75.8% | 76.3% |
| N=40, C=4 | Observed (reactive) | 70.5% | 68.7% | 75.3% | 76.8% |

## 5. Time-budget accuracy

Accuracy at 25/50/75/100% of each cell's own median run length (in that cell's own simulated hours — these differ by up to ~40x between C=1 and C=8).

| cell | total (h) | arm | @25% | @50% | @75% | @100% |
|---|---|---|---|---|---|---|
| N=20, C=1 | 8.80h | Baseline (random) | 79.6% | 81.4% | 83.1% | 82.1% |
| N=20, C=1 | 8.80h | GPU (spec mean) | 79.8% | 81.2% | 81.5% | 82.0% |
| N=20, C=1 | 8.80h | CPU (wrong spec) | 76.2% | 82.1% | 82.2% | 80.8% |
| N=20, C=1 | 8.80h | Observed (reactive) | 77.9% | 82.2% | 81.0% | 82.3% |
| N=20, C=2 | 4.27h | Baseline (random) | 79.5% | 77.5% | 82.7% | 84.0% |
| N=20, C=2 | 4.27h | GPU (spec mean) | 77.5% | 81.9% | 80.1% | 83.5% |
| N=20, C=2 | 4.27h | CPU (wrong spec) | 80.9% | 82.4% | 82.2% | 83.9% |
| N=20, C=2 | 4.27h | Observed (reactive) | 74.9% | 81.7% | 78.8% | 81.8% |
| N=20, C=4 | 1.94h | Baseline (random) | 79.8% | 82.5% | 82.1% | 84.2% |
| N=20, C=4 | 1.94h | GPU (spec mean) | 80.7% | 82.4% | 80.1% | 83.9% |
| N=20, C=4 | 1.94h | CPU (wrong spec) | 80.2% | 80.4% | 81.0% | 80.8% |
| N=20, C=4 | 1.94h | Observed (reactive) | 78.6% | 82.0% | 82.5% | 82.8% |
| N=20, C=4 | 1.94h | GPU Oracle (per-client truth) | 81.7% | 80.0% | 83.3% | 82.7% |
| N=20, C=8 | 48.2m | Baseline (random) | 79.1% | 81.8% | 81.1% | 81.3% |
| N=20, C=8 | 48.2m | GPU (spec mean) | 80.5% | 81.2% | 82.6% | 82.7% |
| N=20, C=8 | 48.2m | CPU (wrong spec) | 78.2% | 81.0% | 81.5% | 82.0% |
| N=20, C=8 | 48.2m | Observed (reactive) | 79.1% | 80.5% | 82.4% | 82.2% |
| N=20, C=8 | 48.2m | GPU Oracle (per-client truth) | 80.2% | 81.5% | 82.5% | 77.9% |
| N=40, C=4 | 2.04h | Baseline (random) | 70.0% | 73.2% | 76.0% | 76.0% |
| N=40, C=4 | 2.04h | GPU (spec mean) | 71.9% | 73.8% | 76.0% | 76.3% |
| N=40, C=4 | 2.04h | CPU (wrong spec) | 70.9% | 72.0% | 75.5% | 75.9% |
| N=40, C=4 | 2.04h | Observed (reactive) | 71.5% | 75.3% | 76.5% | 76.8% |

## 6. Dispatch share per tier (selector control check)

| cell | arm | dispatches | share xavier | share orin | share thor |
|---|---|---|---|---|---|
| N=20, C=1 | Baseline (random) | — | 49.1% | 35.5% | 15.4% |
| N=20, C=1 | GPU (spec mean) | — | 44.1% | 39.1% | 16.8% |
| N=20, C=1 | CPU (wrong spec) | — | 50.5% | 34.0% | 15.5% |
| N=20, C=1 | Observed (reactive) | — | 50.8% | 34.2% | 15.0% |
| N=20, C=2 | Baseline (random) | — | 47.1% | 37.2% | 15.7% |
| N=20, C=2 | GPU (spec mean) | — | 45.5% | 37.3% | 17.2% |
| N=20, C=2 | CPU (wrong spec) | — | 55.2% | 30.8% | 13.9% |
| N=20, C=2 | Observed (reactive) | — | 48.8% | 35.7% | 15.6% |
| N=20, C=4 | Baseline (random) | — | 41.8% | 40.4% | 17.8% |
| N=20, C=4 | GPU (spec mean) | — | 43.1% | 38.7% | 18.1% |
| N=20, C=4 | CPU (wrong spec) | — | 54.4% | 32.0% | 13.6% |
| N=20, C=4 | Observed (reactive) | — | 43.0% | 39.4% | 17.6% |
| N=20, C=4 | GPU Oracle (per-client truth) | — | 42.9% | 39.8% | 17.3% |
| N=20, C=8 | Baseline (random) | — | 31.7% | 47.3% | 20.9% |
| N=20, C=8 | GPU (spec mean) | — | 33.4% | 46.1% | 20.5% |
| N=20, C=8 | CPU (wrong spec) | — | 45.0% | 38.4% | 16.6% |
| N=20, C=8 | Observed (reactive) | — | 32.4% | 47.1% | 20.4% |
| N=20, C=8 | GPU Oracle (per-client truth) | — | 33.6% | 45.9% | 20.5% |
| N=40, C=4 | Baseline (random) | — | 45.7% | 38.2% | 16.1% |
| N=40, C=4 | GPU (spec mean) | — | 42.4% | 41.8% | 15.7% |
| N=40, C=4 | CPU (wrong spec) | — | 54.9% | 33.1% | 12.0% |
| N=40, C=4 | Observed (reactive) | — | 43.3% | 41.8% | 14.9% |

(dispatch counts omitted here; see `results/run_summary.csv` column `dispatches` if needed.)

## 7. Mean / max staleness per cell

| cell | arm | mean staleness (rounds) | max staleness (rounds) |
|---|---|---|---|
| N=20, C=1 | Baseline (random) | 0.000 | 0 |
| N=20, C=1 | GPU (spec mean) | 0.000 | 0 |
| N=20, C=1 | CPU (wrong spec) | 0.000 | 0 |
| N=20, C=1 | Observed (reactive) | 0.000 | 0 |
| N=20, C=2 | Baseline (random) | 1.000 | 11 |
| N=20, C=2 | GPU (spec mean) | 0.999 | 12 |
| N=20, C=2 | CPU (wrong spec) | 1.000 | 8 |
| N=20, C=2 | Observed (reactive) | 0.999 | 10 |
| N=20, C=4 | Baseline (random) | 2.997 | 16 |
| N=20, C=4 | GPU (spec mean) | 2.996 | 22 |
| N=20, C=4 | CPU (wrong spec) | 2.996 | 14 |
| N=20, C=4 | Observed (reactive) | 2.996 | 18 |
| N=20, C=4 | GPU Oracle (per-client truth) | 2.993 | 18 |
| N=20, C=8 | Baseline (random) | 6.977 | 33 |
| N=20, C=8 | GPU (spec mean) | 6.985 | 36 |
| N=20, C=8 | CPU (wrong spec) | 6.987 | 31 |
| N=20, C=8 | Observed (reactive) | 6.980 | 40 |
| N=20, C=8 | GPU Oracle (per-client truth) | 6.978 | 39 |
| N=40, C=4 | Baseline (random) | 2.994 | 17 |
| N=40, C=4 | GPU (spec mean) | 2.996 | 17 |
| N=40, C=4 | CPU (wrong spec) | 2.996 | 14 |
| N=40, C=4 | Observed (reactive) | 2.996 | 18 |
