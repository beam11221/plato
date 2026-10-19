# Stage 13: CIFAR-100 pilot, analysis (seed 4126)

> **One seed, not a result.** Six runs: the five methods plus one replicate.
> Raw tables: `STAGE13_pilot_report.md` (auto-generated). Series:
> `series_<#>_*.csv`.

The CIFAR-10 column is the identical Stage 10 open config at the same seed
(`stage10nf_s4126` / `stage10f1_s4126`). Only the dataset and output layer
differ.

---

## 1. Soundness

- All six runs reached round 4,000.
- Conservation error: −0.05% to −0.45%.
- Fallback: ≤ 0.6%.
- Median pool after Stage 0: 5–8.
- Predictor tier agreement: 100%.

There were no failures. The plateau rule was disabled, as the spec asks.

---

## 2. The selection mechanics do not depend on the dataset

| # | run | T @ 4,000, C100 | T @ 4,000, C10 | waste C100 / C10 | share x / o / t, C100 | share x / o / t, C10 |
|---|---|---|---|---|---|---|
| 1 | baseline / none | 7,139 | 7,139 | 12.78 / 12.58% | 34.4 / 43.6 / 22.1 | 34.5 / 43.2 / 22.3 |
| 2 | baseline + F | 4,650 | 4,594 | 0.47 / 0.45% | 22.5 / 49.1 / 28.4 | 22.2 / 50.1 / 27.8 |
| 3 | gpu + F | 3,908 | 3,893 | 0.30 / 0.27% | 17.3 / 52.7 / 30.0 | 17.0 / 52.5 / 30.4 |
| 4 | observed + F | 4,422 | 4,407 | 0.62 / 0.64% | 21.0 / 51.0 / 28.0 | 21.0 / 51.7 / 27.3 |
| 5 | cpu + F | 5,377 | 5,319 | 0.60 / 0.64% | 28.2 / 43.9 / 27.9 | 27.7 / 43.9 / 28.4 |
| 6 | baseline + F, replicate | 4,410 | (none) | 0.35% | 20.9 / 49.9 / 29.1 | (none) |

- Wall-clock per round, waste and tier mix match CIFAR-10 to within about 1%
  for every arm.
- This is expected. They are set by the trace, sleeps, filter and selection
  rule, none of which changed.
- gpu+F's score also uses G_i, which does depend on the gradients. It still
  lands on the same tier mix (17 / 53 / 30).
- So **the filter's and the score's effect on time per round carries over to
  CIFAR-100 unchanged**:
  - filter alone: −35% T@4000;
  - gpu+F vs baseline+F: −16.0% (−11.4% against the replicate). CIFAR-10
    was −15.2%.
  - cpu+F is the slowest filtered arm, at +38% vs gpu+F.

---

## 3. What changes: accuracy

| # | run | plateau | best | acc @ 1k | noise SD (pts) | AUC 30 min | AUC 60 min | plateau C10 | AUC 30 min C10 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline / none | 46.60 | 47.64 | 43.16 | 3.91 | 36.23 | 39.54 | 82.70 | 73.57 |
| 2 | baseline + F | 47.74 | 48.54 | 39.29 | 3.69 | 38.65 | 41.60 | 82.13 | 74.64 |
| 3 | gpu + F | 46.44 | 47.72 | 39.71 | 3.58 | **38.99** | **41.81** | 82.47 | 75.07 |
| 4 | observed + F | 45.89 | 47.42 | 42.68 | 3.68 | 38.59 | 41.41 | 82.16 | 74.89 |
| 5 | cpu + F | 44.80 | 47.02 | 41.99 | 3.84 | 36.56 | 39.71 | 83.06 | 73.72 |
| 6 | baseline + F, replicate | 46.56 | 47.85 | 43.08 | 3.71 | 38.25 | 41.37 | — | — |

**How these columns are computed:**
- **Noise SD:** the scatter of single evaluations around an 101-evaluation
  rolling median (`analyze_stage7.curve_stats`).
- **AUC:** mean accuracy over the first 30 or 60 simulated minutes (the E5
  metric of the TTA study). Higher is better.

**What the table shows:**
- **The plateau is about 45–48%**, against about 82–83% on CIFAR-10. Most of
  it is reached by round ~1,000; later rounds add only 3–8 points. That is
  why no run reached TTA targets of 50% or more.
- **Single readings are relatively noisier.** The SD is about 3.7 points on
  a ~46% level (≈ 8% relative), against about 4.5 points on ~82% (≈ 5%) for
  CIFAR-10. One evaluation can be far off: cpu+F reads 38.0% at round 4,000
  while its plateau is 44.8%. This is why plateau differences of 1–3 points
  between arms cannot be interpreted at one seed. The replicate pair (#6 vs
  #2) is itself 1.2 points apart.

---

## 4. The pairs (one seed)

**How the verdict is read:** the replicate (#6 vs #2) is the noise yardstick.
A difference smaller than the replicate's own is "inside noise".

| pair | T @ 4,000 | TTA 30% | TTA 40% | AUC 30 min (pts) | AUC 60 min (pts) | reading |
|---|---|---|---|---|---|---|
| #6 vs #2, replicate | **−5.2%** | +12.0% | +29.0% | −0.40 | −0.23 | the noise yardstick |
| #2 vs #1, filter | −34.9% | −32.7% | −39.7% | **+2.42** | **+2.06** | beyond noise on every metric |
| #3 vs #2, gpu vs baseline (+F) | −16.0% | −7.5% | −4.1% | +0.34 | +0.21 | faster per round; TTA and AUC inside noise |
| #3 vs #6, gpu vs replicate | −11.4% | −18.8% | −25.6% | +0.74 | +0.44 | same direction, about the size of the noise |
| #4 vs #2, observed vs baseline (+F) | −4.9% | −7.2% | −0.2% | −0.06 | −0.19 | inside noise |
| #5 vs #3, cpu vs gpu (+F) | +37.6% | +37.1% | +55.0% | **−2.43** | **−2.10** | beyond noise: cpu is the slowest |

**Readings:**
1. **The filter result replicates on CIFAR-100.**
   - It saves 33–40% of the time to 30% / 40% accuracy.
   - It adds +2.4 points of mean accuracy over the first 30 minutes. On
     CIFAR-10 the same comparison is +1.1 points.
   - Waste falls from 12.8% to 0.5%.
2. **gpu+F is still 11–16% faster per round than baseline+F** (T@4000), as
   on CIFAR-10.
   - On accuracy-over-time it is ahead of both baseline+F runs: AUC30 +0.34
     and +0.74 points, TTA 40% −4% and −26%.
   - Every one of these is within the replicate's own spread, so one seed
     cannot separate it.
3. **cpu+F is clearly the worst filtered arm**, as on CIFAR-10.
4. **observed+F is indistinguishable from baseline+F.**
5. **The replicate's T@4000 differs by 5.2%.** That is above the ±3.5%
   execution-noise band accepted from the CIFAR-10 re-runs. The configs are
   identical. The difference comes from PLATO's host-time component of the
   simulated clock (Stage 9 finding), so the noise band on T@4000 should be
   taken as **at least ±5%** here.

---

## 5. What it means for the next step

- The method's **time-per-round advantage transfers to CIFAR-100 intact**,
  because it does not depend on the data.
- Whether that becomes an **accuracy-over-time advantage** cannot be decided
  from one seed:
  - the per-run accuracy noise is larger relative to the gains;
  - the curve flattens early, at about 46%, which compresses all
    time-to-target differences;
  - TTA targets above 40% are unreachable at 4,000 rounds.
- Stage 13B (10,000 rounds, running now) shows whether the plateau rises
  with longer training. More seeds would be needed to resolve the
  gpu+F vs baseline+F accuracy difference. That is not part of this spec.

**Caveat:** AUC, plateau-noise and the CIFAR-10 side-by-side go beyond the
spec's "standard metrics". They were added at the user's request for an
analysis (2026-10-04) and use only existing logs. No run was added.
