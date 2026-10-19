# Stage 10: final report. The full method (Stage 0 filter + scoring) in the open population

Spec: `prompts/STAGE10_filter_open.md`. 48 runs: Wave A 30 (α = 1 filtered
arms plus `baseline / none`) and Wave B 18 (α = 10 filtered arms). Wave A
finished 2026-09-29 12:05 UTC; Wave B finished 2026-09-30 12:25 UTC. There
were no failures and no stop condition, apart from the `baseline / none`
drift that the user accepted as noise (§5 of `STAGE10_report.md`).

Seeds: physics 4126 pinned; run seeds 4126 / 4127 / 4128. Every comparison is
paired by seed and signed per seed. **Execution noise on a baseline cell is at
least ±3.5%**, and any comparison against a baseline has to clear it.

**Data:**

| file | contents |
|---|---|
| `stage10_analysis_waveB.json` | all 90 analysed runs, comparisons A–I |
| `stage10_all_runs.csv` | 90 rows × 49 columns, one row per run |
| `STAGE10A_report.md` | the Wave A tables and prediction scoring |
| `STAGE10_report.md` | process notes: audit, implementation, pre-flight, decisions |

---

## 1. Soundness (all 48 Stage 10 runs)

| check | limit | Wave A | Wave B |
|---|---|---|---|
| conservation-law error | ≤ 3% | ≤ 0.53% | ≤ 0.41% |
| plateau | ≥ 80.1% | 81.67 – 83.60% | 81.85 – 83.49% |
| `stage0_fallback` | ≤ 10% of rounds | ≤ 0.9% | ≤ 0.9% |
| median pool after Stage 0 | ≥ C = 4 | ≥ 5 | ≥ 5 |
| predictor tier == occupant tier | 100% | 100% | 100% |
| occupancy sequence == 7b / 8 / 9 | identical | identical across all 90 analysed runs (common prefix of 1,139 recycles) | |

---

## 2. T @ 4,000 (s)

| cell | 4126 | 4127 | 4128 |
|---|---|---|---|
| baseline / none, open | 7,138.9 | 6,996.5 | 7,185.7 |
| baseline / none, closed | 7,144.1 | 7,229.1 | 7,213.7 |
| baseline+F, open | 4,593.6 | 4,504.9 | 4,412.4 |
| baseline+F, closed | 4,445.3 | 4,592.4 | 4,501.5 |
| **gpu+F α = 1, open** | **3,893.4** | **3,907.3** | **3,999.2** |
| gpu+F α = 10, open | 4,873.6 | 4,940.3 | 4,809.9 |
| gpu+F α = 1, closed | 4,943.3 | 4,854.0 | 4,675.3 |
| gpu+F α = 10, closed | 4,836.8 | 4,928.5 | 4,759.0 |
| observed+F α = 1, open | 4,406.9 | 4,470.5 | 4,479.0 |
| observed+F α = 10, open | 4,869.7 | 4,850.2 | 4,833.8 |
| observed+F α = 1, closed | 4,917.9 | 4,992.4 | 4,703.5 |
| observed+F α = 10, closed | 4,866.9 | 4,969.2 | 4,566.9 |
| cpu+F α = 1, open | 5,318.9 | 5,327.6 | 5,352.3 |
| cpu+F α = 10, open | 5,675.4 | 5,558.5 | 5,663.9 |
| cpu+F α = 1, closed | 5,962.8 | 6,039.0 | 5,811.1 |
| cpu+F α = 10, closed | 5,902.8 | 6,037.9 | 5,817.3 |

---

## 3. Comparisons A–I (per seed 4126 / 4127 / 4128; negative = first arm faster)

| # | comparison | open | closed |
|---|---|---|---|
| **A / B** | **gpu+F vs baseline+F, α = 1** | **−15.2 / −13.3 / −9.4%: held** | +11.2 / +5.7 / +3.9%: gpu slower in every seed |
| C | gpu+F vs observed+F, α = 1 | −11.7 / −12.6 / −10.7%: held | +0.5 / −2.8 / −0.6%: undecided |
| D | baseline+F vs baseline / none | −35.7 / −35.6 / −38.6% | −37.8 / −36.5 / −37.6% |
| D | gpu+F vs baseline / none | −45.5 / −44.2 / −44.3% | −30.8 / −32.9 / −35.2% |
| D | observed+F vs baseline / none | −38.3 / −36.1 / −37.7% | −31.2 / −30.9 / −34.8% |
| D | cpu+F vs baseline / none | −25.5 / −23.9 / −25.5% | −16.5 / −16.5 / −19.4% |
| E | C1 in the open population (= D, baseline) | held, −36 to −39% | held, −37 to −38% |
| F | combined vs additive / multiplicative, gpu open | −44 to −45% vs −54 to −59% / −48 to −51%: **sub-additive** | combined −31 to −35% ≈ multiplicative |
| G | cpu+F vs gpu+F | +33.8 to +36.6% | +20.6 to +24.4% |
| H | baseline / none re-run vs 7b / 8 | −1.37 / **−3.51** / +1.06% | −1.60 / −1.50 / −0.71% |
| **I** | **gpu+F, α = 1 vs α = 10** | **−20.1 / −20.9 / −16.9%: α = 1 faster in every seed** | +2.2 / −1.5 / −1.8%: inside noise |
| I | observed+F, α = 1 vs α = 10 | −9.5 / −7.8 / −7.3%: held | +1.1 / +0.5 / +3.0%: α = 10 marginally faster, inside noise |
| I | cpu+F, α = 1 vs α = 10 | −6.3 / −4.2 / −5.5%: held | +1.0 / 0.0 / −0.1%: ≈ 0 |

**Wave B's key extra result:** with the filter on, **α = 10 turns the score
from a gain into a loss** in the open population. gpu+F at α = 10 is
**+6.1 / +9.7 / +9.0% slower than baseline+F** (observed+F α = 10:
+6.0 / +7.7 / +9.6%). The Polaris default prior makes every newcomer
outrank every known vehicle, so the score degenerates to "newest first". It
then feeds the filter's pool in arrival order instead of by spec, and loses
to random selection with the same filter.

---

## 4. The mechanism

**Waste and mix, open population (vs baseline / none; log-points):**

| run | total | mix | waste |
|---|---|---|---|
| baseline+F α = 1 | −44 to −49 | −31 to −34 | −13.0 to −14.4 |
| gpu+F α = 1 | −58 to −61 | −44.6 to −47.5 | −13.1 to −14.3 |
| gpu+F α = 10 | −35 to −40 | −22 to −26 | −13.1 to −14.4 |
| observed+F α = 1 | −45 to −48 | −32 to −34 | −12.7 to −13.9 |
| cpu+F α = 1 | −27 to −29 | −14 to −16 | −12.8 to −14.0 |
| cpu+F α = 10 | −23 to −24 | −9.9 | −12.8 to −14.0 |
| observed+F α = 10 | −37 to −40 | −22 to −25 | −13.3 to −14.5 |

- The **waste channel is the filter's**. It is the same ~13–14 points for
  every filtered arm, whatever the α. With the filter on, raw waste falls to
  0.05–0.8% in every run.
- Everything the score contributes comes through **mix**: which vehicles
  train. gpu+F α = 1 has the best mix of any arm by 11–14 points, and α = 10
  gives almost all of that back.

**Membership, gpu+F open:**

| | xavier / orin / thor ever dispatched | never dispatched (whole fleet) | Jain | median wait, xavier |
|---|---|---|---|---|
| α = 1 | 85.6–87.5 / 99.2–99.7 / 98.9–100% | 6.2–7.1% | 0.58–0.59 | 7.4–7.6 s |
| α = 10 | 99.1–99.3 / 100 / 100% | 0.4% | 0.60–0.61 | 2.6–2.7 s |

α = 10 dispatches practically every vehicle, and quickly. α = 1 lets 12–15%
of xavier vehicles go undispatched (0.8–1.0% at α = 10). **That exclusion is where the α = 1
advantage comes from, and it is the fairness price**: it is the same size as
without the filter (Stage 9: 12.3–14.7%).

**G_i xavier : thor, rounds 2,001–4,000, seen vehicles:**

| arm | regime | α = 1 + F | α = 10 + F |
|---|---|---|---|
| gpu | open | 2.25 / 1.72 / 2.03 | 2.38 / 1.74 / 1.84 |
| gpu | closed | 4.04 / 4.52 / 3.74 | 4.31 / 4.87 / 3.69 |

The filter-driven G_i divergence in the closed pool, about 4×, is there at
either α. α does not change the antagonism; persistent identity drives it.

**observed+F's reactive `T̂`:**
- A never-seen vehicle is filtered with the fleet running mean of delivered
  durations: **4.3–4.7 s at α = 1** (4.6–5.2 s at α = 10), not the ~8 s assumed in the prediction. The mean
  is taken over delivered reports, which are mostly fast vehicles.
- In the open population, ~70% of its xavier dispatches were admitted on that
  fleet mean (at either α; ~1% in closed).
- Admitted-xavier waste is 2.9–3.6% for observed+F against 1.1–1.7% for
  gpu+F. Spec-based filtering rejects doomed xavier newcomers; the fleet
  average admits some of them.

**Filter accuracy** (gpu+F α = 1, open, 4126): of the rejected candidates,
11% of xavier would in fact have completed in range, against 27% of orin and
43% of thor. The conservative v_max bound costs the fast tiers far more than
the slow one.

---

## 5. Predictions (written before the runs)

| # | prediction | verdict |
|---|---|---|
| 1 | C1 replicates in the open population, 30–42%, waste channel ≈ −12 to −13 | **held**: −35.6 to −38.6% open, −36.5 to −37.8% closed; waste −12.8 to −14.5 |
| **2** | **gpu+F beats baseline+F in the open population in all seeds, by less than Stage 9's −20.6% (sub-additive)** | **held**: −15.2 / −13.3 / −9.4%, each beyond ±3.5% noise; sub-additive against both models |
| 3 | Closed: gpu+F slower than baseline+F in at least 2 of 3 seeds | **held, 3 / 3**: +11.2 / +5.7 / +3.9% |
| 4 | G_i ratio of filtered gpu larger than its no-filter partner in closed, close in open | **held in direction**: +37–61% closed; +11–24% open, which is not really "close" |
| 5 | C2 with the filter: gpu+F beats observed+F in open in all seeds, and observed+F wastes more | **held**: −10.7 to −12.6%; observed+F waste higher in every seed (small) |
| 6 | cpu+F slowest filtered arm; waste channel intact; mix channel ≈ 0 or positive | **partly**: slowest and waste intact both held. The **mix channel is negative** (open −14 to −16, closed −5 to −8 points): cpu's filter still improves the mix |
| **7** | **α = 1 beats α = 10 with the filter in open, all seeds; α = 10's waste advantage disappears (within 1 point)** | **held**: −20.1 / −20.9 / −16.9%. Waste difference α = 1 − α = 10 is −0.05 to +0.15 points for gpu+F (≤ 0.6 for observed+F) |
| 8 | baseline / none re-runs within 2% of 7b / 8 in every seed | **not held**: s4127 open −3.51%, accepted as execution noise. The other five re-runs are within 1.6% |
| 9 | Fairness price: xavier never-dispatched under gpu+F (open) exceeds Stage 9's 12–15% | **not held**: 12.5–14.4%, the same as without the filter. The filter adds no exclusion |

---

## 6. What Stage 10 establishes

1. **The filter carries most of the gain, in both populations.** It takes
   ~36–39% off wall time on its own, through a ~13-point waste channel and a
   large mix channel (it stops dispatching slow vehicles that cannot finish
   in range).
2. **In the open population, the score adds a further 9–15% on top of the
   filter**, beyond noise in every seed. It is the method's main claim, and
   it holds with the full method. The two halves overlap, since both remove
   xavier dispatches, so the combination is sub-additive: −44 to −45%
   against −48 to −51% multiplicative.
3. **The score's value depends entirely on the neutral prior.** With the
   Polaris default α = 10, the same filtered method is 6–10% **slower than
   random selection plus the filter**. α = 10's "newest first" ordering keeps
   the spec from deciding who trains.
4. **Under persistent identity (closed), the score is harmful with the
   filter at either α.** gpu+F is 4–11% slower than baseline+F in every seed.
   The filter-driven G_i divergence (xavier : thor ≈ 4×) makes the score
   chase the vehicles the filter excludes. **In a closed pool, filter alone
   is the best configuration.** Stage 11 measures where between the two
   populations the sign turns.
5. **The price** is tier-selective exclusion: 12.5–14.4% of xavier vehicles
   are never dispatched under gpu+F α = 1 (open), against 0.8–1.0% under
   α = 10. Overall fairness is otherwise no worse than random selection with
   the filter.

**Caveats:**
- Three run seeds on one trace (physics seed 4126).
- `baseline / none` noise is at least ±3.5%.
- TTA-80 differences between gpu+F α = 1 and α = 10 are small
  (0.31–0.33 h), because the model converges about equally fast. The gain is
  in wall-clock time per round, not in rounds to accuracy.
