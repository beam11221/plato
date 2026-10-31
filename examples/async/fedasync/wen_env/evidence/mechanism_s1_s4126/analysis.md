# Why each method picks the vehicles it picks — S1, M0, seed 4126 (final)

Status 31 Oct 2026, 21:15 UTC. All four arms are final. Numbers come from
`mechanism_s1_s4126.md`; this file interprets them. It supersedes
`analysis_interim.md` (written when Wen was at round 755), which is kept unchanged.

## Outcome

| arm | TTA@80 % | TTA@85 % | TTA@90 % | max accuracy | s/round |
|---|---|---|---|---|---|
| **Ours** | **1,326 s** (r440) | **2,100 s** | **6,065 s** (r1,870) | 0.9002 | **3.24** |
| Random | 1,379 s (r370) | 2,639 s | 6,645 s (r1,770) | 0.9077 | 3.75 |
| Polaris | 1,612 s (r430) | 2,663 s | 6,609 s (r1,750) | 0.9018 | 3.78 |
| Wen et al. | 2,024 s (r580) | 6,722 s (r1,950) | not reached in 4,000 rounds | 0.8639 | 3.46 |

These runs replace the October seed-4126 runs. The overview agrees with them
(`logs/promptM/partA_compare.txt`, `partA_compare_wen.txt`). For example, the October Wen run had
TTA@80 % 1,967 s, TTA@85 % 7,263 s and max accuracy 0.8683, and s/round agrees within 0.3 % for
every arm.

Common axes: every arm logs, for every candidate at every decision, what Ours' scorer
*would* compute — predicted staleness τ̂ (rounds; GPU training estimate + upload time at
the predicted finish position) and data value G (norm of the vehicle's last update).
So "picked low τ̂" means the same thing in every arm.

## Headline (cross-arm table, plot 14)

| arm | τ̂ picked vs all candidates | G picked vs all candidates | mean upload distance from RSU |
|---|---|---|---|
| Ours | **2.61 vs 3.04** (lower) | **1.59 vs 1.36** (higher) | **279 m** |
| Polaris | 3.01 vs 3.02 (same) | 1.58 vs 1.56 (same) | 489 m |
| Random | 2.99 vs 3.02 (same) | 1.57 vs 1.58 (same) | 492 m |
| Wen et al. | 2.77 vs 3.04 | **0.74 vs 2.33 (far lower)** | 460 m |

- **Ours** is the only arm that picks *both* faster-to-return and more informative vehicles
  than the pool offers.
- **Polaris** and **Random** pick the average candidate on both axes: they behave the same.
- **Wen** picks vehicles with *less* information than the pool (lower G): the vehicles whose
  data the model already fits.

## Ours: what decides a pick

1. **Both terms matter, and their weight shifts during the run** (plot 2, plot 4).
   Of the exploit picks, the share that is the min-τ̂ candidate falls from 0.26 (rounds
   1–500) to 0.09 (rounds 1,500+), and the share that is the max-G candidate rises from
   0.26 to 0.61. Why: the spread of the τ̂ factor across one decision's candidates stays
   constant (√(1+τ̂) max/min ≈ 1.31 throughout), but the spread of G grows (max/min G
   1.42 → 1.70). So in V = G/√(1+τ̂), time leads early and data value leads later.
2. **This is the self-correcting rotation behind the early-only tier effect.** Early, Ours
   prefers fast, near-RSU vehicles: Xavier gets 31 % of picks in rounds 1–500 (Random 43 %).
   The Xavier vehicles it skipped keep larger updates, so their G rises, and G pulls them
   back. Xavier's share climbs to 47 % (rounds 501–1,000) and then 54 % (rounds 1,001+),
   about its fleet share of 50 %. Ours gets the speed gain early without starving half the
   data.
3. **Neither term alone reproduces Ours** (plot 3, static counterfactual). Picking by τ̂
   only would change 81 % of exploit picks and raise the dispatch Gini from 0.10 to 0.42
   (a few fast vehicles monopolise). Picking by G only would change 52 % and give a Gini
   of 0.30. The product V keeps participation broad (Gini 0.10), close to Random's.
4. **Where the time goes:** Ours' picks upload 279 m from the RSU on average (Random 492 m),
   so true upload time per task is 8.3 s against 9.9 s. Training time is not lower (1.38 s
   against 1.39 s for Random). The time gain comes mainly from the predicted upload
   position, with tier knowledge acting early. This matches the earlier decomposition
   (position ≈ 11 points of s/round, GPU tier ≈ 3 points).
5. ε-greedy: 391 of the 1,872 picks (21 %) are explored. In plot 1 they sit in the grey
   cloud, below the iso-V lines.

## Polaris: why it behaves like Random

- q is nearly uniform: q × |pool| has median 1.00 and a 5–95 % range of 0.81–1.19. The
  effective pool is 99 % of the real pool (plot 7).
- In problem P4, q_i ∝ √(f1/(f2+μ)). With p_i equal for every vehicle and G within
  about ×1.5, the square root flattens the differences again.
- Its τ is observed staleness (in merges), which is almost the same for every vehicle at
  C = 3. It does not see position or tier, so it cannot pick near-RSU vehicles: mean
  upload distance is 489 m, the same as Random.
- Per vehicle, the number of dispatches does not follow G, τ or τ̂ (Spearman |ρ| ≤ 0.23,
  p ≥ 0.2).
- **Result:** Polaris's s/round is 3.78 against Random's 3.75, so it has no time
  advantage. Its rounds to 80 % are about the same as Random's.

## Wen et al.: why it is the worst (whole run, 4,000 rounds)

1. **F3 (own-data loss) and F4 (label KL) decide; F1 (time) barely matters** (plot 10).
   Of the picks, 50 % are the min-F3 candidate, 40 % are min-F4, 11 % are min-F2 and only 2 %
   are min-F1. In the static counterfactual (plot 13):
   - Dropping F1 or F2 changes 14 % of picks each.
   - Dropping F3 changes 31 %, and dropping F4 changes 46 %.
   - Using F1 + F2 alone would change 89 % of picks and lower the Gini from 0.82 to 0.29.
   - F3 + F4 alone gives a Gini of 0.89, close to what Wen does.
2. **F3 feeds back on itself** (plot 11). The vehicles Wen picks are the ones the global
   model gets trained on, so their loss falls fastest. The top-5 vs other-25 F3 ratio
   grows from 1.0 at round 1 to 2.6 at round 400, 38 at round 2,000 and 85 at round
   4,000. A low F3 gets picked again, so the same vehicles win again.
   - The top-5 vehicles (16, 19, 22, 5, 21) hold 3,699 of 4,002 dispatches (92 %).
   - Their F4 ranks are 2, 1, 4, 3 and 5: the five lowest-KL vehicles start the loop,
     because F3 is equal for everyone at round 0.
   - Only 25 of 30 vehicles are ever picked, and the dispatch Gini is 0.82 (Ours 0.10).
3. **This means Wen picks the least informative vehicles, and it gets worse over time.**
   A low own-data loss means the model already fits that vehicle's data, so its update is
   small. On the shared axes, Wen's picks have G 0.74 against 2.33 for its candidates.
   The gap widens during the run: 1.56 vs 2.07 at round 755, 0.98 vs 2.22 at round
   1,919, and 0.74 vs 2.33 over the whole run. Every other arm picks at or above the
   pool's average G. Wen spends its time on updates that carry the least new information,
   so it plateaus at 0.864 and never reaches 85 % before round 1,950.
4. **Thor is never picked.** E_max cuts 60 % of Thor's candidate rows; Xavier and Orin
   lose only 12–13 % (to T_max). Thor gets 0 % of picks, Xavier 20 % and Orin 80 %.
   Prompt N Part 1 (`evidence/wen_thor_replay_v1.md`) shows E_max is not the main cause:
   - With E_max removed, Thor still gets ≤ 0.3 % of picks.
   - The cause is Wen's own CPU energy model (Thor F2 ≈ +36 %) together with min–max
     normalisation inside the Pareto set (best Thor nF2 ≈ 0.97 against ≈ 0.05).
5. **Its time estimate is wrong, so position does not help it.** Wen's CPU T̂_com is
   about 20–23 s against a true 0.2–2.8 s, so its predicted upload position is hundreds of
   metres off. Mean upload distance is 460 m (Ours 279 m, Random 492 m). It does get a lower
   s/round than Random (3.46 vs 3.75), because its favourite vehicles happen to train
   fast (true T_train 0.79 s against 1.39 s for Random). That speed buys no accuracy,
   because their updates carry little information.

## Answers in one line each

- **Why Ours is best:** it is the only rule that combines a correct time prediction
  (it picks vehicles that will upload near the RSU) with a data-value signal that grows
  as training goes on and rotates the fleet back. So it is fast early and still covers
  all the data.
- **Why Polaris ≈ Random:** in this regime its P4 sampling is almost uniform, and it
  knows nothing about position or speed.
- **Why Wen is worst:** F3 + F4 lock it onto five low-KL, low-loss vehicles, which then hold
  92 % of dispatches. Their updates carry the least new information (G 0.74 vs the pool's
  2.33). Its CPU-based energy model and normalisation remove Thor, and its CPU time model
  misplaces the upload position.

## Caveats

- Single seed (4126).
- Static counterfactuals re-decide logged decisions without feedback. They show which term
  moves single picks, not how a full run under that rule would end.
- This seed-4126 set replaces the October seed-4126 runs (author's decision). Bit-wise
  replay is not possible on this pod; the overview agrees (`logs/promptM/partA_compare.txt`).
