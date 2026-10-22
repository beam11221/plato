# S1: time and energy to 80 / 85 / 90% — random, Ours (gpu), Wen et al., Polaris

Generated 22 Oct 2026 by `wen_env/tools/s1_tta_energy.py`.

- **TTA:** the first evaluated round (every 10 rounds) at or above the threshold, on the online clock.
- **Energy to X%:** ground-truth E_train + E_up summed over the dispatches that finished by that time. This is the same definition as "energy to 80%" in `wen_main_report.py`.
- **Budget:** 4,000 rounds. "Not reached" means not reached within that budget.
- **Reading B:** S1 gpu at seed 4126 counts as reaching 90% at its maximum, 0.8975.
- **Missing seeds:** `wen_fresh` and `wen_fresh_gpu` (Wen et al. + GPU, Study 2b) have seed 4126 only.
- **Spread:** mean ± sd over seeds.

| method | seeds | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | energy to 80% (kJ) | energy to 85% (kJ) | energy to 90% (kJ) |
|---|---|---|---|---|---|---|---|
| random | 3 | 1,733 ± 345 | 3,102 ± 500 | 7,108 ± 573 | 30.2 ± 6.7 | 54.4 ± 9.8 | 125.8 ± 12.8 |
| Ours (gpu) | 3 | 1,315 ± 207 | 2,429 ± 295 | 7,114 ± 696 | 22.2 ± 3.3 | 45.0 ± 5.6 | 140.1 ± 13.0 |
| Wen et al. (wen) | 3 | 2,772 ± 150 | 9,734 ± 1,147 | 0/3 reached | 38.2 ± 12.1 | 132.4 ± 39.3 | 0/3 reached |
| Wen et al. (wen_fresh) | 1 | 1,967 | 7,263 | 0/1 reached | 28.6 | 94.7 | 0/1 reached |
| Wen et al. + GPU (wen_fresh_gpu) | 1 | 1,675 | 5,046 | 0/1 reached | 16.0 | 47.9 | 0/1 reached |
| Polaris | 3 | 1,824 ± 216 | 2,784 ± 289 | 8,063 ± 643 | 32.3 ± 4.2 | 49.7 ± 5.8 | 145.3 ± 11.0 |

| method | seed | TTA@80% (s) | TTA@85% (s) | TTA@90% (s) | energy to 80% (kJ) | energy to 85% (kJ) | energy to 90% (kJ) | max accuracy |
|---|---|---|---|---|---|---|---|---|
| random | 4126 | 1,419 | 2,568 | 6,645 | 23.6 | 43.6 | 114.7 | 0.9029 |
| random | 4127 | 1,678 | 3,177 | 7,749 | 30.1 | 57.0 | 139.8 | 0.9003 |
| random | 4128 | 2,102 | 3,559 | 6,931 | 36.9 | 62.7 | 122.9 | 0.9025 |
| Ours (gpu) | 4126 | 1,158 | 2,089 | 6,326 | 19.4 | 38.6 | 125.7 | 0.8975 |
| Ours (gpu) | 4127 | 1,237 | 2,616 | 7,644 | 21.2 | 49.2 | 151.0 | 0.9007 |
| Ours (gpu) | 4128 | 1,550 | 2,581 | 7,373 | 25.9 | 47.2 | 143.6 | 0.9006 |
| Wen et al. (wen) | 4126 | 2,625 | 11,049 | not reached | 33.0 | 138.4 | not reached | 0.8513 |
| Wen et al. (wen) | 4127 | 2,925 | 8,938 | not reached | 29.7 | 90.5 | not reached | 0.8617 |
| Wen et al. (wen) | 4128 | 2,767 | 9,214 | not reached | 52.0 | 168.4 | not reached | 0.8719 |
| Wen et al. (wen_fresh) | 4126 | 1,967 | 7,263 | not reached | 28.6 | 94.7 | not reached | 0.8683 |
| Wen et al. + GPU (wen_fresh_gpu) | 4126 | 1,675 | 5,046 | not reached | 16.0 | 47.9 | not reached | 0.8661 |
| Polaris | 4126 | 1,580 | 2,452 | 7,646 | 27.8 | 43.1 | 137.4 | 0.9043 |
| Polaris | 4127 | 1,903 | 2,924 | 8,803 | 32.9 | 51.9 | 157.8 | 0.9031 |
| Polaris | 4128 | 1,988 | 2,976 | 7,739 | 36.1 | 54.1 | 140.8 | 0.9032 |
