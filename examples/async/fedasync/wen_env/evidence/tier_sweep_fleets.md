# Tier-mix fleets (Prompt M B0.1)

Each mix is M0 with only the tier labels re-blocked (contiguous Xavier → Orin → Thor). Speeds come from the SUMO bundle by id; partitions from `wen_partition.build(targets, 4126)`, which does not depend on the fleet.

| mix | X / O / T | fleet dir | fleet sha256 (12) | mean speed X / O / T (m/s) | eps_i = M0 (30 ids) | partition = M0 (30 ids) |
|---|---|---|---|---|---|---|
| M0 | 15 / 10 / 5 | `n30_x15-o10-t5_phys_p1667_j0p149_s4126` | 45957513ee7b | 26.77 / 24.86 / 28.13 | yes | yes |
| M1 | 10 / 10 / 10 | `n30_mixM1_x10-o10-t10_phys_p1667_j0p149_s4126` | d0832abde1b9 | 25.68 / 26.27 / 27.13 | yes | yes |
| M2 | 5 / 10 / 15 | `n30_mixM2_x5-o10-t15_phys_p1667_j0p149_s4126` | 6c0c09834a0e | 22.99 / 28.67 / 25.95 | yes | yes |
| M3 | 2 / 8 / 20 | `n30_mixM3_x2-o8-t20_phys_p1667_j0p149_s4126` | ba0756b88725 | 24.38 / 26.00 / 26.70 | yes | yes |
| T1 | 15 / 0 / 15 | `n30_mixT1_x15-o0-t15_phys_p1667_j0p149_s4126` | 75bfe9632802 | 26.77 / — / 25.95 | yes | yes |
| T2 | 5 / 0 / 25 | `n30_mixT2_x5-o0-t25_phys_p1667_j0p149_s4126` | 7ae687e581a9 | 22.99 / — / 27.04 | yes | yes |
| T3 | 15 / 15 / 0 | `n30_mixT3_x15-o15-t0_phys_p1667_j0p149_s4126` | b382277bc607 | 26.77 / 25.95 / — | yes | yes |

## Tier per client id (X = Xavier, O = Orin, T = Thor)

| mix | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | X | X | X | X | X | X | X | X | X | X | X | X | X | X | X | O | O | O | O | O | O | O | O | O | O | T | T | T | T | T |
| M1 | X | X | X | X | X | X | X | X | X | X | O | O | O | O | O | O | O | O | O | O | T | T | T | T | T | T | T | T | T | T |
| M2 | X | X | X | X | X | O | O | O | O | O | O | O | O | O | O | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T |
| M3 | X | X | O | O | O | O | O | O | O | O | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T |
| T1 | X | X | X | X | X | X | X | X | X | X | X | X | X | X | X | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T |
| T2 | X | X | X | X | X | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T |
| T3 | X | X | X | X | X | X | X | X | X | X | X | X | X | X | X | O | O | O | O | O | O | O | O | O | O | O | O | O | O | O |

## Partition fingerprint per id (sha256 of the sorted indices, 12 hex)

| id | size | fingerprint |
|---|---|---|
| 1 | 1667 | 7e0bcf996752 |
| 2 | 1667 | 1f898434819b |
| 3 | 1667 | 4e00ad56697d |
| 4 | 1667 | 5acef30cf49a |
| 5 | 1667 | fdcb2f99be5a |
| 6 | 1667 | 7fde6605f8ae |
| 7 | 1667 | 1d424691d6ce |
| 8 | 1667 | 6ddef4e0965c |
| 9 | 1667 | a2f9b1052dbd |
| 10 | 1667 | a86a016d4d37 |
| 11 | 1667 | ce3b0f163901 |
| 12 | 1667 | 74f77ede176d |
| 13 | 1667 | 336a5a3c5e3f |
| 14 | 1667 | c1238d820956 |
| 15 | 1667 | 0d2371e4e222 |
| 16 | 1667 | d060fea5c391 |
| 17 | 1667 | 8c037d30e79f |
| 18 | 1667 | b5f72c1f3c03 |
| 19 | 1667 | 1a10d9500021 |
| 20 | 1667 | 79d49348d0f5 |
| 21 | 1666 | 81eee79e5170 |
| 22 | 1666 | bd499bb18ba2 |
| 23 | 1666 | c6a7701cd71a |
| 24 | 1666 | 4a2696ef75d9 |
| 25 | 1666 | 561a0a5a185e |
| 26 | 1666 | 434a66440167 |
| 27 | 1666 | db2da1f509bb |
| 28 | 1666 | 237222b1960b |
| 29 | 1666 | f100111af3ef |
| 30 | 1666 | cb91dc0fd13f |

Check: PASS.
