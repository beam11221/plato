# Footnote to `wen_faithful_v1.md`, Step 6 TTA table (Prompt F, Step 5)

`wen_faithful_v1.md` is not rewritten; this file supersedes the two affected rows.

S1 gpu and S1 cpu at seed 4126 are the runs that hung when the disk filled, at rounds 1,957 and 1,929. Their maximum accuracies were 0.8975 and 0.8968, and their last evaluated accuracies 0.8975 and 0.8901. There are two readings:

| run | Reading A: strict threshold (`wen_faithful_v1.md` Step 6) | Reading B: earlier user decision (`main_runs_v1.md`, `cpu_cal_v1.md`) |
|---|---|---|
| S1_gpu_s4126 | TTA@90% not reached (last 0.8975) | counted as reaching 90%; last accuracy 0.8975 |
| S1_cpu_s4126 | TTA@90% not reached (last 0.8901) | counted as reaching 90%; last accuracy 0.8901 |

From Prompt F onward, every table uses **Reading B**: the two runs count as reaching 90%.

None of the Prompt F tables use TTA@90%. They use dispatch, merge and loss logs only, and they read these two runs' logs up to the hang.
