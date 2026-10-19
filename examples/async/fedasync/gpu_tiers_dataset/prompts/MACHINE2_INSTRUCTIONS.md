# Machine 2 instructions — Stage 3 split, Batch 2 + Batch 3

Written 2026-09-14, revised the same day after confirming Machine 2 mounts
the **same** network volume as Machine 1 in real time — same absolute paths,
same files, no lag. That changes this document from what it originally was:
there is **no git clone, no push, no pull** in here. Code, fleets, configs
and results are the same files on disk for both machines the moment either
one writes them. The only things that do NOT come for free from the shared
mount are the compute environment (Python/CUDA — each machine is a separate
GPU host) and process supervision (each machine's `screen`/`pgrep` only sees
its own processes — see §3, this is the part that needs care).

Machine 1 (this pod) is running Batch 1 (the C ∈ {1,2,4,8} sweep at N=20,
17 runs) in `screen -dmS gpu3sweep` / `gpu3watch`, working through
`batch1_names.txt` in order, gate cell (C=4) first. Machine 2's job is
**Batch 2** (N=40, C=4, 4 arms) then **Batch 3** (`gpu_oracle`, 2 runs, run
LAST — see `prompts/STAGE3_run_sweep.md`). Run name lists are already
written down: `batch2_names.txt`, `batch3_names.txt`, and
`machine2_names.txt` (both batches combined, for the watchdog scope — §4).

Read this whole file before running anything.

---

## 0. What you're building

| Batch | Cell | Arms | Runs | Order |
|---|---|---|---|---|
| 2 | N=40, C=4 | baseline, gpu, cpu, observed | 4 | any order among themselves |
| 3 | N=20, `gpu_oracle` | C=4, C=8 | 2 | **after all of Batch 2 has a `.done` marker** |

## 1. Get onto the shared volume

Whatever "rent a second machine" turns out to mean concretely (a pod that
attaches the same network volume, an SSH box with the volume mounted at the
same path, etc.), confirm on Machine 2 that this resolves to the exact same
files Machine 1 sees, not a copy:

```bash
ls -la /workspace/plato/examples/async/fedasync/gpu_tiers_dataset/logs/gpu_a0p8_N20_C4_s4126.log
# should show it already exists and is actively growing (Machine 1's live run)
```

If that file isn't there, or is empty, or doesn't grow between two checks a
few seconds apart — stop. That means the mount isn't actually shared and
everything below (which assumes it is) is the wrong plan; go back to a
git-based sync instead.

Do **not** `git clone` a second copy of the repo next to it — that would
create two independent `runs/`, `configs/`, `fleets/` trees at different
paths, defeating the whole point and doubling disk usage for nothing.

## 2. Environment (the one thing that is NOT shared)

Match this exactly — the determinism guarantees (Stage 1/2/Step-0) were
verified against these versions:

- Python 3.11.10
- torch 2.4.1+cu124, torchvision 0.19.1+cu124 (confirm the GPU driver on
  Machine 2 supports CUDA 12.4 before installing)
- aiohttp 3.14.3, boto3 1.43.93, matplotlib 3.11.2, munch 4.0.0, numpy 2.4.6,
  pandas 3.0.5, scikit-learn 1.9.1, scipy 1.17.1, timm 1.0.29,
  transformers 4.45.2, zstandard 0.25.0, python-socketio 5.16.4,
  tenseal 0.3.17, torch-optimizer 0.3.0

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install aiohttp==3.14.3 boto3==1.43.93 matplotlib==3.11.2 munch==4.0.0 \
    numpy==2.4.6 pandas==3.0.5 scikit-learn==1.9.1 scipy==1.17.1 timm==1.0.29 \
    transformers==4.45.2 zstandard==0.25.0 python-socketio==5.16.4 \
    tenseal==0.3.17 torch-optimizer==0.3.0
```

Use a venv (or equivalent) local to Machine 2 — don't try to share a
`.venv/` over the network mount; it's not designed for that and can behave
unpredictably across two hosts.

## 3. Why a shared mount needs `--names-file` — read before launching

`watchdog.sh` decides whether a run's checkpoint cleaner is alive by
`pgrep`-ing the local process table. That only sees processes on the host
running the check. Since `configs/`, `runs/` and `logs/` are now genuinely
the same directory for both machines, a watchdog with no scoping would see
**every** run — including the other machine's — and, unable to find the
other machine's cleaner process locally, would conclude it's dead and start
a second cleaner against the same `checkpoints/` directory from a different
host. Two processes on two machines deleting/writing the same files over the
network is exactly the kind of concurrent-write pressure that stalled every
live run once already on Machine 1 (2026-09-14, bulk `du`+`rm` incident) —
just triggered by a second host instead of a bulk command this time.

The fix is already in `watchdog.sh` (added today): `--names-file FILE`
restricts what a given watchdog instance watches to the names listed in
`FILE`, one per line. Machine 1's `gpu3watch` has already been restarted
with `--names-file batch1_names.txt`. **Machine 2 must use
`--names-file machine2_names.txt` from the very first launch** — never run
`watchdog.sh` unscoped while another machine's watchdog is also running
against this shared tree.

The other new flag, `--no-janitor`: `janitor.sh` prunes checkpoints for
*any* `.done`-marked run across the whole shared tree, regardless of which
machine ran it — that part is fine to leave unscoped, and Machine 1's
janitor (already running, restarted automatically by `gpu3watch`) already
covers Machine 2's runs too. Running a second janitor on Machine 2 would
just be redundant polling of the same tree, not incorrect, but there's no
reason to pay for it twice — pass `--no-janitor` on Machine 2's watchdog.

## 4. Confirm the disk quota before doing anything else

Because the mount is shared, disk usage is **shared too** — Machine 2's
writes count against the same pool Machine 1 is already watching. Do not
guess a quota independently; confirm with the user what the real number is
(Machine 1's confirmed quota is 200 GB, but do not assume Machine 2 draws
from the same allocation without asking) before passing `--quota-gb` to
anything. `janitor.sh` is already running from Machine 1 and covers the
whole tree — Machine 2 does not need its own janitor (see §3).

## 5. Regenerate the fleet — only if it isn't already there

`fleets/` lives on the shared mount, so if Machine 1 (or an earlier Stage)
already built the N=40 fleet, it's already there — check before building:

```bash
cd examples/async/fedasync/gpu_tiers_dataset
ls fleets/ | grep '^n40_'
```

If nothing matches, build it (deterministic — running this on either
machine produces byte-identical output, so there's no "wrong" machine to run
it from):

```bash
python3 build_fleet.py --total 40 --partition-size 1250 \
    --jitter-sigma 0.149166 --seed 4126
```

If a matching directory already exists, leave it alone — do not pass
`--force` "just to be sure"; that flag is for a genuine digest mismatch, and
rebuilding a fleet already in use by a config risks a race if anything is
mid-read.

## 6. Render configs and preflight — also idempotent, also shared

Same logic: if Machine 1 or a previous session already rendered these, skip
straight to preflight. `make_configs.py` writes deterministic output, so
re-rendering is harmless but unnecessary.

```bash
ls configs/ | grep -E 'N40_C4|gpu_oracle'
```

If Batch 2's configs are missing:

```bash
python3 make_configs.py --alphas 0.8 --seeds 4126 --clients 40 \
    --concurrency 4 --arms baseline gpu cpu observed
```

If Batch 3's are missing:

```bash
python3 make_configs.py --alphas 0.8 --seeds 4126 --clients 20 \
    --concurrency 4 8 --arms gpu_oracle
```

Then preflight everything this machine is about to run:

```bash
python3 preflight.py --configs \
    configs/baseline_a0p8_N40_C4_s4126.toml \
    configs/gpu_a0p8_N40_C4_s4126.toml \
    configs/cpu_a0p8_N40_C4_s4126.toml \
    configs/observed_a0p8_N40_C4_s4126.toml \
    configs/gpu_oracle_a0p8_N20_C4_s4126.toml \
    configs/gpu_oracle_a0p8_N20_C8_s4126.toml
```

Must pass clean. If it fails, stop and report rather than working around it
— this is the same check that caught a real cross-fleet contamination bug
during Stage 2.

## 7. Launch — in `screen`, scoped, from the very first command

Non-negotiable, for a reason that already cost real time on Machine 1: a run
launched tied to whatever shell/session started it (the Bash tool's
`run_in_background`, an SSH session that gets closed, etc.) dies with that
session. `screen` detaches it. Verify the detachment — don't just trust the
`-dmS` call:

```bash
# Batch 2 first
screen -dmS m2sweep bash -c \
  'cd /workspace/plato/examples/async/fedasync/gpu_tiers_dataset && \
   ./run_sweep.sh --from-file batch2_names.txt --max-parallel 4'

screen -dmS m2watch bash -c \
  'cd /workspace/plato/examples/async/fedasync/gpu_tiers_dataset && \
   ./watchdog.sh --names-file machine2_names.txt --no-janitor --log-tag m2 --quota-gb <REAL_QUOTA>'

# confirm both are actually detached (PPID must be 1, not your shell's PID):
ps -eo pid,ppid,sid,cmd | grep -E 'run_sweep|watchdog' | grep -v grep
```

If `ps` shows PPID anything other than 1 for the `screen`-owned process,
stop — it was launched outside `screen` by mistake.

**Do not launch Batch 3 until Batch 2's 4 runs all have a `.done` marker.**
This is the brief's own ordering rule, not a resource constraint —
`gpu_oracle` is the non-deployable upper-bound arm and isn't needed to read
the other results, so it runs last on principle.

```bash
ls logs/*N40_C4*.done   # wait until this lists all 4
```

Once it does:

```bash
screen -dmS m2oracle bash -c \
  'cd /workspace/plato/examples/async/fedasync/gpu_tiers_dataset && \
   ./run_sweep.sh --from-file batch3_names.txt --max-parallel 2'
```

`m2watch` (already scoped to `machine2_names.txt`, which includes both
batches) keeps watching through the whole thing without needing a restart.

## 8. What each `screen` session actually shows you

Neither `m2sweep` nor `m2watch` shows live training progress if you
`screen -r` into them:

- `m2sweep` launches the trainer processes, then just waits — silent until
  something finishes or fails.
- `m2watch` prints a heartbeat only once per interval (15 min by default).
- **Actual live progress**: `gpu_tiers_dataset/logs/<run_name>.log`, one
  file per run. `tail -f logs/gpu_a0p8_N40_C4_s4126.log`, or
  `grep -c 'Starting round' logs/<name>.log` for a quick count.
- Machine 2's watchdog heartbeat specifically lives in
  `logs/watchdog_m2.log` (the `--log-tag m2` above keeps it out of Machine
  1's `logs/watchdog.log`, since both would otherwise interleave into the
  same shared file).

## 9. Cleanup discipline — do not repeat the other incident

A bulk `du -sh` over the whole runs tree immediately followed by `rm -rf`
across thousands of files, run once while 5 trainers were live, stalled all
5 within seconds — the shared filesystem is a network mount, and the bulk
op appears to have starved a blocking checkpoint-write syscall inside
PLATO's single-threaded event loop. `janitor.sh` and each run's own cleaner
(started by `run_sweep.sh`) already handle routine cleanup safely in small
increments — you shouldn't need to intervene by hand. If you ever do need to
manually delete something under `runs/` while runs are live on *either*
machine:

- throttle it: `nice -n 19 ionice -c3 <command>`
- delete in small chunks (tens of files, not thousands, per call)
- check both machines' live logs are still growing between chunks
- never run a bulk `du` over the whole tree followed by a bulk `rm` —
  that combination is exactly what caused the stall

If disk pressure looks urgent, stop and ask rather than improvising a bulk
delete.

## 10. Results

Nothing to transfer — `runs/gpu_tiers_dataset/<run_name>/results/`,
`*.log`, and `config_used.toml` are already visible to Machine 1 the instant
Machine 2 writes them, since it's the same files. Whoever runs `collect.py` /
`report.py` for the final analysis can read straight across both batches
without any merge step.

## 11. If something breaks

Report it rather than working around it — in particular:

- `preflight.py` fails → stop, don't `--force` past it
- a run's log shows `Traceback` / `CUDA out of memory` / gets killed →
  `watchdog.sh` flags it as an ERROR line in `logs/watchdog_m2.log`; leave
  the run stopped and report what the traceback says rather than
  restarting it — whether a partially-finished cell is still comparable to
  the others is a judgement call the brief leaves to a person (see
  `watchdog.sh`'s own "what it will not do" section)
- disk usage crosses the real quota → Machine 1's janitor (shared, whole
  tree) should already be handling it; if it isn't, stop new runs before
  doing anything else, on both machines

Fixes that are genuinely mechanical (a missing package, a wrong path) are
fine to just fix and continue. Anything that changes what got measured —
restarting a stalled run, deleting a `.done` marker, editing a config after
the fact — needs a person's decision first.
