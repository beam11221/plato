# `gpu_tiers_dataset/` — the tier specs, the fleets, and the runs built on them

This folder is the single source of truth for **what the hardware is** and **what
each client sleeps**. Everything downstream reads from here; nothing here reads
from an experiment folder.

```
tiers/          tier_id -> GPU spec.  The datasheet. Two files, one per belief.
fleets/         one immutable directory per fleet: composition + sleep times +
                audit table + manifest
build_fleet.py  the ONLY thing that writes a fleet
make_configs.py fleet + arm + cell -> a TOML per run
preflight.py    everything agrees, and prints the numbers the run must reproduce
fedasync_seeded.py  the entry point; seeds every RNG, one script for all arms
```

## Why this folder exists

Three problems in the older layout, in the order they bite.

**1. The `gpu` arm was an oracle.** `../separability/build_inputs.sh` fed
`tiers_estimated_15x.csv` to the sleep generator, and
`../separability/make_configs.py` pointed the `gpu` arm's predictor at the same
file with the same anchor. Its belief equalled reality **exactly, by
construction**. It was never an estimator, so the claim it exists to test — that a
spec-derived estimate is good enough — had never been tested. The generator here
applies a persistent per-client `exp(eps_i)` that no datasheet can know, which
leaves `gpu` holding the tier mean and adds `gpu_oracle` for the upper bound.

**2. Nothing seeded torch.** `../separability/PARALLELISM_CHECK.md` measured two
runs of one config, both alone on an idle box, differing by **6.29 %** of
simulated clock. That variance sat inside the project's noise floor and was read
as natural variation. `seeding.py` derives every RNG from the config's own
`random_seed`.

**3. The inputs were two loose files that had to agree.**
`../separability/README.md` lists it as a landmine: *"The fleet file and the sleep
CSV are one fleet seen from two sides... crossing files from different builds is
silent."* Here one command writes one directory and a config names that directory.
There is nothing left to cross.

The tier specs themselves were also duplicated byte-for-byte in
`../gpu_tiers/input/` and `../surgical/input/`, with live configs pointing at
**both** — `tiers_cpu.csv` was referenced 28 times from one and 13 from the other.
Editing one would have moved some arms and not others, silently.

## The arms

| arm | what T-hat it holds | role |
|---|---|---|
| `baseline` | — (random selection) | control |
| `gpu` | the **tier mean** from the datasheet | the deployable method; what the thesis claims |
| `cpu` | a tier file that cannot tell xavier from orin | a wrong spec, deliberately |
| `observed` | the client's own last measured staleness | Polaris's reactive estimator |
| `gpu_oracle` | the **per-client truth**, jitter included | upper bound; not deployable |

Arms of one cell differ in four keys and no others (`make_configs.validate`):
`base_path`, `port`, `staleness_predictor`, `tiers_csv`, plus `oracle_sleep_csv`
on the one arm allowed to hold it.

## Building a fleet

```bash
cd examples/async/fedasync/gpu_tiers_dataset
python3.11 build_fleet.py --total 20 --partition-size 2500 \
    --jitter-sigma 0.18 --seed 4126
```

writes `fleets/n20_x10-o7-t3_xa15_p2500_j0p180_s4126/`:

| file | what it is |
|---|---|
| `clients.csv` | `client_id, tier_id` — the fleet |
| `sleep_times.csv` | `client_id, sleep_time` — **what PLATO reads** |
| `sleep_table.csv` | `client_id, tier, T_tier_s, eps, sleep_s` — the audit trail |
| `manifest.json` | every parameter, the tier file's sha256, and the realised stats |

The directory name carries the swept axes; the manifest's `inputs_digest` covers
everything else, including the tier file's contents and the generator version.
**`build_fleet.py` refuses to overwrite a directory built from different inputs**,
because configs address fleets by directory name and a fleet swapped underneath
one would change what a run did without changing what it was called.

`--jitter-sigma 0` reproduces the pre-jitter generator byte for byte — asserted
against `../separability/output/` and `../separability_n100/output/` in the tests
— so an oracle-branch fleet can be rebuilt here and checked against disk.

## The jitter

```
sleep_i = T_tier(tier_i) * exp(eps_i),   eps_i ~ Normal(0, sigma)
```

`eps_i` is drawn **once per client when the fleet is built** and held for the
whole run: it models a persistent per-unit offset (memory bandwidth, thermals,
background load), not transient load.

Three things about it are easy to get wrong and are worth knowing before reading
any result:

- **`T_tier` is the tier's MEDIAN, not its mean.** `exp(eps)` has median 1 and
  mean `exp(sigma^2/2)`. No correction is applied — the brief specifies the
  formula above, and a correction would quietly move the belief/reality gap `gpu`
  exists to measure.
- **The binding constraint on sigma is orin/thor, not the CV target.** Those tiers
  are only 1.92x apart before jitter. At sigma = 0.20 their ranges overlap in the
  N = 20 draw; at 0.18 they clear by 5.1 %. `manifest.json` records the margin.
- **The realised within-tier CV will not all land in 15-20 %,** and no sigma makes
  it. With 3-10 clients per tier the sampling error is several points, and orin is
  systematically the lowest while thor is the highest. sigma is chosen on the
  *nominal* CV, `sqrt(exp(sigma^2)-1)`, which is a property of sigma alone.
  `stage1_evidence/jitter_check.txt` § 2 has the numbers for all three fleet sizes.

## Running

```bash
cd examples/async/fedasync
export PYTHON=python3.11          # the bundled .venv is broken

python3.11 gpu_tiers_dataset/make_configs.py --alphas 0.8 --clients 20 --concurrency 4
python3.11 gpu_tiers_dataset/preflight.py          # must pass before any launch

R=$(cd ../../.. && pwd)
PYTHONPATH="$R:$PWD/gpu_tiers:$PWD" python3.11 \
    gpu_tiers_dataset/fedasync_seeded.py --arm gpu \
    -c gpu_tiers_dataset/configs/gpu_a0p8_N20_C4_s4126.toml
```

The repo root **must** be on `PYTHONPATH`: PLATO is imported from the source tree,
not installed.

## Landmines

Everything in `../gpu_tiers/HANDOFF.md` § 4 and `../surgical/CODE_AUDIT.md` § 4
still applies. These are the ones specific to this folder.

**`sim_time_s` can never be bit-identical between two runs.** PLATO adds the
host's own `processing_time` — a `time.perf_counter()` delta — to every report's
finish time, and that finish time becomes the simulated clock. Seeding fixes the
dispatch sequence, not this. See `stage1_evidence/determinism.txt`.

**Two preflight checks are the OPPOSITE of `../separability/preflight.py`.** That
file fails when a tier has more than one sleep time; this one fails when a tier
has only one. Copying a check between the two folders breaks it in whichever
direction it is copied.

**`--arm` must survive into the spawned client processes.** PLATO's clients are
`spawn`ed and re-import the entry module, so they re-parse argv. `fedasync_seeded.py`
strips `--arm` for the PLATO import and puts it back; deleting the restore makes
every client exit at launch while the server waits forever — no traceback, just a
hang.

**Ports are planned, not chosen, and the TAG is part of the plan.** Real runs live
at 51000-55913, tagged builds at 56000+. `--tag _det1` and `--tag _det2` are the
same config run twice and must not share a port; unregistered tags are refused.

**N = 30 is not a 50/35/15 fleet.** It cannot be: 50/35/15 of 30 is 15/10.5/4.5.
The remainder goes to thor, giving 15/10/5 = 50/33.3/16.7. Thor is the fastest
tier, so a sweep along N moves fleet size *and* a little tier mix at that point.
The fleet is named after its realised counts (`n30_x15-o10-t5`) and the manifest
records both.

**`avg_training_time` must be present and `0.0`.** Absent, the first client report
raises; non-zero, it is an additive constant identical for every client and the
effective tier gap collapses toward 1.

**`comm_simulation = true` means "charge nothing for communication".** It is not a
switch that turns communication on.

**Do not "fix" the frozen denominator** in the observed arm, the optimistic cold
start, or the 10x exploration bonus. They are Polaris's behaviour and they are the
treatment (`../surgical/CODE_AUDIT.md` § 4.1, § 4.2).

## Relationship to the older folders

`../surgical/`, `../separability/`, `../separability_n100/` and
`../sumo_close_loop/` are the **oracle branch**: their `gpu` arm was an oracle and
nothing they produced is poolable with anything produced here. They are kept for
reference and were **not modified** — they still point at their own copies of the
tier files and run their own unseeded entry scripts, so they remain reproducible
as they are.

Two files outside this folder were edited, both default-off, so those folders are
unaffected: `../gpu_tiers/gpu_tier_sleep.py` (the `oracle_sleep_csv` key) and
`../fedasync_polaris_vi_server.py` (the `gpu_oracle` arm). See
`stage1_evidence/CHANGES.md`.

`../gpu_tiers/generate_sleep_times.py` is superseded by `build_fleet.py` and was
left untouched; the folders that call it keep producing exactly what they produced.

## Tests

```bash
cd gpu_tiers_dataset && python3.11 -m pytest tests/ -q
```

No GPU needed for most of them, ~35 s. They have already caught two bugs that a
sweep would have paid for: an infinite recursion in the oracle branch, and the
`--arm` argv regression that killed every client process at launch.
