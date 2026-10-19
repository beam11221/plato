# Stage 3 — Step 0: keeping processing_time out of finish_time

## Consumer check, before patching

`processing_time` is read in exactly two places under `plato/`:

| file | use |
|---|---|
| `plato/servers/base.py:1028` | `finish_time` -- the merge-order term, the target |
| `plato/servers/fedavg.py:313-322` | `get_logged_items()` -- two LOG COLUMNS only (`processing_time`, `round_time`) |

Not used in aggregation weighting, and not used in staleness -- staleness is
`current_round - starting_round` (`base.py:1168/1219/1246`), a pure round count
with no wall-clock term at all. This folder's configs record
`round, elapsed_time, accuracy, train_loss, valid_loss` -- neither log column
above is written, so nothing downstream reads them either.
`tests/test_runtime_patches.py::TestNoOtherConsumer` asserts this by scanning
`plato/` itself, not by inspection alone.

## The patch

`gpu_tiers_dataset/runtime_patches.py` wraps
`plato.servers.base.Server.process_client_info` -- the one entry point both
report-arrival paths call (`base.py:942`, `:997`) -- and zeroes
`report.processing_time` before the original method runs. Installed from
`fedasync_seeded.py`, before `Server` is constructed; nothing under `plato/`
is edited. Switched by `[server] exclude_processing_time_from_clock` in the
config, default on (`make_configs.EXCLUDE_PROCESSING_TIME = True`) now that
verification below has passed.

## Verification: two 200-round runs, patch ON

Same config as `stage2_evidence/determinism_after.txt` (N=20, C=4, alpha=0.8,
seed 4126, `gpu` arm, sigma=0.149166), run sequentially, alone on the box.

| check | result |
|---|---|
| dispatch sequence identical | **True** |
| merge sequence identical | **True** |
| reconstructed clock identical | **True**  (A=285.320754119 s, B=285.320754119 s, diff=0.000e+00 s) |
| accuracy curve identical row for row | **True** |
| merge order ascending in reconstructed time | A: True (0 problems)  B: True (0 problems) |

All three of the brief's success criteria hold exactly, not approximately.

**Bonus, not required by the brief:** PLATO's OWN clock now matches too --
A=285.320756 s, B=285.320756 s, spread 0.0000%.
Before the patch (`stage2_evidence/determinism_after.txt`), the corresponding
pair spread by 2.1577% on PLATO's clock and 2.1435% reconstructed, with the
divergence traced to a 0.592 s near-tie in the merge heap. That mechanism is
gone: `finish_time` no longer carries anything the host measured, so two runs
of one config produce the identical heap order, not just a numerically close one.

## Decision

**Step 0 is IN**, for the whole Stage 3 sweep. `exclude_processing_time_from_clock`
defaults to `true` in every config `make_configs.py` renders from here on.

Consequence for the parallelism rule: `determinism_after.txt`'s 0.80%/2.16%
residuals came from `processing_time` reordering the merge heap on a machine
under load, and Step 0 removes exactly that mechanism. Per the brief --
"If Step 0 succeeded, this constraint is gone and cells may be run in any
arrangement" -- cells are no longer required to run one at a time; the campaign
may run multiple cells concurrently, GPU memory permitting.

