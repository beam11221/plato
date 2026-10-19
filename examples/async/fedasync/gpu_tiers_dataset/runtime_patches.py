"""
Keep the host's own elapsed time out of the simulated clock, without editing PLATO.

The problem (stage2_evidence/determinism_after.txt section 6)
-------------------------------------------------------------
PLATO builds every report's finish time as

    training_time + processing_time + comm_time + start_time
                                              plato/servers/base.py:1025-1030

and merges by popping the smallest `(finish_time, client_id)` off a heap
(`base.py:1036-1047`). `processing_time` is a `time.perf_counter()` delta measured
on the host (`plato/clients/strategies/defaults.py:252`), ~0.065 s per report, and
it accumulates per queue slot -- so by a few dozen rounds in, two slots differ by
several tenths of a second.

Measured consequence: at merge 38 of two otherwise identical runs, clients #7, #10
and #20 finished within **0.592 s** of each other in simulated terms and the two
runs merged them in different orders. One run's order was ascending in simulated
time; the other's was not. **The machine was deciding which client merged first,
not the experiment**, and from that point the two runs handed different models to
different clients and diverged.

Stage 2 concluded this needed an edit to `plato/servers/base.py`. It does not:
`fedasync_seeded.py` already runs before PLATO is imported, so the value can be
neutralised at run time instead.

What this patch does
--------------------
Wraps `Server.process_client_info`, the single entry point through which every
report reaches the finish-time computation (`base.py:942` and `base.py:997` both
call it), and sets the report's `processing_time` to 0 before the original runs.
The measured value is logged first, so the host overhead is still recorded -- it
just stops entering the simulation.

This is the same quantity `clock_reconstruction.py` already removes when rebuilding
the clock. Removing it at run time as well makes PLATO's merge ORDER agree with the
time axis the analysis actually uses.

Is `processing_time` used anywhere else?
----------------------------------------
Checked before patching, across all of `plato/`:

    plato/servers/base.py:1028      finish_time          <- the target
    plato/servers/fedavg.py:313-322 get_logged_items()   <- log columns only
    plato/clients/**                produce it; never read it back

`fedavg.py` writes it into the results CSV as the `processing_time` and
`round_time` columns. This folder's configs record
`round, elapsed_time, accuracy, train_loss, valid_loss`, so neither column is
written, and nothing downstream reads them. It is NOT used in aggregation
weighting, and it is NOT used in staleness -- staleness is
`current_round - starting_round` (`base.py:1098`, `base.py:1172`), which contains
no wall-clock term at all.

So the patch changes the merge order and nothing else.

Switched from the config, not hard-coded
----------------------------------------
`[server] exclude_processing_time_from_clock`. A run's identity has to live in its
config: `config_used.toml` is copied into every run directory, so an analysis can
always tell whether a given run had the patch. A silent always-on patch would make
runs from before and after indistinguishable.
"""

from __future__ import annotations

import logging
from typing import Any

_PATCHED = False
_RECORD: dict[str, Any] = {"applied": False, "reports_seen": 0, "overhead_s": 0.0}

CONFIG_KEY = "exclude_processing_time_from_clock"


def wanted() -> bool:
    """Whether `[server] exclude_processing_time_from_clock` asks for the patch.

    Absent means False, so a config written before this existed produces a run
    identical to what it produced then.
    """
    try:
        from plato.config import Config  # noqa: PLC0415

        return bool(getattr(Config().server, CONFIG_KEY, False))
    except Exception:  # noqa: BLE001 — Config not initialised in tests
        return False


def record() -> dict[str, Any]:
    """What the patch did in this process. Empty-ish if it never applied."""
    return dict(_RECORD)


def apply_if_requested() -> bool:
    """Install the patch when the config asks for it. Returns whether it is on.

    Idempotent: wrapping twice would zero an already-zero value, which is
    harmless, but the guard keeps the log honest about how many layers are on.
    """
    global _PATCHED

    if not wanted():
        logging.info(
            "[RuntimePatch] [server] %s is not set; PLATO's clock keeps the "
            "host's processing_time, as it always has.", CONFIG_KEY,
        )
        return False

    if _PATCHED:
        return True

    from plato.servers import base  # noqa: PLC0415

    original = base.Server.process_client_info

    async def process_client_info(self, client_id, sid):
        report = self.reports.get(sid)
        if report is not None:
            measured = float(getattr(report, "processing_time", 0.0))
            _RECORD["reports_seen"] += 1
            _RECORD["overhead_s"] += measured
            try:
                report.processing_time = 0.0
            except Exception as exc:  # noqa: BLE001
                # Fail loudly. A patch that silently does nothing would produce a
                # run that looks patched in the log and behaves like an
                # unpatched one -- the exact class of undetectable difference
                # this project keeps guarding against.
                raise RuntimeError(
                    f"[RuntimePatch] could not clear processing_time on the "
                    f"report from client {client_id}: {exc}. The simulated clock "
                    f"would still carry the host's elapsed time."
                ) from exc
            logging.debug(
                "[RuntimePatch] client %s: processing_time %.6f s measured, "
                "0 used in finish_time", client_id, measured,
            )
        return await original(self, client_id, sid)

    base.Server.process_client_info = process_client_info
    _PATCHED = True
    _RECORD["applied"] = True
    logging.info(
        "[RuntimePatch] finish_time will EXCLUDE the host's processing_time. "
        "training_time + comm_time + start_time only, so the merge order is a "
        "property of the experiment rather than of the machine. The measured "
        "overhead is still logged and still reported in "
        "`accuracy_curves_reconstructed.csv` via sim_time_plato."
    )
    return True


def log_summary() -> None:
    """Emit what the patch removed over the run. Called at shutdown."""
    if not _RECORD["applied"]:
        return
    seen = _RECORD["reports_seen"]
    total = _RECORD["overhead_s"]
    logging.info(
        "[RuntimePatch] removed %.4f s of host processing_time over %d reports "
        "(mean %.6f s/report); none of it entered the simulated clock.",
        total, seen, total / seen if seen else 0.0,
    )
