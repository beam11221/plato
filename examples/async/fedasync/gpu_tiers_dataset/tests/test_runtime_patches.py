"""Tests for Stage 3 Step 0 -- the host's elapsed time kept out of the clock.

The patch replaces one method on `plato.servers.base.Server`, so these tests
install it against a stand-in server object and check the three things that
matter: it only fires when the config asks, it actually zeroes the value, and it
fails loudly rather than silently if it cannot.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import runtime_patches

DATASET_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reset_patch_state(monkeypatch):
    """The patch is a module-level latch; each test starts from unpatched."""
    from plato.servers import base
    original = base.Server.process_client_info
    runtime_patches._PATCHED = False
    runtime_patches._RECORD.update(
        {"applied": False, "reports_seen": 0, "overhead_s": 0.0}
    )
    yield
    base.Server.process_client_info = original
    runtime_patches._PATCHED = False
    runtime_patches._RECORD.update(
        {"applied": False, "reports_seen": 0, "overhead_s": 0.0}
    )


def set_config(monkeypatch, **server_keys):
    fake = SimpleNamespace(server=SimpleNamespace(**server_keys))
    monkeypatch.setattr("plato.config.Config", lambda *a, **k: fake)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

class TestSwitch:
    def test_absent_key_means_off(self, monkeypatch):
        set_config(monkeypatch)
        assert runtime_patches.wanted() is False
        assert runtime_patches.apply_if_requested() is False

    def test_false_means_off(self, monkeypatch):
        set_config(monkeypatch, exclude_processing_time_from_clock=False)
        assert runtime_patches.apply_if_requested() is False

    def test_true_means_on(self, monkeypatch):
        set_config(monkeypatch, exclude_processing_time_from_clock=True)
        assert runtime_patches.apply_if_requested() is True
        assert runtime_patches.record()["applied"] is True

    def test_off_leaves_the_method_untouched(self, monkeypatch):
        from plato.servers import base
        before = base.Server.process_client_info
        set_config(monkeypatch)
        runtime_patches.apply_if_requested()
        assert base.Server.process_client_info is before

    def test_on_replaces_the_method(self, monkeypatch):
        from plato.servers import base
        before = base.Server.process_client_info
        set_config(monkeypatch, exclude_processing_time_from_clock=True)
        runtime_patches.apply_if_requested()
        assert base.Server.process_client_info is not before

    def test_applying_twice_does_not_stack(self, monkeypatch):
        from plato.servers import base
        set_config(monkeypatch, exclude_processing_time_from_clock=True)
        runtime_patches.apply_if_requested()
        once = base.Server.process_client_info
        runtime_patches.apply_if_requested()
        assert base.Server.process_client_info is once

    def test_the_key_name_matches_what_make_configs_writes(self):
        template = (DATASET_DIR / "make_configs.py").read_text()
        assert f"{runtime_patches.CONFIG_KEY} = {{exclude_processing_time}}" in template


# ---------------------------------------------------------------------------
# What it does to a report
# ---------------------------------------------------------------------------

class FakeServer:
    """Enough of a server for the patched method to run against."""

    def __init__(self, reports):
        self.reports = reports
        self.seen = []

    async def process_client_info(self, client_id, sid):
        # Stands in for the real method; records what it WOULD have used.
        self.seen.append((client_id, self.reports[sid].processing_time))


def install_against(fake_server_cls, monkeypatch):
    """Point the patch at a stand-in class instead of the real Server.

    `apply_if_requested()` mutates `base.Server.process_client_info` with a
    plain attribute assignment, not through `monkeypatch` -- so if `base.Server`
    is monkeypatched to `fake_server_cls` here, the wrapped method lands
    PERMANENTLY on `fake_server_cls` itself (test classes are module-level
    objects, reused across every test in this file) unless something tells
    `monkeypatch` to undo it.

    The fix: register `fake_server_cls.process_client_info` with `monkeypatch`
    BEFORE `apply_if_requested()` overwrites it. `monkeypatch.setattr(obj, name,
    value)` records `(obj, name, CURRENT value)` for teardown regardless of what
    `value` is passed, so setting it to its own current value is a real no-op now
    and a correct restoration later -- `apply_if_requested()`'s later direct
    assignment is invisible to `monkeypatch`'s bookkeeping, but its teardown
    still puts the ORIGINAL attribute back on `fake_server_cls` when the test
    ends, which is what actually matters here.
    """
    from plato.servers import base
    monkeypatch.setattr(fake_server_cls, "process_client_info",
                        fake_server_cls.process_client_info)
    monkeypatch.setattr(base, "Server", fake_server_cls)
    runtime_patches.apply_if_requested()
    return fake_server_cls


class TestZeroing:
    def report(self, processing_time=0.0654):
        return SimpleNamespace(
            processing_time=processing_time,
            training_time=15.0,
            comm_time=0.0,
            client_id=7,
        )

    def test_processing_time_is_zero_when_the_original_runs(self, monkeypatch):
        set_config(monkeypatch, exclude_processing_time_from_clock=True)
        install_against(FakeServer, monkeypatch)

        reports = {"sid-1": self.report(0.0654)}
        server = FakeServer(reports)
        asyncio.run(FakeServer.process_client_info(server, 7, "sid-1"))

        assert server.seen == [(7, 0.0)]
        assert reports["sid-1"].processing_time == 0.0

    def test_training_and_comm_time_are_untouched(self, monkeypatch):
        """Only the host-measured term goes; the simulated ones stay."""
        set_config(monkeypatch, exclude_processing_time_from_clock=True)
        install_against(FakeServer, monkeypatch)

        reports = {"s": self.report()}
        asyncio.run(FakeServer.process_client_info(FakeServer(reports), 7, "s"))
        assert reports["s"].training_time == 15.0
        assert reports["s"].comm_time == 0.0

    def test_the_measured_value_is_accumulated_for_the_record(self, monkeypatch):
        set_config(monkeypatch, exclude_processing_time_from_clock=True)
        install_against(FakeServer, monkeypatch)

        reports = {"a": self.report(0.05), "b": self.report(0.07)}
        server = FakeServer(reports)
        asyncio.run(FakeServer.process_client_info(server, 1, "a"))
        asyncio.run(FakeServer.process_client_info(server, 2, "b"))

        record = runtime_patches.record()
        assert record["reports_seen"] == 2
        assert record["overhead_s"] == pytest.approx(0.12)

    def test_a_missing_report_is_not_an_error(self, monkeypatch):
        """The original handles it; the patch must not turn it into a crash."""
        set_config(monkeypatch, exclude_processing_time_from_clock=True)

        class NoReport(FakeServer):
            async def process_client_info(self, client_id, sid):
                self.seen.append((client_id, None))

        install_against(NoReport, monkeypatch)
        server = NoReport({})
        asyncio.run(NoReport.process_client_info(server, 7, "absent"))
        assert server.seen == [(7, None)]

    def test_an_unsettable_report_raises_rather_than_passing_silently(
        self, monkeypatch
    ):
        """A patch that quietly does nothing is worse than no patch: the run
        would look patched in the log and behave like an unpatched one."""
        set_config(monkeypatch, exclude_processing_time_from_clock=True)
        install_against(FakeServer, monkeypatch)

        class Frozen:
            client_id = 7
            processing_time = 0.0654

            def __setattr__(self, name, value):
                raise AttributeError("read-only")

        server = FakeServer({"s": Frozen()})
        with pytest.raises(RuntimeError, match="could not clear processing_time"):
            asyncio.run(FakeServer.process_client_info(server, 7, "s"))


# ---------------------------------------------------------------------------
# Nothing else in PLATO reads the value
# ---------------------------------------------------------------------------

class TestNoOtherConsumer:
    """The brief: check whether processing_time is consumed anywhere else --
    aggregation weighting, staleness, or logging later analysis depends on."""

    def consumers(self) -> dict[str, list[str]]:
        import plato
        root = Path(plato.__file__).parent
        hits: dict[str, list[str]] = {}
        for path in root.rglob("*.py"):
            lines = [
                f"{path.relative_to(root)}:{i}"
                for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
                if "processing_time" in line
            ]
            if lines:
                hits[str(path.relative_to(root))] = lines
        return hits

    def test_only_two_modules_read_it(self):
        files = set(self.consumers())
        readers = {f for f in files if f.startswith("servers/")}
        assert readers == {"servers/base.py", "servers/fedavg.py"}, readers

    def test_staleness_does_not_use_it(self):
        """Staleness is current_round - starting_round, a pure round count."""
        import plato.servers.base as base_module
        source = Path(base_module.__file__).read_text()
        for line in source.splitlines():
            if "staleness" in line and "processing_time" in line:
                pytest.fail(f"staleness touches processing_time: {line.strip()}")

    def test_aggregation_does_not_use_it(self):
        import plato
        root = Path(plato.__file__).parent
        for path in (root / "servers" / "strategies").rglob("*.py"):
            assert "processing_time" not in path.read_text(errors="replace"), path

    def test_the_only_server_reader_besides_finish_time_is_logging(self):
        import plato.servers.fedavg as fedavg_module
        source = Path(fedavg_module.__file__).read_text()
        for i, line in enumerate(source.splitlines(), 1):
            if "processing_time" in line:
                window = "\n".join(source.splitlines()[max(0, i - 20):i])
                assert "get_logged_items" in window, f"fedavg.py:{i} is not logging"

    def test_this_folder_does_not_record_those_columns(self):
        """So zeroing the value cannot even change a CSV we keep."""
        template = (DATASET_DIR / "make_configs.py").read_text()
        results_line = next(l for l in template.splitlines()
                            if l.startswith("types = "))
        assert "processing_time" not in results_line
        assert "round_time" not in results_line
