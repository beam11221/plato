"""Tests for Change C -- the clock rebuilt outside PLATO.

The replay is arithmetic, so most of it is checked against hand-computed
schedules rather than against a run. The two tests that use a real run are the
ones that matter for C1: the gap must be positive everywhere, and it must track
the per-slot report count.
"""

from __future__ import annotations

import csv
import re
import textwrap
from pathlib import Path

import pytest

import clock_reconstruction as cr

DATASET_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# run_metadata -- experiment/arm/seed, parsed from the run directory name
# ---------------------------------------------------------------------------

class TestRunMetadata:
    """Regression tests for a real bug: the previous implementation split on
    the literal string "_a0p" to find where the arm name ends, which happens
    to work for alpha = 0.8 (Stage 3's only value) but silently breaks for any
    other alpha -- "a1p0" never contains "_a0p" at all, so the split falls back
    to `stem.split("_")[0]`, and `gpu_oracle_a1p0_...` would misparse as arm
    "gpu" and lose "_oracle" into the "experiment" column instead."""

    def config_for(self, seed=4126):
        return {"clients": {"random_seed": seed}}

    @pytest.mark.parametrize("stem,arm,experiment", [
        ("gpu_a0p8_N20_C4_s4126", "gpu", "a0p8_N20_C4_s4126"),
        ("baseline_a0p8_N20_C4_s4126", "baseline", "a0p8_N20_C4_s4126"),
        ("gpu_oracle_a0p8_N20_C8_s4126", "gpu_oracle", "a0p8_N20_C8_s4126"),
        # The exact case the old split-on-"_a0p" implementation could not
        # handle: a DIFFERENT alpha.
        ("gpu_a1p0_N20_C4_s4126", "gpu", "a1p0_N20_C4_s4126"),
        ("cpu_a0p1_N40_C4_s4127", "cpu", "a0p1_N40_C4_s4127"),
    ])
    def test_arm_and_experiment_are_parsed_correctly(self, stem, arm, experiment):
        meta = cr.run_metadata(Path(stem), self.config_for())
        assert meta["arm"] == arm
        assert meta["experiment"] == experiment

    def test_experiment_is_shared_across_the_arms_of_one_cell(self):
        """The whole point of separating "experiment" from "arm": every arm run
        on the same cell must group under one experiment id."""
        cell_stems = ["baseline_a0p8_N20_C4_s4126", "gpu_a0p8_N20_C4_s4126",
                     "cpu_a0p8_N20_C4_s4126", "observed_a0p8_N20_C4_s4126"]
        experiments = {cr.run_metadata(Path(s), self.config_for())["experiment"]
                      for s in cell_stems}
        assert len(experiments) == 1

    def test_a_tagged_probe_still_parses(self):
        meta = cr.run_metadata(Path("gpu_a0p8_N20_C4_s4126_det1"),
                               self.config_for())
        assert meta["arm"] == "gpu"

    def test_seed_comes_from_the_config_not_the_name(self):
        meta = cr.run_metadata(Path("gpu_a0p8_N20_C4_s4126"),
                               self.config_for(seed=9999))
        assert meta["seed"] == 9999


# ---------------------------------------------------------------------------
# C3 -- reading the order out of a log
# ---------------------------------------------------------------------------

LOG = textwrap.dedent("""
    [INFO][00:00:01]: [Server #1] Starting round 1/3.
    [INFO][00:00:01]: [Server #1] Selecting client #20 for training.
    [INFO][00:00:01]: [Server #1] Selecting client #18 for training.
    [INFO][00:00:01]: [Server #1] Selecting client #15 for training.
    [INFO][00:00:01]: [Server #1] Selecting client #14 for training.
    [INFO][00:00:02]: [Server #1] Adding client #18 to the list of clients for aggregation.
    [INFO][00:00:03]: [Server #1] Starting round 2/3.
    [INFO][00:00:03]: [Server #1] Selecting client #4 for training.
    [INFO][00:00:04]: [Server #1] Adding client #20 to the list of clients for aggregation.
    [INFO][00:00:05]: [Server #1] Starting round 3/3.
    [INFO][00:00:05]: [Server #1] Selecting client #13 for training.
    [INFO][00:00:06]: [Server #1] Adding client #15 to the list of clients for aggregation.
""")


class TestReadEvents:
    def test_dispatch_order_is_file_order(self):
        dispatches, _merges, _rounds = cr.read_events(LOG)
        assert [cid for _r, cid in dispatches] == [20, 18, 15, 14, 4, 13]

    def test_round_1_holds_c_dispatches(self):
        """Which is why the round number cannot carry the order."""
        dispatches, _m, _r = cr.read_events(LOG)
        assert sum(1 for rnd, _ in dispatches if rnd == 1) == 4
        assert sum(1 for rnd, _ in dispatches if rnd == 2) == 1

    def test_merges_are_read_in_order(self):
        _d, merges, _r = cr.read_events(LOG)
        assert merges == [(1, 18), (2, 20), (3, 15)]

    def test_total_rounds(self):
        assert cr.read_events(LOG)[2] == 3

    def test_a_log_with_no_events_yields_nothing(self):
        assert cr.read_events("nothing to see") == ([], [], 0)


# ---------------------------------------------------------------------------
# The queue replay, against hand arithmetic
# ---------------------------------------------------------------------------

class TestReplay:
    SLEEPS = {20: 1.0, 18: 0.5, 15: 2.0, 14: 4.0, 4: 8.0, 13: 3.0}

    def schedule(self, comm=0.0):
        dispatches, _m, _r = cr.read_events(LOG)
        return cr.replay(dispatches, self.SLEEPS, concurrency=4, comm_time=comm)

    def test_first_c_dispatches_all_start_at_zero(self):
        schedule = self.schedule()
        assert [d.start for d in schedule[:4]] == [0.0, 0.0, 0.0, 0.0]
        assert [d.slot for d in schedule[:4]] == [0, 1, 2, 3]

    def test_finish_is_start_plus_sleep(self):
        for item in self.schedule():
            assert item.finish == pytest.approx(item.start + self.SLEEPS[item.client_id])

    def test_a_refill_takes_the_slot_that_freed_first(self):
        """Slot 1 holds client 18, which finishes at 0.5 -- the earliest. Client
        4 is dispatched next and must start there."""
        schedule = self.schedule()
        client4 = next(d for d in schedule if d.client_id == 4)
        assert client4.slot == 1
        assert client4.start == pytest.approx(0.5)
        assert client4.finish == pytest.approx(8.5)

    def test_the_second_refill_takes_the_next_slot(self):
        """After client 4, the earliest free is slot 0 (client 20 at 1.0)."""
        schedule = self.schedule()
        client13 = next(d for d in schedule if d.client_id == 13)
        assert client13.slot == 0
        assert client13.start == pytest.approx(1.0)

    def test_comm_time_is_added_per_dispatch(self):
        plain = self.schedule(comm=0.0)
        withc = self.schedule(comm=0.25)
        assert withc[0].finish == pytest.approx(plain[0].finish + 0.25)
        # ...and it compounds down a slot's chain, as a real per-report cost does
        assert withc[4].finish == pytest.approx(plain[4].finish + 0.5)

    def test_a_client_missing_from_the_sleep_csv_is_fatal(self):
        dispatches, _m, _r = cr.read_events(LOG)
        with pytest.raises(SystemExit, match="not in the sleep CSV"):
            cr.replay(dispatches, {20: 1.0}, concurrency=4)

    def test_concurrency_changes_the_schedule(self):
        dispatches, _m, _r = cr.read_events(LOG)
        wide = cr.replay(dispatches, self.SLEEPS, concurrency=4)
        narrow = cr.replay(dispatches, self.SLEEPS, concurrency=2)
        assert narrow[2].start > 0.0          # third dispatch must queue at C=2
        assert wide[2].start == 0.0


class TestClockPerRound:
    SLEEPS = TestReplay.SLEEPS

    def test_round_clock_is_the_merged_client_finish(self):
        dispatches, merges, _r = cr.read_events(LOG)
        schedule = cr.replay(dispatches, self.SLEEPS, 4)
        clocks = cr.clock_per_round(schedule, merges)
        assert clocks[1] == pytest.approx(0.5)    # client 18
        assert clocks[2] == pytest.approx(1.0)    # client 20
        assert clocks[3] == pytest.approx(2.0)    # client 15

    def test_merges_match_dispatches_in_order_when_a_client_repeats(self):
        """A client dispatched twice must have its merges matched to its
        dispatches in order; matching by client alone would be ambiguous."""
        log = textwrap.dedent("""
            [Server #1] Starting round 1/2.
            [Server #1] Selecting client #7 for training.
            [Server #1] Adding client #7 to the list of clients for aggregation.
            [Server #1] Starting round 2/2.
            [Server #1] Selecting client #7 for training.
            [Server #1] Adding client #7 to the list of clients for aggregation.
        """)
        dispatches, merges, _r = cr.read_events(log)
        schedule = cr.replay(dispatches, {7: 2.0}, concurrency=1)
        clocks = cr.clock_per_round(schedule, merges)
        assert clocks[1] == pytest.approx(2.0)
        assert clocks[2] == pytest.approx(4.0)

    def test_a_merge_with_no_dispatch_is_fatal(self):
        dispatches, merges, _r = cr.read_events(LOG)
        schedule = cr.replay(dispatches, TestReplay.SLEEPS, 4)
        with pytest.raises(SystemExit, match="no unconsumed dispatch"):
            cr.clock_per_round(schedule, merges + [(3, 99)])

    def test_verify_merge_order_accepts_an_ascending_run(self):
        dispatches, merges, _r = cr.read_events(LOG)
        schedule = cr.replay(dispatches, TestReplay.SLEEPS, 4)
        ok, problems = cr.verify_merge_order(schedule, merges)
        assert ok and problems == []

    def test_verify_merge_order_reports_an_inversion(self):
        dispatches, merges, _r = cr.read_events(LOG)
        schedule = cr.replay(dispatches, TestReplay.SLEEPS, 4)
        swapped = [merges[1], merges[0], merges[2]]
        ok, problems = cr.verify_merge_order(schedule, swapped)
        assert not ok and problems


class TestMatchedDispatches:
    """The one place dispatch<->merge pairing happens; clock_per_round,
    staleness_per_merge and verify_merge_order all build on this."""

    def test_returns_one_dispatch_per_merge_in_merge_order(self):
        dispatches, merges, _r = cr.read_events(LOG)
        schedule = cr.replay(dispatches, TestReplay.SLEEPS, 4)
        matched = cr.matched_dispatches(schedule, merges)
        assert [d.client_id for d in matched] == [cid for _r, cid in merges]

    def test_raises_the_same_way_clock_per_round_did(self):
        dispatches, merges, _r = cr.read_events(LOG)
        schedule = cr.replay(dispatches, TestReplay.SLEEPS, 4)
        with pytest.raises(SystemExit, match="no unconsumed dispatch"):
            cr.matched_dispatches(schedule, merges + [(3, 99)])


class TestStalenessPerMerge:
    """merge_round - dispatch_round, verified against PLATO's own logged actual
    staleness in `tests/test_clock_reconstruction.py::TestAgainstARealRun` below.

    Every dispatch in `LOG` is round 1 (the pipeline filling C=4 slots at the
    start), so each client's staleness at merge time is exactly the merge's own
    round number minus 1.
    """

    def test_staleness_is_merge_round_minus_dispatch_round(self):
        dispatches, merges, _r = cr.read_events(LOG)
        schedule = cr.replay(dispatches, TestReplay.SLEEPS, 4)
        staleness = cr.staleness_per_merge(schedule, merges)
        # merges = [(1, 18), (2, 20), (3, 15)]; all three were dispatched at round 1.
        assert staleness == [0, 1, 2]

    def test_one_value_per_merge_not_per_round(self):
        """A staleness-bound flush can merge several clients under one round
        number; collapsing into a dict keyed by round would drop all but one."""
        log = textwrap.dedent("""
            [Server #1] Starting round 1/1.
            [Server #1] Selecting client #1 for training.
            [Server #1] Selecting client #2 for training.
            [Server #1] Adding client #1 to the list of clients for aggregation.
            [Server #1] Adding client #2 to the list of clients for aggregation.
        """)
        dispatches, merges, _r = cr.read_events(log)
        schedule = cr.replay(dispatches, {1: 1.0, 2: 2.0}, concurrency=2)
        staleness = cr.staleness_per_merge(schedule, merges)
        assert len(staleness) == 2 == len(merges)

    def test_repeat_dispatches_are_matched_in_order(self):
        """Same scenario as the clock_per_round repeat test: client 7 dispatched
        at round 1 and again at round 2, each merged one round later."""
        log = textwrap.dedent("""
            [Server #1] Starting round 1/2.
            [Server #1] Selecting client #7 for training.
            [Server #1] Starting round 2/2.
            [Server #1] Adding client #7 to the list of clients for aggregation.
            [Server #1] Selecting client #7 for training.
            [Server #1] Starting round 3/2.
            [Server #1] Adding client #7 to the list of clients for aggregation.
        """)
        dispatches, merges, _r = cr.read_events(log)
        schedule = cr.replay(dispatches, {7: 1.0}, concurrency=1)
        staleness = cr.staleness_per_merge(schedule, merges)
        assert staleness == [1, 1]


# ---------------------------------------------------------------------------
# comm_time comes from the config, never from a literal
# ---------------------------------------------------------------------------

class TestCommTime:
    def test_comm_simulation_true_means_zero(self):
        assert cr.comm_time_seconds({"clients": {"comm_simulation": True}}) == 0.0

    def test_comm_simulation_false_is_refused(self):
        """PLATO measures comm_time on the host there, so there is no configured
        value to rebuild from. Assuming zero would be silently wrong."""
        with pytest.raises(SystemExit, match="measures comm_time on the host"):
            cr.comm_time_seconds({"clients": {"comm_simulation": False}})

    def test_an_absent_key_is_refused(self):
        with pytest.raises(SystemExit, match="Refusing to assume zero"):
            cr.comm_time_seconds({"clients": {}})


# ---------------------------------------------------------------------------
# C1, against a real finished run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_run():
    try:
        directory = cr.run_dir("det1")
    except SystemExit:
        pytest.skip("det1 has not been run")
    return cr.reconstruct(directory)


class TestAgainstARealRun:
    def test_every_round_has_both_clocks(self, real_run):
        assert real_run["rows"]
        for row in real_run["rows"]:
            assert row["sim_time_plato"] > 0
            assert row["sim_time_reconstructed"] > 0

    def test_the_gap_is_positive_at_every_round(self, real_run):
        """It is accumulated processing_time, which cannot be negative. A
        negative gap would mean the reconstruction is not reproducing what the
        server did."""
        negative = [(r, g) for r, g in cr.gap_series(real_run) if g < 0]
        assert not negative, negative[:5]

    def test_the_gap_tracks_the_per_slot_report_count(self, real_run):
        """The gap is per-SLOT accumulation: a client's finish carries the host
        overhead of every report that ran in ITS slot before it. This is the
        positive statement that the queue model is right."""
        import statistics
        pending: dict[int, list] = {}
        for item in real_run["schedule"]:
            pending.setdefault(item.client_id, []).append(item)
        for items in pending.values():
            items.reverse()

        plato = {row["round"]: row["sim_time_plato"] for row in real_run["rows"]}
        counts: dict[int, int] = {}
        xs, ys = [], []
        for rnd, cid in real_run["merges"]:
            item = pending[cid].pop()
            n = counts.get(item.slot, 0)
            counts[item.slot] = n + 1
            if rnd in plato:
                xs.append(n)
                ys.append(plato[rnd] - item.finish)

        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs)
               * sum((b - my) ** 2 for b in ys)) ** 0.5
        assert num / den > 0.9

    def test_accuracy_is_untouched(self, real_run):
        """Only the time axis is recomputed; accuracy is indexed by round."""
        with cr.results_csv(real_run["directory"]).open(newline="") as fh:
            original = {int(r["round"]): float(r["accuracy"])
                        for r in csv.DictReader(fh)}
        for row in real_run["rows"]:
            assert row["accuracy"] == original[row["round"]]

    def test_dispatch_and_merge_counts_are_consistent(self, real_run):
        """C dispatches fill the pipeline, then one per merge."""
        assert (len(real_run["dispatches"])
                == len(real_run["merges"]) + real_run["concurrency"] - 1)

    def test_the_csv_carries_both_clocks(self, real_run, tmp_path):
        out = cr.write_curves([real_run], tmp_path / "curves.csv")
        with out.open(newline="") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == [
                "experiment", "arm", "seed", "round",
                "sim_time_plato", "sim_time_reconstructed", "accuracy",
            ]
            assert len(list(reader)) == len(real_run["rows"])

    def test_staleness_matches_platos_own_logged_actual_staleness(self, real_run):
        """The independent check: `staleness_per_merge` is derived purely from
        the dispatch/merge event stream this replay already parses (C3), never
        from PLATO's own staleness text. This cross-checks it against that text
        anyway, for the `gpu` arm, where it exists
        (`[PolarisVi] ... actual=%.4f rounds`,
        `fedasync_polaris_vi_selection.py::on_reports_received`).

        `baseline` never logs that line at all -- it runs
        `fedasync_selection.FedAsyncClientSelectionStrategy`, not Polaris -- which
        is the whole reason this replay-based derivation exists: Table C needs
        one staleness source that works for every arm, and a text-parsing
        approach has no answer for that one.
        """
        text = cr.server_log(real_run["directory"])
        actual_re = re.compile(
            r"cid=(\d+)\s+predicted=[\d.]+ rounds\s+actual=([\d.]+) rounds"
        )
        logged = [(int(cid), float(actual))
                  for cid, actual in actual_re.findall(text)]
        if not logged:
            pytest.skip("det1 was not a Polaris arm; nothing to cross-check")

        mine = cr.staleness_per_merge(real_run["schedule"], real_run["merges"])
        assert len(mine) == len(logged) == len(real_run["merges"])
        for i, ((logged_cid, logged_actual), (merge_round, merge_cid)) in enumerate(
            zip(logged, real_run["merges"])
        ):
            assert merge_cid == logged_cid, i
            assert mine[i] == pytest.approx(logged_actual, abs=1e-9), i
