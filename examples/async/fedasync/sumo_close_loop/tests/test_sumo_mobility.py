"""Tests for the coverage oracle -- the read-only view of the SUMO trace."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

from sumo_mobility import CoverageOracle  # noqa: E402

COVERAGE_N20 = HERE.parent / "input" / "coverage_n20.csv"


@pytest.fixture
def oracle() -> CoverageOracle:
    """A three-client toy trace with hand-checkable windows."""
    return CoverageOracle(
        {
            1: [(0.0, 10.0), (30.0, 40.0)],
            2: [(5.0, 15.0)],
            3: [(50.0, 60.0)],
        }
    )


class TestInCoverage:
    def test_inside_a_window(self, oracle):
        assert oracle.in_coverage(1, 5.0)

    def test_both_endpoints_are_inside(self, oracle):
        # SUMO sampled the vehicle in range at both instants, so the window is
        # closed at both ends.
        assert oracle.in_coverage(1, 0.0)
        assert oracle.in_coverage(1, 10.0)

    def test_between_two_windows_is_outside(self, oracle):
        assert not oracle.in_coverage(1, 20.0)

    def test_second_window_of_the_same_client(self, oracle):
        assert oracle.in_coverage(1, 35.0)

    def test_before_the_first_window(self, oracle):
        assert not oracle.in_coverage(3, 10.0)

    def test_unknown_client_is_never_in_coverage(self, oracle):
        assert not oracle.in_coverage(99, 5.0)


class TestClientsInCoverage:
    def test_overlapping_windows(self, oracle):
        assert oracle.clients_in_coverage(7.0) == [1, 2]

    def test_returns_empty_when_nobody_is_in_range(self, oracle):
        assert oracle.clients_in_coverage(20.0) == []

    def test_sorted_by_client_id(self, oracle):
        assert oracle.clients_in_coverage(5.0) == sorted(
            oracle.clients_in_coverage(5.0)
        )


class TestNextEntryTime:
    def test_finds_the_next_entry_across_all_clients(self, oracle):
        assert oracle.next_entry_time(20.0) == 30.0

    def test_a_time_that_is_itself_an_entry_returns_that_time(self, oracle):
        assert oracle.next_entry_time(30.0) == 30.0

    def test_returns_none_after_the_last_entry(self, oracle):
        assert oracle.next_entry_time(55.0) is None


class TestWindowEnd:
    def test_returns_the_exit_of_the_current_window(self, oracle):
        assert oracle.window_end(1, 5.0) == 10.0

    def test_returns_none_when_out_of_range(self, oracle):
        assert oracle.window_end(1, 20.0) is None


class TestWindowStart:
    def test_returns_the_entry_of_the_current_window(self, oracle):
        assert oracle.window_start(1, 5.0) == 0.0

    def test_returns_the_entry_of_the_second_window(self, oracle):
        assert oracle.window_start(1, 35.0) == 30.0

    def test_returns_none_when_out_of_range(self, oracle):
        assert oracle.window_start(1, 20.0) is None

    def test_returns_none_for_an_unknown_client(self, oracle):
        assert oracle.window_start(99, 5.0) is None

    def test_at_the_entry_instant_itself(self, oracle):
        assert oracle.window_start(1, 0.0) == 0.0


class TestWrapAround:
    def test_times_within_the_trace_pass_through(self, oracle):
        assert oracle.normalize(30.0) == 30.0

    def test_times_past_the_end_wrap(self, oracle):
        # t_end is 60.0, so 65 s into a longer run is 5 s into the next loop.
        assert oracle.normalize(65.0) == pytest.approx(5.0)

    def test_coverage_is_evaluated_on_the_wrapped_time(self, oracle):
        assert oracle.in_coverage(1, 65.0) is oracle.in_coverage(1, 5.0)


@pytest.fixture(scope="module")
def real() -> CoverageOracle:
    return CoverageOracle.from_csv(COVERAGE_N20)


@pytest.mark.skipif(
    not COVERAGE_N20.exists(), reason="run build_inputs.sh n20 first"
)
class TestRealTrace:
    """Sanity checks against the numbers measured from the n20 trace."""

    def test_fleet_size(self, real):
        assert real.client_ids == list(range(1, 21))

    def test_trace_covers_ten_hours(self, real):
        assert real.t_end == pytest.approx(36000.0, abs=100.0)

    def test_about_half_the_fleet_is_in_range_at_any_time(self, real):
        sampled = [len(real.clients_in_coverage(t)) for t in range(0, 36000, 137)]
        mean = sum(sampled) / len(sampled)
        # Measured over the whole trace: 9.83 of 20.
        assert 9.0 < mean < 11.0

    def test_the_oracle_agrees_with_the_source_csv(self, real):
        """Spot-check the first rows straight out of the file."""
        with COVERAGE_N20.open(newline="") as f:
            rows = [next(csv.DictReader(f)) for _ in range(1)]
        for row in rows:
            client_id = int(row["client_id"])
            enter, exit_ = float(row["t_enter_s"]), float(row["t_exit_s"])
            midpoint = (enter + exit_) / 2
            assert real.in_coverage(client_id, midpoint)
            assert real.window_end(client_id, midpoint) == exit_
            assert real.window_start(client_id, midpoint) == enter
