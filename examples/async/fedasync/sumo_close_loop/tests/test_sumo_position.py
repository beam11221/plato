"""Tests for PositionOracle (Stage 5b, prompts/STAGE5B_fix_Ri.md)."""

from __future__ import annotations

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

from sumo_position import PositionOracle  # noqa: E402

POSITIONS_N20R400 = HERE.parent / "input" / "positions_n20r400.csv"


@pytest.fixture
def oracle() -> PositionOracle:
    """One client crossing +x at a steady 10 m/s, samples 1s apart."""
    return PositionOracle({
        1: [(0.0, 200.0, 1), (1.0, 210.0, 1), (2.0, 220.0, 1), (3.0, 230.0, 1)],
        2: [(0.0, 1000.0, -1), (1.0, 990.0, -1)],
    })


class TestXAt:
    def test_exact_sample(self, oracle):
        assert oracle.x_at(1, 1.0) == 210.0

    def test_interpolates_between_samples(self, oracle):
        assert oracle.x_at(1, 1.5) == pytest.approx(215.0)

    def test_none_before_the_first_sample(self, oracle):
        assert oracle.x_at(1, -1.0) is None

    def test_none_after_the_last_sample(self, oracle):
        assert oracle.x_at(1, 10.0) is None

    def test_none_for_an_unknown_client(self, oracle):
        assert oracle.x_at(99, 1.0) is None

    def test_none_across_a_gap_bigger_than_one_sample_interval(self):
        # Two separate coverage windows for the same client -- 1.0 then a
        # jump to 50.0 is not one continuous span.
        gapped = PositionOracle({1: [(0.0, 200.0, 1), (1.0, 210.0, 1),
                                      (50.0, 700.0, 1), (51.0, 710.0, 1)]})
        assert gapped.x_at(1, 25.0) is None
        assert gapped.x_at(1, 0.5) == pytest.approx(205.0)
        assert gapped.x_at(1, 50.5) == pytest.approx(705.0)


class TestDirectionAt:
    def test_positive_direction(self, oracle):
        assert oracle.direction_at(1, 0.5) == 1

    def test_negative_direction(self, oracle):
        assert oracle.direction_at(2, 0.5) == -1

    def test_none_outside_the_recorded_range(self, oracle):
        assert oracle.direction_at(1, 10.0) is None


@pytest.mark.skipif(
    not POSITIONS_N20R400.exists(), reason="run remap_positions.py for n20r400 first"
)
class TestRealTrace:
    """Sanity checks against the radius-400 trace's own remapped positions."""

    @pytest.fixture(scope="class")
    @staticmethod
    def real() -> PositionOracle:
        return PositionOracle.from_csv(POSITIONS_N20R400)

    def test_fleet_size(self, real):
        assert sorted(real._t) == list(range(1, 21))

    def test_positions_stay_within_the_coverage_span(self, real):
        # Every remapped row came from an in_coverage=1 trace row, so every
        # x_m this oracle can return must fall in [200, 1000].
        for client_id, times in real._t.items():
            for t in times[::137]:  # sample sparsely -- this is 470k rows
                x = real.x_at(client_id, t)
                assert x is None or 200.0 <= x <= 1000.0

    def test_direction_is_always_plus_or_minus_one(self, real):
        for client_id, times in real._t.items():
            for t in times[::137]:
                d = real.direction_at(client_id, t)
                assert d in (1, -1, None)
