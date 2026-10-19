"""Tests for make_fleet.py -- the fleet COMPOSITION generator.

The load-bearing test is `test_n20_reproduces_separability_fleet`: the whole
point of moving this generator into `gpu_tiers_dataset/` is that it is the same
generator, and a fleet that changed byte for byte during the move would silently
invalidate the comparison between the oracle branch and everything after it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import make_fleet

FEDASYNC = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# counts_for -- the remainder rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "total,mix,expected",
    [
        (20, (50, 35, 15), (10, 7, 3)),    # divides evenly
        (40, (50, 35, 15), (20, 14, 6)),   # divides evenly
        (100, (50, 35, 15), (50, 35, 15)), # divides evenly
        (30, (50, 35, 15), (15, 10, 5)),   # 15 / 10.5 / 4.5 -> remainder to thor
        (10, (50, 35, 15), (5, 3, 2)),     # 5 / 3.5 / 1.5   -> remainder to thor
    ],
)
def test_counts_for(total, mix, expected):
    assert make_fleet.counts_for(total, mix) == expected


def test_counts_always_sum_to_total():
    for total in range(1, 201):
        counts = make_fleet.counts_for(total, (50, 35, 15))
        assert sum(counts) == total, total


def test_remainder_goes_to_the_fastest_tier():
    """30 at 50/35/15 asks for 4.5 thor; thor must get 5, not 4."""
    xavier, orin, thor = make_fleet.counts_for(30, (50, 35, 15))
    assert (xavier, orin) == (15, 10)      # floor for every tier but the last
    assert thor == 5                       # ...and the remainder to the fastest
    assert thor > 30 * 15 / 100            # strictly more than the exact share


def test_mix_must_sum_to_100():
    with pytest.raises(SystemExit):
        make_fleet.counts_for(20, (50, 35, 10))


def test_negative_mix_entry_refused():
    with pytest.raises(SystemExit):
        make_fleet.counts_for(20, (60, 50, -10))


def test_non_positive_total_refused():
    with pytest.raises(SystemExit):
        make_fleet.counts_for(0, (50, 35, 15))


def test_divides_evenly():
    assert make_fleet.divides_evenly(20, (50, 35, 15))
    assert make_fleet.divides_evenly(40, (50, 35, 15))
    assert not make_fleet.divides_evenly(30, (50, 35, 15))


# ---------------------------------------------------------------------------
# fleet_tag -- the name the fleet is stored under
# ---------------------------------------------------------------------------

def test_fleet_tag_names_counts_not_percentages():
    assert make_fleet.fleet_tag((10, 7, 3)) == "n20_x10-o7-t3"
    assert make_fleet.fleet_tag((15, 10, 5)) == "n30_x15-o10-t5"
    assert make_fleet.fleet_tag((20, 14, 6)) == "n40_x20-o14-t6"


def test_fleet_tag_distinguishes_mixes_that_round_alike():
    """15/10/5 and 15/11/4 both round to 50/33-37/13-17; the tags must differ."""
    assert make_fleet.fleet_tag((15, 10, 5)) != make_fleet.fleet_tag((15, 11, 4))


def test_fleet_tag_has_no_dot():
    """Dots read as file extensions in run directories (separability precedent)."""
    for counts in ((10, 7, 3), (15, 10, 5), (20, 14, 6)):
        assert "." not in make_fleet.fleet_tag(counts)


# ---------------------------------------------------------------------------
# rows_for -- contiguous blocks, slowest first
# ---------------------------------------------------------------------------

def test_rows_are_contiguous_blocks_slowest_first():
    rows = make_fleet.rows_for((3, 2, 1))
    assert [r["client_id"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert [r["tier_id"] for r in rows] == [
        "xavier", "xavier", "xavier", "orin", "orin", "thor",
    ]


def test_fleets_are_nested_across_sizes():
    """Every thor client at N=20 is still thor at N=40 -- block layout property."""
    small = {r["client_id"]: r["tier_id"]
             for r in make_fleet.rows_for(make_fleet.counts_for(20, (50, 35, 15)))}
    large = {r["client_id"]: r["tier_id"]
             for r in make_fleet.rows_for(make_fleet.counts_for(40, (50, 35, 15)))}
    # IDs 1-10 are xavier in both; the blocks grow rather than reshuffle.
    for cid in range(1, 11):
        assert small[cid] == large[cid] == "xavier"


def test_schema_version_is_written():
    rows = make_fleet.rows_for((1, 1, 1))
    assert {r["schema_version"] for r in rows} == {make_fleet.SCHEMA_VERSION}


# ---------------------------------------------------------------------------
# The move must not have changed the fleet
# ---------------------------------------------------------------------------

def test_n20_reproduces_separability_fleet(tmp_path):
    """Byte-identical to the fleet the oracle-branch runs actually used.

    `../separability/input/clients_n20.csv` was written by
    `../surgical/make_fleet.py`. If this generator produces anything else, every
    result produced after the move sits on a different fleet than the results it
    is meant to be compared against, and nothing in a run log would say so.
    """
    reference = FEDASYNC / "separability" / "input" / "clients_n20.csv"
    if not reference.exists():
        pytest.skip(f"{reference} not present")

    out = tmp_path / "clients.csv"
    make_fleet.write_fleet_csv(
        out, make_fleet.rows_for(make_fleet.counts_for(20, (50, 35, 15)))
    )
    assert out.read_bytes() == reference.read_bytes()


def test_n100_reproduces_surgical_fleet(tmp_path):
    """Same check at the other fleet size the project has on disk."""
    reference = FEDASYNC / "surgical" / "input" / "clients_n100.csv"
    if not reference.exists():
        pytest.skip(f"{reference} not present")

    out = tmp_path / "clients.csv"
    make_fleet.write_fleet_csv(
        out, make_fleet.rows_for(make_fleet.counts_for(100, (50, 35, 15)))
    )
    assert out.read_bytes() == reference.read_bytes()


# ---------------------------------------------------------------------------
# write_fleet_csv
# ---------------------------------------------------------------------------

def test_write_creates_parent_directories(tmp_path):
    out = tmp_path / "deep" / "deeper" / "clients.csv"
    make_fleet.write_fleet_csv(out, make_fleet.rows_for((1, 1, 1)))
    assert out.exists()


def test_written_csv_round_trips(tmp_path):
    out = tmp_path / "clients.csv"
    counts = make_fleet.counts_for(30, (50, 35, 15))
    make_fleet.write_fleet_csv(out, make_fleet.rows_for(counts))
    with out.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 30
    assert [r["tier_id"] for r in rows].count("thor") == 5
    assert [int(r["client_id"]) for r in rows] == list(range(1, 31))
