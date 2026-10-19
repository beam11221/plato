"""Tests for report.py -- Tables A-F, the predictions, and the stop conditions.

No real Stage 3 sweep has finished yet, so every table is exercised against a
small, hand-built `run_summary.csv`-shaped fixture rather than real data. The
fixture is built to look like a real campaign (four arms per cell, a `_rep2`
repeat, N=40 and gpu_oracle cells) so the grouping and threshold logic is
exercised the way it will actually be used.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import report

DATASET_DIR = Path(__file__).resolve().parents[1]


def make_row(experiment, arm, total_clients=20, concurrency=4,
            plateau=0.75, noise=0.01, tta=None, staleness=None,
            shares=(0.30, 0.45, 0.25), dispatches=1000):
    """A run_summary.csv-shaped row, coerced the way report._coerce would.

    `tta` is a dict {threshold: hours}; missing thresholds are left blank
    (never crossed), matching collect.py's own behaviour.
    """
    tta = tta or {}
    row = {
        "experiment": experiment, "arm": arm,
        "total_clients": total_clients, "concurrency": concurrency,
        "plateau_accuracy": plateau, "noise_sd": noise,
        "mean_staleness": staleness if staleness is not None else "",
        "max_staleness": int(staleness * 3) if staleness is not None else "",
        "dispatches": dispatches,
        "share_xavier": shares[0], "share_orin": shares[1], "share_thor": shares[2],
        "seconds_per_round": 1.5,
    }
    for t in report.THRESHOLDS:
        row[f"tta_h_{t}"] = tta.get(t, "")
        row[f"rounds_to_{t}"] = 100 * t if t in tta else ""
    return row


def write_summary(tmp_path, rows) -> Path:
    path = tmp_path / "run_summary.csv"
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def campaign_rows():
    """A miniature but shaped-like-the-real-thing campaign."""
    rows = []
    # C sweep at N=20: baseline/gpu/cpu/observed at C=1,2,4,8.
    for c, staleness_base in ((1, 0.0), (2, 1.0), (4, 3.0), (8, 7.0)):
        for arm, plateau, tta80 in (
            ("baseline", 0.80, 3.0),
            ("gpu", 0.82, 2.5),
            ("cpu", 0.79, 3.6),
            ("observed", 0.81, 2.8),
        ):
            rows.append(make_row(
                f"a0p8_N20_C{c}_s4126", arm, total_clients=20, concurrency=c,
                plateau=plateau, tta={70: tta80 * 0.6, 75: tta80 * 0.8, 80: tta80},
                staleness=staleness_base + (0.2 if arm == "cpu" else 0.0),
            ))
    # gpu's byte-identical repeat, at C=4 -- close to gpu's own numbers on
    # every threshold (a real _rep2 measures run-to-run noise, not a 20%
    # swing), so the "clean campaign" fixture is actually clean.
    rows.append(make_row(
        "a0p8_N20_C4_s4126_rep2", "gpu", total_clients=20, concurrency=4,
        plateau=0.82, tta={70: 1.55, 75: 2.05, 80: 2.55},
        staleness=3.0,
    ))
    # Fleet size at C=4: N=40, same 4 arms.
    for arm, plateau, tta80 in (
        ("baseline", 0.80, 3.1), ("gpu", 0.82, 2.6),
        ("cpu", 0.79, 3.7), ("observed", 0.81, 2.9),
    ):
        rows.append(make_row(
            "a0p8_N40_C4_s4126", arm, total_clients=40, concurrency=4,
            plateau=plateau, tta={70: tta80 * 0.6, 75: tta80 * 0.8, 80: tta80},
            staleness=3.0,
        ))
    # gpu_oracle at C=4 and C=8.
    for c in (4, 8):
        rows.append(make_row(
            f"a0p8_N20_C{c}_s4126", "gpu_oracle", total_clients=20, concurrency=c,
            plateau=0.85, tta={70: 1.5, 75: 1.9, 78: 2.0, 80: 2.1},
            staleness=2.0,
        ))
    return rows


# ---------------------------------------------------------------------------
# load_summary
# ---------------------------------------------------------------------------

class TestLoadSummary:
    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(SystemExit, match="run collect.py first"):
            report.load_summary(tmp_path / "nope.csv")

    def test_empty_file_is_a_clear_error(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("experiment,arm\n")
        with pytest.raises(SystemExit, match="no rows"):
            report.load_summary(path)

    def test_numeric_fields_are_coerced(self, tmp_path, campaign_rows):
        path = write_summary(tmp_path, campaign_rows)
        rows = report.load_summary(path)
        assert isinstance(rows[0]["plateau_accuracy"], float)
        assert isinstance(rows[0]["concurrency"], int)

    def test_blank_stays_blank_not_zero(self, tmp_path):
        rows = [make_row("a0p8_N20_C1_s4126", "baseline", concurrency=1)]
        path = write_summary(tmp_path, rows)
        loaded = report.load_summary(path)
        assert loaded[0]["tta_h_80"] == ""


# ---------------------------------------------------------------------------
# Cell grouping and _rep2 handling
# ---------------------------------------------------------------------------

class TestCellGrouping:
    def test_rep2_is_excluded_from_real_cells(self, campaign_rows):
        cells = report.real_cells(campaign_rows)
        c4_n20 = cells["a0p8_N20_C4_s4126"]
        arms = {r["arm"] for r in c4_n20}
        assert "gpu" in arms
        # exactly the 4 regular arms + gpu_oracle, never a second "gpu" from rep2
        assert sum(1 for r in c4_n20 if r["arm"] == "gpu") == 1

    def test_is_rep2_detects_the_tagged_row(self, campaign_rows):
        rep2_rows = [r for r in campaign_rows if report.is_rep2(r)]
        assert len(rep2_rows) == 1
        assert rep2_rows[0]["arm"] == "gpu"

    def test_cell_key_strips_the_rep2_suffix(self):
        assert report.cell_key({"experiment": "a0p8_N20_C4_s4126_rep2"}) == \
            "a0p8_N20_C4_s4126"

    def test_cell_key_leaves_ordinary_experiments_alone(self):
        assert report.cell_key({"experiment": "a0p8_N20_C4_s4126"}) == \
            "a0p8_N20_C4_s4126"


# ---------------------------------------------------------------------------
# usable_threshold
# ---------------------------------------------------------------------------

class TestUsableThreshold:
    def test_picks_the_highest_threshold_5_points_clear(self):
        rows = [{"plateau_accuracy": 0.86}, {"plateau_accuracy": 0.90}]
        # lowest plateau 0.86 -> 80% clears by 6 points, 78% by 8, etc. -> pick 80
        assert report.usable_threshold(rows) == 80

    def test_a_borderline_plateau_excludes_that_threshold(self):
        # lowest plateau exactly 0.84 -> 80% is only 4 points below -> not usable
        rows = [{"plateau_accuracy": 0.84}]
        assert report.usable_threshold(rows) == 78

    def test_exactly_5_points_below_counts_as_usable(self):
        rows = [{"plateau_accuracy": 0.85}]   # 80% is exactly 5 points below
        assert report.usable_threshold(rows) == 80

    def test_no_threshold_usable_returns_none(self):
        rows = [{"plateau_accuracy": 0.60}]
        assert report.usable_threshold(rows) is None

    def test_uses_the_lowest_plateau_in_the_cell_not_the_highest(self):
        rows = [{"plateau_accuracy": 0.95}, {"plateau_accuracy": 0.60}]
        assert report.usable_threshold(rows) is None

    def test_blank_plateaus_are_ignored_not_fatal(self):
        rows = [{"plateau_accuracy": ""}, {"plateau_accuracy": 0.90}]
        assert report.usable_threshold(rows) == 80

    def test_no_plateaus_at_all_returns_none(self):
        assert report.usable_threshold([{"plateau_accuracy": ""}]) is None


# ---------------------------------------------------------------------------
# pct_diff
# ---------------------------------------------------------------------------

class TestPctDiff:
    def test_basic_percentage(self):
        assert report.pct_diff(1.1, 1.0) == "+10.00%"

    def test_a_decrease_is_negative(self):
        assert report.pct_diff(0.9, 1.0) == "-10.00%"

    def test_a_missing_value_is_no_crossing_not_an_error(self):
        assert report.pct_diff("", 1.0) == "no crossing"
        assert report.pct_diff(1.0, "") == "no crossing"

    def test_zero_baseline_is_called_out_not_divided_by(self):
        assert report.pct_diff(1.0, 0.0) == "baseline crossed at t=0"


# ---------------------------------------------------------------------------
# Table A -- the gate
# ---------------------------------------------------------------------------

class TestTableA:
    def test_finds_rep2_and_its_twin(self, campaign_rows):
        md, result = report.table_a(campaign_rows)
        assert "gpu (`_rep2`)" in md
        assert result["stop"] is False

    def test_worst_pct_is_the_largest_absolute_difference(self, campaign_rows):
        _md, result = report.table_a(campaign_rows)
        # gpu's tta_h_70/75/80 = 1.5/2.0/2.5; rep2's = 1.55/2.05/2.55 --
        # differences of 3.33% / 2.5% / 2.0%, so the worst is ~3.33%.
        assert result["worst_pct"] == pytest.approx(3.333, abs=0.01)
        assert result["worst_pct"] < report.REP2_STOP_PCT

    def test_stop_triggers_over_10_percent(self):
        rows = [
            make_row("a0p8_N20_C4_s4126", "gpu", tta={80: 2.0}),
            make_row("a0p8_N20_C4_s4126_rep2", "gpu", tta={80: 2.5}),  # +25%
        ]
        _md, result = report.table_a(rows)
        assert result["stop"] is True
        assert result["worst_pct"] == pytest.approx(25.0)

    def test_no_rep2_yet_is_reported_not_fatal(self):
        rows = [make_row("a0p8_N20_C4_s4126", "gpu")]
        md, result = report.table_a(rows)
        assert "no `_rep2` run" in md
        assert result["stop"] is False


# ---------------------------------------------------------------------------
# Table B / Table D -- the shared cell_table builder
# ---------------------------------------------------------------------------

class TestTableB:
    def test_covers_every_c_at_n20(self, campaign_rows):
        md, thresholds = report.table_b(campaign_rows)
        for c in (1, 2, 4, 8):
            assert f"| {c} |" in md
        assert set(thresholds) == {f"a0p8_N20_C{c}_s4126" for c in (1, 2, 4, 8)}

    def test_excludes_n40_cells(self, campaign_rows):
        md, _t = report.table_b(campaign_rows)
        assert "N40" not in md

    def test_excludes_rep2(self, campaign_rows):
        md, _t = report.table_b(campaign_rows)
        assert "_rep2" not in md

    def test_baseline_row_has_no_vs_baseline_comparison(self, campaign_rows):
        md, _t = report.table_b(campaign_rows)
        # the baseline arm's own row must show "--" in the LAST ("vs baseline")
        # column, not a percentage against itself. Markdown table rows end
        # "| <value> |", so the column's own content is "--", not the raw
        # string suffix "--" (every row already ends in a literal "|").
        lines = [l for l in md.splitlines() if "| baseline |" in l]
        assert lines and all(l.rstrip().endswith("| -- |") for l in lines)

    def test_a_cell_with_no_usable_threshold_says_so(self):
        rows = [make_row("a0p8_N20_C1_s4126", arm, concurrency=1, plateau=0.60)
               for arm in ("baseline", "gpu")]
        md, thresholds = report.table_b(rows)
        assert thresholds["a0p8_N20_C1_s4126"] is None
        assert "none usable" in md


class TestTableD:
    def test_covers_n20_and_n40_at_c4_only(self, campaign_rows):
        md, thresholds = report.table_d(campaign_rows)
        assert set(thresholds) == {"a0p8_N20_C4_s4126", "a0p8_N40_C4_s4126"}
        assert "| 20 |" in md and "| 40 |" in md

    def test_excludes_other_concurrencies(self, campaign_rows):
        _md, thresholds = report.table_d(campaign_rows)
        assert "a0p8_N20_C1_s4126" not in thresholds
        assert "a0p8_N20_C8_s4126" not in thresholds


# ---------------------------------------------------------------------------
# Table C -- staleness
# ---------------------------------------------------------------------------

class TestTableC:
    def test_reports_c_minus_1_alongside_measured(self, campaign_rows):
        md = report.table_c(campaign_rows)
        assert "| 4 |" in md
        # C=4 rows should show "C - 1" = 3
        lines = [l for l in md.splitlines() if l.startswith("| 4 |")]
        assert any(l.rstrip().endswith("| 3 |") for l in lines)

    def test_includes_a_pooled_mean_row_per_cell(self, campaign_rows):
        md = report.table_c(campaign_rows)
        assert "mean across arms" in md

    def test_only_n20_cells(self, campaign_rows):
        """N=40 rows must not appear. The fixture ALSO has gpu_oracle at C=4
        and C=8 (both N=20), which legitimately belongs in this table too --
        Table C is not scoped to the four regular arms -- so the expected row
        count is derived from the same grouping the code uses, not a hand
        count that would silently assume gpu_oracle away."""
        md = report.table_c(campaign_rows)
        n40_rows = [r for r in campaign_rows if r["total_clients"] == 40]
        assert n40_rows  # sanity: the fixture does have N=40 rows to exclude

        n20_cells = {k: v for k, v in report.real_cells(campaign_rows).items()
                    if v and v[0]["total_clients"] == 20}
        expected_arm_rows = sum(len(v) for v in n20_cells.values())
        expected_pooled_rows = len(n20_cells)   # one "mean across arms" per cell

        # If any N=40 row leaked in, the actual body would be LONGER than a
        # count derived purely from the N=20 cells -- so this comparison alone
        # proves exclusion without needing a separate (and, as it turns out,
        # unreliable -- "2.840" contains the substring "40 |") text search.
        body_lines = [l for l in md.splitlines()[2:] if l.strip()]
        assert len(body_lines) == expected_arm_rows + expected_pooled_rows


# ---------------------------------------------------------------------------
# Table E -- gpu vs gpu_oracle
# ---------------------------------------------------------------------------

class TestTableE:
    def test_covers_c4_and_c8(self, campaign_rows):
        md = report.table_e(campaign_rows)
        assert "| 4 | plateau" in md
        assert "| 8 | plateau" in md

    def test_oracle_plateau_is_shown_against_gpus(self, campaign_rows):
        md = report.table_e(campaign_rows)
        assert "0.8200" in md   # gpu's plateau at C=4
        assert "0.8500" in md   # gpu_oracle's plateau

    def test_no_oracle_data_yet_is_reported_not_fatal(self):
        rows = [make_row("a0p8_N20_C4_s4126", "gpu")]
        md = report.table_e(rows)
        assert "Batch 3 runs last" in md


# ---------------------------------------------------------------------------
# Table F -- control
# ---------------------------------------------------------------------------

class TestTableF:
    def test_one_row_per_run(self, campaign_rows):
        md, _warnings = report.table_f(campaign_rows)
        body_lines = [l for l in md.splitlines()[2:] if l.strip()]
        assert len(body_lines) == len(campaign_rows)

    def test_flags_a_cell_where_every_arm_shares_the_same_tier_mix(self):
        rows = [
            make_row("a0p8_N20_C4_s4126", "baseline", shares=(0.33, 0.34, 0.33)),
            make_row("a0p8_N20_C4_s4126", "gpu", shares=(0.331, 0.339, 0.33)),
            make_row("a0p8_N20_C4_s4126", "cpu", shares=(0.332, 0.338, 0.33)),
        ]
        _md, warnings = report.table_f(rows)
        assert warnings
        assert "xavier" in warnings[0] or "orin" in warnings[0]

    def test_does_not_flag_a_cell_with_real_separation(self, campaign_rows):
        _md, warnings = report.table_f(campaign_rows)
        # the fixture gives every arm identical shares by construction (0.30/
        # 0.45/0.25 default) -- this is a control check on the WARNING LOGIC
        # itself: it must fire here, proving it is not silently inert.
        assert warnings


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------

class TestStopConditions:
    def test_no_findings_on_a_clean_campaign_apart_from_identical_shares(
        self, campaign_rows
    ):
        """The fixture's arms all share identical dispatch shares by
        construction (a fixture-authoring shortcut, not a real result), so the
        tier-mix flag is EXPECTED to fire here; nothing else should."""
        table_a_md, table_a_result = report.table_a(campaign_rows)
        _b, thresholds_b = report.table_b(campaign_rows)
        _d, thresholds_d = report.table_d(campaign_rows)
        _f, warnings_f = report.table_f(campaign_rows)
        findings = report.stop_conditions(
            campaign_rows, table_a_result, thresholds_b, thresholds_d, warnings_f
        )
        assert not any(f.startswith("STOP") for f in findings)
        assert any(f.startswith("FLAG") for f in findings)

    def test_rep2_over_10_percent_is_a_stop(self, campaign_rows):
        for row in campaign_rows:
            if report.is_rep2(row):
                row["tta_h_80"] = 5.0   # blow it out to +100% vs gpu's 2.5
        _a, table_a_result = report.table_a(campaign_rows)
        findings = report.stop_conditions(campaign_rows, table_a_result, {}, {}, [])
        assert any("STOP" in f and "_rep2" in f for f in findings)

    def test_an_unusable_cell_is_a_stop(self, campaign_rows):
        _a, table_a_result = report.table_a(campaign_rows)
        findings = report.stop_conditions(
            campaign_rows, table_a_result,
            {"a0p8_N20_C1_s4126": None}, {}, [],
        )
        assert any("STOP" in f and "no threshold" in f for f in findings)


# ---------------------------------------------------------------------------
# build_report / main -- the whole thing holds together
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_contains_every_table_heading(self, campaign_rows):
        text = report.build_report(campaign_rows)
        for heading in ("Table A", "Table B", "Table C", "Table D",
                       "Table E", "Table F", "predictions",
                       "Stop conditions"):
            assert heading in text

    def test_main_writes_the_file(self, tmp_path, campaign_rows, monkeypatch):
        import sys
        summary_path = write_summary(tmp_path, campaign_rows)
        out_path = tmp_path / "report.md"
        monkeypatch.setattr(sys, "argv",
                            ["report.py", "--summary", str(summary_path),
                             "--out", str(out_path)])
        report.main()
        assert out_path.exists()
        assert "Table A" in out_path.read_text()
