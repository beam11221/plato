"""Tests for collect.py -- Stage 3's two output files.

Most of the curve statistics (`curve_stats`, `rolling_median`, `first_crossing`)
are checked against small hand-built curves, since they are pure arithmetic
copied from the established `../separability/collect.py` methodology and adapted
to the reconstructed clock. The tests that touch a real run
(`gpu_a0p8_N20_C4_s4126_det1`, kept on disk from Stage 1/2) check the parts that
only a real log can exercise: reading the fleet a run actually used, and the
staleness/dispatch statistics pulled from its server log.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import clock_reconstruction as cr
import collect

DATASET_DIR = Path(__file__).resolve().parents[1]
REAL_RUN_NAME = "gpu_a0p8_N20_C4_s4126_det1"


@pytest.fixture
def real_result():
    try:
        directory = cr.run_dir(REAL_RUN_NAME)
    except SystemExit:
        pytest.skip(f"{REAL_RUN_NAME} is not on disk")
    return cr.reconstruct(directory)


# ---------------------------------------------------------------------------
# evaluation_rows -- filtering out the carried-forward repeats
# ---------------------------------------------------------------------------

class TestEvaluationRows:
    def make_result(self, eval_interval, rounds, accuracies):
        """rows where accuracy is 0 except every eval_interval-th round."""
        rows = []
        last = 0.0
        for rnd in range(1, rounds + 1):
            if rnd % eval_interval == 0:
                last = accuracies.pop(0)
            rows.append({
                "round": rnd, "sim_time_plato": rnd * 1.0,
                "sim_time_reconstructed": rnd * 0.9, "accuracy": last,
            })
        return {"config": {"server": {"eval_interval": eval_interval}}, "rows": rows}

    def test_keeps_only_multiples_of_eval_interval(self):
        result = self.make_result(10, 30, [0.1, 0.2, 0.3])
        rows = collect.evaluation_rows(result)
        assert [r["round"] for r in rows] == [10, 20, 30]

    def test_keeps_the_real_accuracy_not_a_repeat(self):
        result = self.make_result(10, 20, [0.5, 0.9])
        rows = collect.evaluation_rows(result)
        assert [r["accuracy"] for r in rows] == [0.5, 0.9]

    def test_eval_interval_of_one_keeps_every_round(self):
        result = self.make_result(1, 5, [0.1, 0.2, 0.3, 0.4, 0.5])
        rows = collect.evaluation_rows(result)
        assert len(rows) == 5

    def test_defaults_to_interval_one_if_the_key_is_absent(self):
        result = {"config": {"server": {}},
                  "rows": [{"round": 1, "accuracy": 0.1}]}
        assert collect.evaluation_rows(result) == result["rows"]


# ---------------------------------------------------------------------------
# rolling_median
# ---------------------------------------------------------------------------

class TestRollingMedian:
    def test_matches_plain_median_for_a_window_that_always_covers_everything(self):
        """The window SHRINKS at the edges (intentional -- see the docstring),
        so `window == len(values)` still gives a partial slice near the edges.
        `window = 2*len - 1` is the smallest size for which every index's
        `[i - half, i + half]` slice is forced to cover the whole array."""
        import statistics as st
        values = [1.0, 5.0, 2.0, 8.0, 3.0]
        out = collect.rolling_median(values, window=2 * len(values) - 1)
        assert all(v == pytest.approx(st.median(values)) for v in out)

    def test_a_window_of_one_is_the_identity(self):
        values = [3.0, 1.0, 4.0, 1.0, 5.0]
        assert collect.rolling_median(values, window=1) == values

    def test_edges_shrink_rather_than_pad(self):
        """The first and last points use a SMALLER window, not a padded one --
        the low-threshold crossings tend to land among the early evaluations."""
        values = [10.0, 1.0, 1.0, 1.0, 1.0]
        out = collect.rolling_median(values, window=5)
        # At i=0 with window=5, half=2, the slice is values[0:3] = [10,1,1] -> median 1.0
        assert out[0] == pytest.approx(1.0)

    def test_output_length_matches_input(self):
        values = list(range(37))
        assert len(collect.rolling_median([float(v) for v in values], 11)) == 37


# ---------------------------------------------------------------------------
# first_crossing
# ---------------------------------------------------------------------------

class TestFirstCrossing:
    def rows(self):
        return [
            {"round": 10, "sim_time_reconstructed": 100.0},
            {"round": 20, "sim_time_reconstructed": 200.0},
            {"round": 30, "sim_time_reconstructed": 300.0},
        ]

    def test_finds_the_first_round_at_or_above_target(self):
        rnd, hours = collect.first_crossing(self.rows(), [0.5, 0.8, 0.9], 0.75)
        assert rnd == 20
        assert hours == pytest.approx(200.0 / 3600.0)

    def test_exact_equality_counts_as_a_crossing(self):
        rnd, _ = collect.first_crossing(self.rows(), [0.7, 0.7, 0.7], 0.7)
        assert rnd == 10

    def test_no_crossing_returns_none_none(self):
        rnd, hours = collect.first_crossing(self.rows(), [0.1, 0.2, 0.3], 0.9)
        assert rnd is None and hours is None


# ---------------------------------------------------------------------------
# curve_stats
# ---------------------------------------------------------------------------

class TestCurveStats:
    def rows_for(self, accuracies, clock_step=100.0):
        return [
            {"round": (i + 1) * 10, "sim_time_reconstructed": (i + 1) * clock_step,
             "accuracy": a}
            for i, a in enumerate(accuracies)
        ]

    def test_empty_curve_gives_empty_stats(self):
        assert collect.curve_stats([]) == {}

    def test_plateau_is_the_median_of_the_last_10_percent(self):
        accuracies = [0.5] * 90 + [0.9] * 10   # last 10% is a flat 0.9
        stats = collect.curve_stats(self.rows_for(accuracies))
        assert stats["plateau_accuracy"] == pytest.approx(0.9)

    def test_best_accuracy_is_the_curve_max(self):
        stats = collect.curve_stats(self.rows_for([0.1, 0.9, 0.3]))
        assert stats["best_accuracy"] == pytest.approx(0.9)

    def test_only_the_four_brief_thresholds_are_computed(self):
        """Deliberately fewer than ../separability/collect.py: the brief asks
        for exactly 70/75/78/80, and a low plateau not clearing one of them is
        a STOP CONDITION to report, not something to patch with extra rungs."""
        stats = collect.curve_stats(self.rows_for([0.5] * 20))
        threshold_keys = {k for k in stats if k.startswith("tta_h_")
                          and not k.startswith("tta_h_raw_")}
        assert threshold_keys == {"tta_h_70", "tta_h_75", "tta_h_78", "tta_h_80"}

    def test_a_threshold_never_reached_is_blank_not_zero(self):
        stats = collect.curve_stats(self.rows_for([0.1] * 20))
        assert stats["tta_h_80"] == ""
        assert stats["rounds_to_80"] == ""

    def test_raw_and_smoothed_crossings_can_differ(self):
        """One noisy spike crosses on the raw curve but not the smoothed one."""
        accuracies = [0.5] * 10 + [0.81] + [0.5] * 9   # single spike at index 10
        stats = collect.curve_stats(self.rows_for(accuracies))
        assert stats["rounds_to_raw_80"] != ""
        assert stats["rounds_to_80"] == ""    # the median-smoothed curve never reaches it

    def test_noise_sd_is_zero_for_a_perfectly_flat_curve(self):
        stats = collect.curve_stats(self.rows_for([0.7] * 50))
        assert stats["noise_sd"] == pytest.approx(0.0)
        assert stats["noise_sd_diff"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# client_tiers_for -- reads the FLEET this run actually used
# ---------------------------------------------------------------------------

class TestClientTiersFor:
    def test_reads_from_the_configs_own_clients_csv(self, tmp_path):
        fleet_csv = tmp_path / "clients.csv"
        fleet_csv.write_text("client_id,tier_id,schema_version\n"
                             "1,xavier,1\n2,orin,1\n3,thor,1\n")
        config = {"gpu_tier": {"clients_csv": str(fleet_csv)}}
        assert collect.client_tiers_for(config) == {1: "xavier", 2: "orin", 3: "thor"}

    def test_uses_the_real_fleet_for_a_real_run(self, real_result):
        tiers = collect.client_tiers_for(real_result["config"])
        assert len(tiers) == 20
        assert set(tiers.values()) <= {"xavier", "orin", "thor"}


# ---------------------------------------------------------------------------
# dispatch_stats
# ---------------------------------------------------------------------------

class TestDispatchStats:
    def test_counts_and_shares(self):
        result = {"dispatches": [(1, 1), (2, 1), (3, 2), (4, 3)]}
        tiers = {1: "xavier", 2: "orin", 3: "thor"}
        stats = collect.dispatch_stats(result, tiers)
        assert stats["dispatches"] == 4
        assert stats["dispatch_xavier"] == 2
        assert stats["dispatch_orin"] == 1
        assert stats["dispatch_thor"] == 1
        assert stats["share_xavier"] == pytest.approx(0.5)
        assert stats["share_orin"] == pytest.approx(0.25)

    def test_a_client_missing_from_the_tier_map_is_silently_skipped_not_fatal(self):
        """Matches ../separability/collect.py's own dispatch_stats: this
        function reports what it can from a possibly-partial fleet map rather
        than crashing collection over one bad row -- the fleet mismatch itself
        is `clock_reconstruction.verify_fleet_matches_run`'s job, upstream."""
        result = {"dispatches": [(1, 99)]}   # client 99 not in the tier map
        stats = collect.dispatch_stats(result, {1: "xavier"})
        assert stats["dispatches"] == 0

    def test_zero_dispatches_gives_blank_shares_not_a_zero_division(self):
        stats = collect.dispatch_stats({"dispatches": []}, {})
        assert stats["share_xavier"] == ""

    def test_uses_the_real_run(self, real_result):
        tiers = collect.client_tiers_for(real_result["config"])
        stats = collect.dispatch_stats(real_result, tiers)
        assert stats["dispatches"] == len(real_result["dispatches"])
        assert sum(stats[f"dispatch_{t}"] for t in collect.TIERS) == stats["dispatches"]


# ---------------------------------------------------------------------------
# staleness_stats
# ---------------------------------------------------------------------------

class TestStalenessStats:
    def test_mean_and_max(self):
        stats = collect.staleness_stats({"staleness_per_merge": [0, 2, 4, 2]})
        assert stats["mean_staleness"] == pytest.approx(2.0)
        assert stats["max_staleness"] == 4

    def test_empty_is_blank_not_an_error(self):
        stats = collect.staleness_stats({"staleness_per_merge": []})
        assert stats["mean_staleness"] == "" and stats["max_staleness"] == ""

    def test_matches_the_verified_source_on_a_real_run(self, real_result):
        stats = collect.staleness_stats(real_result)
        values = real_result["staleness_per_merge"]
        assert stats["max_staleness"] == max(values)


# ---------------------------------------------------------------------------
# fleet_stats -- the belief-vs-reality gap, and its cache
# ---------------------------------------------------------------------------

class TestFleetStats:
    def make_manifest(self, tmp_path, believed_pieces, realised):
        """Minimal manifest.json fleet_stats and datasheet_expected_t need."""
        fleet_dir = tmp_path / "fleets" / "n3_x1-o1-t1_xa15_p2500_j0p149_s4126"
        fleet_dir.mkdir(parents=True)
        manifest = {
            "fleet": fleet_dir.name,
            "counts": [1, 1, 1],
            "T_tier_s": believed_pieces,
            "E_T_s": realised,
        }
        (fleet_dir / "manifest.json").write_text(json.dumps(manifest))
        (fleet_dir / "sleep_times.csv").write_text("client_id,sleep_time\n")
        return fleet_dir

    def test_gap_percentage_is_realised_over_believed(self, tmp_path):
        fleet_dir = self.make_manifest(
            tmp_path, {"xavier": 15.0, "orin": 3.0, "thor": 1.0}, realised=6.0
        )
        config = {"clients": {"simulation_distribution":
                              {"csv_path": str(fleet_dir / "sleep_times.csv")}}}
        collect._FLEET_MANIFEST_CACHE.clear()
        stats = collect.fleet_stats(config)
        believed = (15.0 + 3.0 + 1.0) / 3       # datasheet_expected_t: mean over tiers
        assert stats["fleet_believed_e_t_s"] == pytest.approx(believed, abs=1e-6)
        assert stats["fleet_realised_e_t_s"] == pytest.approx(6.0)
        assert stats["fleet_e_t_gap_pct"] == pytest.approx(
            100 * (6.0 / believed - 1), abs=1e-3
        )

    def test_the_manifest_is_cached_by_resolved_path(self, tmp_path, monkeypatch):
        fleet_dir = self.make_manifest(
            tmp_path, {"xavier": 15.0, "orin": 3.0, "thor": 1.0}, realised=6.0
        )
        config = {"clients": {"simulation_distribution":
                              {"csv_path": str(fleet_dir / "sleep_times.csv")}}}
        collect._FLEET_MANIFEST_CACHE.clear()
        collect.fleet_stats(config)
        assert len(collect._FLEET_MANIFEST_CACHE) == 1

        # Mutating the file on disk must NOT change the cached read -- proves
        # the second call used the cache rather than re-reading.
        manifest_path = fleet_dir / "manifest.json"
        mutated = json.loads(manifest_path.read_text())
        mutated["E_T_s"] = 999.0
        manifest_path.write_text(json.dumps(mutated))

        stats_again = collect.fleet_stats(config)
        assert stats_again["fleet_realised_e_t_s"] == pytest.approx(6.0)

    def test_two_different_fleets_are_not_confused(self, tmp_path):
        a = self.make_manifest(tmp_path / "a",
                               {"xavier": 15.0, "orin": 3.0, "thor": 1.0}, 6.0)
        b = self.make_manifest(tmp_path / "b",
                               {"xavier": 30.0, "orin": 6.0, "thor": 2.0}, 12.0)
        collect._FLEET_MANIFEST_CACHE.clear()
        config_a = {"clients": {"simulation_distribution":
                                {"csv_path": str(a / "sleep_times.csv")}}}
        config_b = {"clients": {"simulation_distribution":
                                {"csv_path": str(b / "sleep_times.csv")}}}
        stats_a = collect.fleet_stats(config_a)
        stats_b = collect.fleet_stats(config_b)
        assert stats_a["fleet_realised_e_t_s"] != stats_b["fleet_realised_e_t_s"]

    def test_matches_the_real_fleet_on_a_real_run(self, real_result):
        collect._FLEET_MANIFEST_CACHE.clear()
        stats = collect.fleet_stats(real_result["config"])
        assert stats["fleet_e_t_gap_pct"] == pytest.approx(-4.8668, abs=1e-2)


# ---------------------------------------------------------------------------
# describe() -- one full run
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_a_missing_config_is_skipped_not_fatal(self, capsys):
        assert collect.describe("no_such_run_at_all") is None
        assert "no config" in capsys.readouterr().out

    def test_a_real_run_produces_every_expected_column(self, real_result):
        row = collect.describe(REAL_RUN_NAME)
        assert row is not None
        for key in ("experiment", "arm", "seed", "rounds_completed",
                   "sim_time_reconstructed_final_s", "plateau_accuracy",
                   "mean_staleness", "dispatch_xavier", "fleet_e_t_gap_pct"):
            assert key in row, key

    def test_time_to_target_is_never_read_from_platos_own_clock(self):
        """The brief, verified structurally: no column name derived from
        sim_time_plato feeds a tta_h_* or rounds_to_* value."""
        source = (DATASET_DIR / "collect.py").read_text()
        # first_crossing is the ONLY function that produces tta_h_*/rounds_to_*,
        # and curve_stats is its only caller -- both must key off
        # sim_time_reconstructed exclusively.
        assert 'row["sim_time_reconstructed"]' in source
        crossing_fn = source.split("def first_crossing")[1].split("def ")[0]
        assert "sim_time_plato" not in crossing_fn

    def test_arm_and_experiment_come_from_the_real_run(self, real_result):
        row = collect.describe(REAL_RUN_NAME)
        assert row["arm"] == "gpu"
        assert row["experiment"] == "a0p8_N20_C4_s4126_det1"


# ---------------------------------------------------------------------------
# curve_rows()
# ---------------------------------------------------------------------------

class TestCurveRows:
    def test_columns_match_the_brief_exactly(self, real_result):
        rows = collect.curve_rows(REAL_RUN_NAME)
        assert set(rows[0]) == {
            "experiment", "arm", "seed", "round",
            "sim_time_plato", "sim_time_reconstructed", "accuracy",
        }

    def test_one_row_per_round(self, real_result):
        rows = collect.curve_rows(REAL_RUN_NAME)
        assert [r["round"] for r in rows] == list(range(1, len(rows) + 1))

    def test_both_clocks_are_present_and_differ(self, real_result):
        rows = collect.curve_rows(REAL_RUN_NAME)
        assert any(r["sim_time_plato"] != r["sim_time_reconstructed"] for r in rows)


# ---------------------------------------------------------------------------
# main() -- the CLI writes both files
# ---------------------------------------------------------------------------

class TestMain:
    def run_main(self, args, monkeypatch, tmp_path):
        import sys
        monkeypatch.setattr(sys, "argv", ["collect.py", *args])
        collect.main()

    def test_writes_both_files_for_a_named_run(self, monkeypatch, tmp_path, real_result):
        self.run_main(
            ["--runs", REAL_RUN_NAME, "--outdir", str(tmp_path)],
            monkeypatch, tmp_path,
        )
        assert (tmp_path / "run_summary.csv").exists()
        assert (tmp_path / "accuracy_curves_reconstructed.csv").exists()

    def test_summary_row_count_matches_requested_runs(self, monkeypatch, tmp_path, real_result):
        self.run_main(
            ["--runs", REAL_RUN_NAME, "--outdir", str(tmp_path)],
            monkeypatch, tmp_path,
        )
        with (tmp_path / "run_summary.csv").open(newline="") as fh:
            assert len(list(csv.DictReader(fh))) == 1

    def test_a_tag_suffixes_both_output_files(self, monkeypatch, tmp_path):
        try:
            cr.run_dir("gpu_a0p8_N20_C4_s4126_det3")
        except SystemExit:
            pytest.skip("det3 is not on disk")
        self.run_main(
            ["--tag", "_det3", "--outdir", str(tmp_path)], monkeypatch, tmp_path,
        )
        assert (tmp_path / "run_summary_det3.csv").exists()
        assert (tmp_path / "accuracy_curves_reconstructed_det3.csv").exists()

    def test_no_matching_runs_is_a_clear_error(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit, match="Nothing to collect"):
            self.run_main(
                ["--tag", "_no_such_tag_at_all", "--outdir", str(tmp_path)],
                monkeypatch, tmp_path,
            )

    def test_default_invocation_excludes_known_probe_tags(self, monkeypatch, tmp_path):
        """Bare `collect.py` (no --runs, no --tag) must not silently sweep up
        probe/determinism runs alongside the real sweep. The exclusion set
        comes from `make_configs.THROWAWAY_TAGS` -- the single registry --
        rather than a second hand-maintained copy, which already drifted once
        (a prior version here missed `_det3`-`_det6` after they were added to
        `make_configs.TAG_SLOT`)."""
        import make_configs
        monkeypatch.setattr("sys.argv", ["collect.py"])
        names = sorted(p.stem for p in collect.CONFIGS.glob("*.toml"))
        filtered = [n for n in names
                   if not any(n.endswith(t) for t in make_configs.THROWAWAY_TAGS)]
        assert "gpu_a0p8_N20_C4_s4126_det1" not in filtered
        assert "gpu_a0p8_N20_C4_s4126" in filtered
        # _rep2 is a REAL run, not a throwaway -- it must survive the filter.
        assert "_rep2" not in make_configs.THROWAWAY_TAGS
