"""Tests for build_fleet.py -- the sleep-time generator and the fleet directory.

Three groups matter most:

  * `TestNoJitterReproducesOldGenerator` -- with sigma = 0 this must reproduce
    `gpu_tiers/generate_sleep_times.py` byte for byte, or the oracle-branch
    fleets cannot be rebuilt here and the move loses its audit trail.
  * `TestJitterSemantics` -- eps is per client, persistent, seeded, and
    multiplicative. Each of those is a separate claim the experiment rests on.
  * `TestRefusesToClobber` -- two different fleets must never occupy one
    directory, because configs address fleets by directory name.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

import build_fleet
import make_fleet

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent
FEDASYNC = DATASET_DIR.parent

TIERS_CSV = DATASET_DIR / "tiers" / "tiers_estimated_15x.csv"
TIERS_CPU_CSV = DATASET_DIR / "tiers" / "tiers_cpu.csv"

# The separability cell's parameters, which the oracle-branch runs used.
SEPARABILITY = dict(
    total=20, mix=(50, 35, 15), anchor_tier="xavier", anchor_sleep=15.0,
    partition_size=2500, epochs=1,
)


def build_into(tmp_path, **overrides):
    kwargs = dict(
        tiers_csv=TIERS_CSV, jitter_sigma=0.0, seed=4126,
        fleets_dir=tmp_path, **SEPARABILITY,
    )
    kwargs.update(overrides)
    return build_fleet.build(**kwargs)


def read_sleeps(path: Path) -> dict[int, float]:
    with path.open(newline="") as fh:
        return {int(r["client_id"]): float(r["sleep_time"]) for r in csv.DictReader(fh)}


# ---------------------------------------------------------------------------
# The formula, against numbers recorded elsewhere in the project
# ---------------------------------------------------------------------------

class TestFormula:
    def test_tier_times_match_the_recorded_anchor_row(self, tmp_path):
        """T = 15 / 1.9231 / 1.0 s -- separability/README.md "derived" row."""
        _, manifest = build_into(tmp_path)
        t = manifest["T_tier_s"]
        assert t["xavier"] == pytest.approx(15.0)
        assert t["orin"] == pytest.approx(1.923076923, abs=1e-9)
        assert t["thor"] == pytest.approx(1.0)

    def test_expected_t_matches_the_recorded_value(self, tmp_path):
        """E[T] = 8.3231 s -- separability/README.md and its preflight output."""
        _, manifest = build_into(tmp_path)
        assert manifest["E_T_s"] == pytest.approx(8.3231, abs=1e-4)

    def test_nominal_gap_is_15x(self, tmp_path):
        _, manifest = build_into(tmp_path)
        t = manifest["T_tier_s"]
        assert t["xavier"] / t["thor"] == pytest.approx(15.0)

    def test_anchor_is_exact(self, tmp_path):
        directory, _ = build_into(tmp_path)
        sleeps = read_sleeps(directory / "sleep_times.csv")
        assert sleeps[1] == 15.0        # client 1 is xavier, the anchor tier

    def test_anchor_scale_is_uniform_so_ratios_survive(self, tmp_path):
        """Doubling anchor_sleep doubles every sleep and changes no ratio."""
        _, a = build_into(tmp_path, anchor_sleep=15.0)
        _, b = build_into(tmp_path, anchor_sleep=30.0)
        for tier in a["T_tier_s"]:
            assert b["T_tier_s"][tier] == pytest.approx(2 * a["T_tier_s"][tier])

    def test_partition_size_scales_time_linearly(self, tmp_path):
        """Twice the data is twice the work -- but the anchor pins xavier, so the
        RATIOS stay put and only the un-anchored view would move. Check the raw
        formula directly instead."""
        tiers = build_fleet.load_tiers(TIERS_CSV)
        one = build_fleet.raw_seconds(tiers["orin"], 2500, 1)
        two = build_fleet.raw_seconds(tiers["orin"], 5000, 1)
        assert two == pytest.approx(2 * one)

    def test_epochs_scale_time_linearly(self):
        tiers = build_fleet.load_tiers(TIERS_CSV)
        one = build_fleet.raw_seconds(tiers["orin"], 2500, 1)
        two = build_fleet.raw_seconds(tiers["orin"], 2500, 2)
        assert two == pytest.approx(2 * one)

    def test_cpu_tier_file_gives_the_cpu_arm_belief(self, tmp_path):
        """The cpu arm believes xavier == orin and thor 1.3x faster.

        separability/README.md records T_round = 3.620192 s for the cpu arm at
        C = 4, i.e. E[T] = 14.4808 s. That is this file's fleet under the cpu
        tier spec, and it is what makes the cpu arm's wrongness the specific
        wrongness the experiment intends.
        """
        _, manifest = build_into(tmp_path, tiers_csv=TIERS_CPU_CSV)
        t = manifest["T_tier_s"]
        assert t["xavier"] == pytest.approx(15.0)
        assert t["orin"] == pytest.approx(15.0)          # cannot tell them apart
        assert t["thor"] == pytest.approx(15.0 * 0.002 / 0.0026)
        assert manifest["E_T_s"] == pytest.approx(14.4808, abs=1e-3)
        assert manifest["E_T_s"] / 4 == pytest.approx(3.620192, abs=1e-5)

    def test_unknown_anchor_tier_is_fatal(self, tmp_path):
        with pytest.raises(SystemExit, match="anchor tier"):
            build_into(tmp_path, anchor_tier="orion")


# ---------------------------------------------------------------------------
# sigma = 0 must reproduce the generator this one replaces
# ---------------------------------------------------------------------------

class TestNoJitterReproducesOldGenerator:
    def test_matches_separability_sleep_csv_byte_for_byte(self, tmp_path):
        reference = (FEDASYNC / "separability" / "output"
                     / "client_sleep_times_xa15_n20.csv")
        if not reference.exists():
            pytest.skip(f"{reference} not present")
        directory, _ = build_into(tmp_path, jitter_sigma=0.0)
        assert (directory / "sleep_times.csv").read_bytes() == reference.read_bytes()

    def test_matches_separability_fleet_csv_byte_for_byte(self, tmp_path):
        reference = FEDASYNC / "separability" / "input" / "clients_n20.csv"
        if not reference.exists():
            pytest.skip(f"{reference} not present")
        directory, _ = build_into(tmp_path, jitter_sigma=0.0)
        assert (directory / "clients.csv").read_bytes() == reference.read_bytes()

    def test_matches_the_old_generator_at_n100(self, tmp_path):
        reference = (FEDASYNC / "separability_n100" / "output"
                     / "client_sleep_times_xa15_n100.csv")
        if not reference.exists():
            pytest.skip(f"{reference} not present")
        directory, _ = build_into(tmp_path, total=100, jitter_sigma=0.0)
        assert (directory / "sleep_times.csv").read_bytes() == reference.read_bytes()

    def test_zero_sigma_does_not_consume_the_rng(self):
        """All-zero eps without drawing, so a no-jitter build cannot depend on
        the seed at all -- which is what makes it byte-identical for any seed."""
        a = build_fleet.draw_epsilons([1, 2, 3], 0.0, 4126)
        b = build_fleet.draw_epsilons([1, 2, 3], 0.0, 9999)
        assert a == b == {1: 0.0, 2: 0.0, 3: 0.0}

    def test_w_model_flops_matches_runtime(self):
        """The simulator and the predictor must divide the same FLOPs budget.

        A mismatch is invisible at run time: it appears as a constant factor on
        every predicted time, which looks exactly like a real estimator bias.
        """
        sys.path.insert(0, str(FEDASYNC / "gpu_tiers"))
        try:
            import gpu_tier_sleep
        except Exception as exc:  # pragma: no cover - env without PLATO deps
            pytest.skip(f"gpu_tier_sleep not importable: {exc}")
        assert build_fleet.W_MODEL_FLOPS == gpu_tier_sleep.W_MODEL_FLOPS


# ---------------------------------------------------------------------------
# What the jitter actually means
# ---------------------------------------------------------------------------

class TestJitterSemantics:
    def test_eps_is_drawn_once_per_client(self):
        eps = build_fleet.draw_epsilons(list(range(1, 21)), 0.18, 4126)
        assert len(eps) == 20
        assert len(set(eps.values())) == 20        # no client shares another's draw

    def test_eps_is_reproducible_for_a_seed(self):
        a = build_fleet.draw_epsilons(list(range(1, 21)), 0.18, 4126)
        b = build_fleet.draw_epsilons(list(range(1, 21)), 0.18, 4126)
        assert a == b

    def test_eps_changes_with_the_seed(self):
        a = build_fleet.draw_epsilons(list(range(1, 21)), 0.18, 4126)
        b = build_fleet.draw_epsilons(list(range(1, 21)), 0.18, 4127)
        assert a != b

    def test_eps_does_not_depend_on_input_order(self):
        forward = build_fleet.draw_epsilons([1, 2, 3, 4], 0.18, 4126)
        backward = build_fleet.draw_epsilons([4, 3, 2, 1], 0.18, 4126)
        assert forward == backward

    def test_eps_does_not_depend_on_the_global_rng(self):
        """A dedicated Random instance: nothing else in the process can move it."""
        import random as _random
        _random.seed(1)
        a = build_fleet.draw_epsilons([1, 2, 3], 0.18, 4126)
        _random.seed(999)
        [_random.random() for _ in range(17)]
        b = build_fleet.draw_epsilons([1, 2, 3], 0.18, 4126)
        assert a == b

    def test_jitter_is_multiplicative_on_t_tier(self, tmp_path):
        directory, manifest = build_into(tmp_path, jitter_sigma=0.18)
        table = list(csv.DictReader((directory / "sleep_table.csv").open(newline="")))
        for row in table:
            expected = float(row["T_tier_s"]) * math.exp(float(row["eps"]))
            assert float(row["sleep_s"]) == pytest.approx(expected, rel=1e-9)

    def test_sleep_stays_positive_for_a_large_sigma(self, tmp_path):
        """Lognormal cannot go negative; the additive form the old noise_sigma
        column used could, and was clamped at 0."""
        directory, _ = build_into(tmp_path, jitter_sigma=1.5, total=100)
        assert all(v > 0 for v in read_sleeps(directory / "sleep_times.csv").values())

    def test_t_tier_is_the_median_not_the_mean(self):
        """exp(eps) has median 1 and mean exp(sigma^2/2): the tier mean sits
        ABOVE T_tier. Documented, deliberate, and worth pinning."""
        sigma = 0.18
        eps = build_fleet.draw_epsilons(list(range(1, 20001)), sigma, 4126)
        factors = [math.exp(e) for e in eps.values()]
        assert statistics.fmean(factors) == pytest.approx(
            math.exp(sigma**2 / 2), rel=0.02
        )
        assert statistics.median(factors) == pytest.approx(1.0, rel=0.02)

    def test_realised_cv_approaches_sigma_for_a_large_fleet(self, tmp_path):
        """CV of a lognormal is sqrt(exp(sigma^2)-1) -- ~0.1812 at sigma 0.18."""
        sigma = 0.18
        directory, manifest = build_into(
            tmp_path, total=10000, jitter_sigma=sigma, partition_size=2500
        )
        expected = math.sqrt(math.exp(sigma**2) - 1)
        for row in manifest["tier_stats"].values():
            assert row["cv_population"] == pytest.approx(expected, rel=0.05)

    def test_comm_latency_is_added_after_jitter_and_is_identical(self, tmp_path):
        """C is already in simulator units: neither scaled nor jittered."""
        plain, _ = build_into(tmp_path / "a", jitter_sigma=0.18)
        withc, _ = build_into(tmp_path / "b", jitter_sigma=0.18,
                              comm_bandwidth_mbps=100.0)
        base = read_sleeps(plain / "sleep_times.csv")
        withc_sleeps = read_sleeps(withc / "sleep_times.csv")
        deltas = {cid: withc_sleeps[cid] - base[cid] for cid in base}
        expected = build_fleet.DEFAULT_PAYLOAD_MBIT / 100.0
        assert all(d == pytest.approx(expected, abs=1e-6) for d in deltas.values())

    def test_negative_sigma_refused(self):
        with pytest.raises(SystemExit, match="jitter-sigma"):
            build_fleet.draw_epsilons([1], -0.1, 4126)

    def test_non_positive_bandwidth_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="comm-bandwidth"):
            build_into(tmp_path, comm_bandwidth_mbps=0.0)


# ---------------------------------------------------------------------------
# The exact fleet Stage 1 reports on -- pinned so a refactor cannot move it
# ---------------------------------------------------------------------------

class TestStage1Fleet:
    """seed 4126, sigma 0.18, N=20. These numbers go into stage1_evidence/.

    Pinned to 6 decimals: `random.Random.gauss` is deterministic within a CPython
    version but is not a documented cross-version guarantee, so if a future
    interpreter changes it this test fails loudly instead of the fleet silently
    becoming a different one.
    """

    SIGMA = 0.18
    SEED = 4126

    def fleet(self, tmp_path):
        return build_into(tmp_path, jitter_sigma=self.SIGMA, seed=self.SEED)

    def test_epsilon_values_are_pinned(self):
        eps = build_fleet.draw_epsilons(list(range(1, 21)), self.SIGMA, self.SEED)
        # Client 1 and the last client bracket the whole draw sequence.
        assert eps[1] == pytest.approx(-0.063097, abs=1e-6)
        assert eps[20] == pytest.approx(-0.023507, abs=1e-6)
        assert statistics.fmean(eps.values()) == pytest.approx(-0.056182, abs=1e-6)

    def test_sigma_was_chosen_from_the_nominal_cv(self):
        """sigma = 0.18 is picked because the DISTRIBUTION's CV is mid-band.

        CV of a lognormal is sqrt(exp(sigma^2) - 1) = 18.1 % at sigma = 0.18,
        the centre of the brief's 15-20 % target. That is the criterion, and it
        is a property of sigma alone -- it does not move with the draw, so it
        cannot be tuned after seeing a result.
        """
        nominal = math.sqrt(math.exp(self.SIGMA**2) - 1)
        assert 0.15 <= nominal <= 0.20
        assert nominal == pytest.approx(0.1815, abs=5e-4)

    def test_realised_cv_is_pinned_including_the_two_excursions(self, tmp_path):
        """What this fleet actually measures, per tier, under both conventions.

        The fleet has 10 / 7 / 3 clients per tier, so the realised CV carries a
        large sampling error and NO sigma puts all six numbers inside 15-20 %:
        orin (n=7) is always the lowest and thor (n=3) always the highest, and
        the two constraints leave a window about 0.001 wide in sigma. Chasing it
        would be tuning sigma to make a result look better, so sigma stays at the
        value the nominal CV chose and the excursions are recorded here instead:

            orin  population CV  14.90 %  -- 0.10 pt BELOW the band
            thor  sample     CV  21.13 %  -- 1.13 pt ABOVE the band

        Everything else lands inside. Both excursions are small-n artefacts of
        the estimator, not evidence that the jitter is the wrong size.
        """
        _, manifest = self.fleet(tmp_path)
        stats = manifest["tier_stats"]

        assert stats["xavier"]["cv_population"] == pytest.approx(0.165500, abs=1e-6)
        assert stats["orin"]["cv_population"] == pytest.approx(0.149028, abs=1e-6)
        assert stats["thor"]["cv_population"] == pytest.approx(0.172547, abs=1e-6)

        assert stats["xavier"]["cv_sample"] == pytest.approx(0.174452, abs=1e-6)
        assert stats["orin"]["cv_sample"] == pytest.approx(0.160969, abs=1e-6)
        assert stats["thor"]["cv_sample"] == pytest.approx(0.211326, abs=1e-6)

        # Four of the six sit inside the band; the two named above do not.
        inside = [
            0.15 <= stats[tier][key] <= 0.20
            for tier in ("xavier", "orin", "thor")
            for key in ("cv_population", "cv_sample")
        ]
        assert sum(inside) == 4
        assert not (0.15 <= stats["orin"]["cv_population"] <= 0.20)
        assert not (0.15 <= stats["thor"]["cv_sample"] <= 0.20)

    def test_every_tier_is_within_a_point_of_the_band(self, tmp_path):
        """The excursions are small. A tier drifting far outside would mean the
        jitter is doing something other than what it is documented to do."""
        _, manifest = self.fleet(tmp_path)
        for tier, row in manifest["tier_stats"].items():
            for key in ("cv_population", "cv_sample"):
                assert 0.14 <= row[key] <= 0.22, (tier, key, row[key])

    def test_expected_t_after_jitter_is_pinned(self, tmp_path):
        """E[T] moves 8.3231 -> 7.9180 s: the fleet's draw came out slightly
        fast. T_round = E[T]/C follows it, so this number must be visible."""
        _, manifest = self.fleet(tmp_path)
        assert manifest["E_T_s"] == pytest.approx(7.918013, abs=1e-6)

    def test_tiers_do_not_overlap(self, tmp_path):
        """The sanity requirement: jitter must not blur the tiers."""
        _, manifest = self.fleet(tmp_path)
        for row in manifest["tier_overlaps"]:
            assert not row["overlap"], row

    def test_separation_margins_are_pinned(self, tmp_path):
        """How much room is left, not just whether there is any.

        xavier/orin is comfortable (7.6x). orin/thor is the thin one: the two
        tiers are only 1.92x apart before jitter, so at sigma = 0.18 the fastest
        orin and the slowest thor come within 5 % of each other. It clears, but
        it is the constraint that keeps sigma below ~0.19 -- not the CV target.
        """
        _, manifest = self.fleet(tmp_path)
        margins = {(r["slower"], r["faster"]): r["margin"]
                   for r in manifest["tier_overlaps"]}
        assert margins[("xavier", "orin")] == pytest.approx(4.935, abs=1e-3)
        assert margins[("orin", "thor")] == pytest.approx(1.051, abs=1e-3)
        assert all(m > 1.0 for m in margins.values())

    def test_tier_ordering_is_preserved_per_client(self, tmp_path):
        """Every xavier client is slower than every orin client, and so on."""
        directory, _ = self.fleet(tmp_path)
        rows = list(csv.DictReader((directory / "sleep_table.csv").open(newline="")))
        by_tier: dict[str, list[float]] = {}
        for row in rows:
            by_tier.setdefault(row["tier"], []).append(float(row["sleep_s"]))
        for slower, faster in zip(make_fleet.TIERS, make_fleet.TIERS[1:]):
            assert min(by_tier[slower]) > max(by_tier[faster])

    def test_tier_mean_ratio_still_reads_as_a_large_gap(self, tmp_path):
        """~15x before jitter; it must not collapse toward 1."""
        _, manifest = self.fleet(tmp_path)
        stats = manifest["tier_stats"]
        ratio = stats["xavier"]["mean_s"] / stats["thor"]["mean_s"]
        assert 12.0 < ratio < 15.0


# ---------------------------------------------------------------------------
# The fleet directory as an addressable, immutable artefact
# ---------------------------------------------------------------------------

class TestFleetDirectory:
    def test_writes_the_four_files(self, tmp_path):
        directory, _ = build_into(tmp_path, jitter_sigma=0.18)
        for name in ("clients.csv", "sleep_times.csv", "sleep_table.csv",
                     "manifest.json"):
            assert (directory / name).exists(), name

    def test_directory_name_carries_the_swept_axes(self, tmp_path):
        directory, _ = build_into(tmp_path, jitter_sigma=0.18, seed=4126)
        assert directory.name == "n20_x10-o7-t3_xa15_p2500_j0p180_s4126"

    def test_name_has_no_dot(self, tmp_path):
        directory, _ = build_into(tmp_path, jitter_sigma=0.18)
        assert "." not in directory.name

    def test_different_sigma_is_a_different_directory(self, tmp_path):
        a, _ = build_into(tmp_path, jitter_sigma=0.18)
        b, _ = build_into(tmp_path, jitter_sigma=0.15)
        assert a != b

    def test_different_seed_is_a_different_directory(self, tmp_path):
        a, _ = build_into(tmp_path, jitter_sigma=0.18, seed=4126)
        b, _ = build_into(tmp_path, jitter_sigma=0.18, seed=4127)
        assert a != b

    def test_sleep_table_rows_match_sleep_times(self, tmp_path):
        directory, _ = build_into(tmp_path, jitter_sigma=0.18)
        sleeps = read_sleeps(directory / "sleep_times.csv")
        table = list(csv.DictReader((directory / "sleep_table.csv").open(newline="")))
        assert len(table) == len(sleeps)
        for row in table:
            assert float(row["sleep_s"]) == pytest.approx(
                sleeps[int(row["client_id"])], abs=1e-9
            )

    def test_sleep_table_tiers_match_the_fleet(self, tmp_path):
        directory, _ = build_into(tmp_path, jitter_sigma=0.18)
        fleet = {int(r["client_id"]): r["tier_id"] for r in
                 csv.DictReader((directory / "clients.csv").open(newline=""))}
        table = {int(r["client_id"]): r["tier"] for r in
                 csv.DictReader((directory / "sleep_table.csv").open(newline=""))}
        assert fleet == table

    def test_sleep_times_csv_has_exactly_the_two_plato_columns(self, tmp_path):
        directory, _ = build_into(tmp_path, jitter_sigma=0.18)
        with (directory / "sleep_times.csv").open(newline="") as fh:
            assert next(csv.reader(fh)) == ["client_id", "sleep_time"]

    def test_manifest_records_the_tier_file_hash(self, tmp_path):
        _, manifest = build_into(tmp_path, jitter_sigma=0.18)
        assert manifest["tiers_sha256"] == build_fleet.sha256_of(TIERS_CSV)

    def test_manifest_records_the_realised_mix(self, tmp_path):
        """N=30 is 15/10/5, not 15/10.5/4.5 -- the analysis must be able to see it."""
        _, manifest = build_into(tmp_path, total=30, partition_size=1667,
                                 jitter_sigma=0.18)
        assert manifest["counts"] == [15, 10, 5]
        assert manifest["mix"] == [50, 35, 15]      # what was asked for, kept too

    def test_manifest_is_valid_json_on_disk(self, tmp_path):
        directory, manifest = build_into(tmp_path, jitter_sigma=0.18)
        on_disk = json.loads((directory / "manifest.json").read_text())
        assert on_disk["inputs_digest"] == manifest["inputs_digest"]


class TestRefusesToClobber:
    def test_rebuilding_the_same_fleet_is_allowed(self, tmp_path):
        a, ma = build_into(tmp_path, jitter_sigma=0.18)
        b, mb = build_into(tmp_path, jitter_sigma=0.18)
        assert a == b
        assert ma["inputs_digest"] == mb["inputs_digest"]

    def test_different_inputs_under_one_name_are_refused(self, tmp_path):
        """Same directory name, different tier file: the case the name cannot see."""
        build_into(tmp_path, jitter_sigma=0.18)
        with pytest.raises(SystemExit, match="DIFFERENT inputs"):
            build_into(tmp_path, jitter_sigma=0.18, tiers_csv=TIERS_CPU_CSV)

    def test_different_epochs_under_one_name_are_refused(self, tmp_path):
        build_into(tmp_path, jitter_sigma=0.18)
        with pytest.raises(SystemExit, match="DIFFERENT inputs"):
            build_into(tmp_path, jitter_sigma=0.18, epochs=2)

    def test_force_overrides(self, tmp_path):
        build_into(tmp_path, jitter_sigma=0.18)
        directory, manifest = build_into(tmp_path, jitter_sigma=0.18,
                                         tiers_csv=TIERS_CPU_CSV, force=True)
        assert manifest["tiers_sha256"] == build_fleet.sha256_of(TIERS_CPU_CSV)

    def test_digest_is_order_independent(self):
        a = build_fleet.inputs_digest({"x": 1, "y": 2})
        b = build_fleet.inputs_digest({"y": 2, "x": 1})
        assert a == b

    def test_digest_moves_with_the_generator_version(self, tmp_path, monkeypatch):
        _, before = build_into(tmp_path, jitter_sigma=0.18)
        monkeypatch.setattr(build_fleet, "GENERATOR_VERSION",
                            build_fleet.GENERATOR_VERSION + 1)
        _, after = build_into(tmp_path, jitter_sigma=0.18, force=True)
        assert before["inputs_digest"] != after["inputs_digest"]


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

class TestCli:
    def run(self, *args, expect=0):
        proc = subprocess.run(
            [sys.executable, str(DATASET_DIR / "build_fleet.py"), *args],
            capture_output=True, text=True, cwd=DATASET_DIR,
        )
        assert proc.returncode == expect, proc.stderr
        return proc

    def test_dry_run_writes_nothing(self, tmp_path):
        proc = self.run("--total", "20", "--partition-size", "2500",
                        "--jitter-sigma", "0.18", "--seed", "4126",
                        "--fleets-dir", str(tmp_path), "--dry-run")
        assert "would write" in proc.stdout
        assert not list(tmp_path.iterdir())

    def test_summary_states_the_separation_verdict(self, tmp_path):
        proc = self.run("--total", "20", "--partition-size", "2500",
                        "--jitter-sigma", "0.18", "--seed", "4126",
                        "--fleets-dir", str(tmp_path), "--dry-run")
        assert "separated" in proc.stdout
        assert "CV(smp)" in proc.stdout

    def test_jitter_sigma_and_seed_are_required(self, tmp_path):
        self.run("--total", "20", "--partition-size", "2500", expect=2)
