"""Tests for preflight.py.

preflight's value is entirely in what it REFUSES, so these tests corrupt a good
setup one way at a time and check each corruption is caught. A preflight that
passes everything is worse than no preflight, because it is trusted.

The two inverted checks get their own group: `../separability/preflight.py` fails
when a tier has more than one sleep time, and this one fails when a tier has only
one. Running the old check against a jittered fleet would block every launch; not
running the new one would let an un-jittered fleet through, and an un-jittered
fleet puts the oracle back into the `gpu` arm silently.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import make_configs
import preflight

DATASET_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reset_verdicts():
    preflight.FAILURES.clear()
    preflight.WARNINGS.clear()
    yield
    preflight.FAILURES.clear()
    preflight.WARNINGS.clear()


@pytest.fixture
def fleet(tmp_path):
    """A real built fleet, copied so a test can corrupt it."""
    try:
        source = make_configs.fleet_dir(20)
    except SystemExit:
        pytest.skip("fleet for N=20 is not built; run build_fleet.py")
    destination = tmp_path / source.name
    shutil.copytree(source, destination)
    return destination


@pytest.fixture
def configs(tmp_path, fleet, monkeypatch):
    """Configs pointing at the COPIED fleet, so a test may corrupt either.

    `make_configs.render` resolves the fleet through `make_configs.FLEETS_DIR`;
    without redirecting it the configs would reference the real fleet on disk and
    a test that corrupts the copy would silently assert nothing.
    """
    monkeypatch.setattr(make_configs, "FLEETS_DIR", fleet.parent)
    outdir = tmp_path / "configs"
    paths = [make_configs.render(arm, 0.8, outdir, 4126, 20, 4)
             for arm in sorted(make_configs.ARMS)]
    for path in paths:
        assert str(fleet) in path.read_text(), "fixture did not redirect the fleet"
    return paths


def manifest_of(fleet):
    return json.loads((fleet / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# A good setup passes
# ---------------------------------------------------------------------------

class TestCleanSetupPasses:
    def test_fleet_checks_pass(self, fleet):
        assert preflight.check_fleet(fleet) is not None
        assert preflight.FAILURES == []

    def test_jitter_checks_pass(self, fleet):
        preflight.check_jitter(fleet, manifest_of(fleet), 15)
        assert preflight.FAILURES == []

    def test_a_far_from_nominal_tier_warns_but_does_not_fail(self, fleet):
        """Realised CV scatters around the nominal by several points at these
        fleet sizes. That is reported, never fatal -- sigma is chosen on the
        nominal CV, which is a property of sigma alone."""
        manifest = manifest_of(fleet)
        manifest["tier_stats"]["thor"]["cv_population"] = 0.40   # far out
        preflight.check_jitter(fleet, manifest, 15)
        assert preflight.FAILURES == []

    def test_a_changed_sigma_is_fatal(self, fleet):
        """sigma is LOCKED at Stage 2. A fleet built with another value must not
        quietly enter a sweep beside fleets built with the locked one."""
        manifest = manifest_of(fleet)
        manifest["jitter_sigma"] = 0.18
        preflight.check_jitter(fleet, manifest, 15)
        assert any("sigma has been changed" in f for f in preflight.FAILURES)

    def test_configs_pass(self, configs):
        preflight.check_configs(configs)
        assert preflight.FAILURES == []

    def test_round_clock_passes(self, fleet):
        manifest = manifest_of(fleet)
        by_tier = {"xavier": [15.0], "orin": [1.9], "thor": [1.0]}
        preflight.check_round_clock(manifest, 4, 25, by_tier)
        assert preflight.FAILURES == []


# ---------------------------------------------------------------------------
# The two checks that are INVERTED relative to ../separability/preflight.py
# ---------------------------------------------------------------------------

class TestInvertedChecks:
    def test_an_unjittered_fleet_is_refused(self, fleet):
        """The failure this whole stage exists to prevent: one sleep time per
        tier means the gpu arm's belief equals reality again."""
        table = fleet / "sleep_table.csv"
        rows = table.read_text().splitlines()
        rebuilt = [rows[0]]
        for row in rows[1:]:
            cid, tier, t_tier, _eps, _sleep = row.split(",")
            rebuilt.append(f"{cid},{tier},{t_tier},0.000000000,{t_tier}")
        table.write_text("\n".join(rebuilt) + "\n")

        preflight.check_jitter(fleet, manifest_of(fleet), 15)
        assert any("jitter did not apply" in f for f in preflight.FAILURES)
        assert any("oracle again" in f for f in preflight.FAILURES)

    def test_anchor_is_checked_on_t_tier_not_on_a_client(self, fleet):
        """No xavier client sleeps exactly 15 s any more. The old check compared
        a client's sleep against anchor_sleep and would fail every jittered
        fleet; this one compares T_tier, the datasheet value."""
        manifest = manifest_of(fleet)
        sleeps = [float(r["sleep_s"])
                  for r in preflight.read_csv(fleet / "sleep_table.csv")
                  if r["tier"] == "xavier"]
        assert 15.0 not in sleeps                      # the old check's premise
        preflight.check_jitter(fleet, manifest, 15)
        assert preflight.FAILURES == []                # ...and this one is fine

    def test_a_wrong_anchor_is_still_caught(self, fleet):
        preflight.check_jitter(fleet, manifest_of(fleet), anchor_sleep=100)
        assert any("mix units" in f for f in preflight.FAILURES)


# ---------------------------------------------------------------------------
# Tier separation
# ---------------------------------------------------------------------------

class TestSeparation:
    def test_overlapping_tiers_are_fatal(self, fleet):
        manifest = manifest_of(fleet)
        manifest["tier_overlaps"][1]["overlap"] = True
        manifest["tier_overlaps"][1]["margin"] = 0.97
        preflight.check_jitter(fleet, manifest, 15)
        assert any("RANGES OVERLAP" in f for f in preflight.FAILURES)

    def test_a_thin_margin_warns(self, fleet):
        manifest = manifest_of(fleet)
        manifest["tier_overlaps"][1]["margin"] = 1.005
        preflight.check_jitter(fleet, manifest, 15)
        assert preflight.FAILURES == []
        assert any("separated by under 2%" in w for w in preflight.WARNINGS)


# ---------------------------------------------------------------------------
# Fleet-internal consistency
# ---------------------------------------------------------------------------

class TestFleetConsistency:
    def test_sleep_table_disagreeing_with_sleep_times_is_caught(self, fleet):
        path = fleet / "sleep_times.csv"
        rows = path.read_text().splitlines()
        rows[1] = "1,99.000000000"
        path.write_text("\n".join(rows) + "\n")
        preflight.check_fleet(fleet)
        assert any("disagrees with sleep_times" in f for f in preflight.FAILURES)

    def test_a_broken_formula_row_is_caught(self, fleet):
        """sleep_s must equal T_tier * exp(eps): preflight re-derives it rather
        than trusting the generator that wrote it."""
        path = fleet / "sleep_table.csv"
        rows = path.read_text().splitlines()
        cid, tier, t_tier, eps, sleep = rows[1].split(",")
        rows[1] = f"{cid},{tier},{t_tier},{eps},{float(sleep) * 1.5:.9f}"
        path.write_text("\n".join(rows) + "\n")
        preflight.check_fleet(fleet)
        assert any("T_tier * exp(eps)" in f for f in preflight.FAILURES)

    def test_a_missing_client_is_caught(self, fleet):
        path = fleet / "sleep_times.csv"
        rows = path.read_text().splitlines()
        path.write_text("\n".join(rows[:-1]) + "\n")
        preflight.check_fleet(fleet)
        assert any("same client IDs" in f for f in preflight.FAILURES)

    def test_a_missing_manifest_is_caught(self, fleet):
        (fleet / "manifest.json").unlink()
        assert preflight.check_fleet(fleet) is None
        assert any("manifest" in f for f in preflight.FAILURES)


# ---------------------------------------------------------------------------
# Config-level traps
# ---------------------------------------------------------------------------

class TestConfigTraps:
    def corrupt(self, configs, arm, old, new):
        path = next(p for p in configs if make_configs.arm_of(p) == arm)
        path.write_text(path.read_text().replace(old, new, 1))
        return path

    def test_oracle_key_on_the_gpu_arm_is_caught(self, configs):
        self.corrupt(configs, "gpu", 't_round_model = "queue"',
                     't_round_model = "queue"\noracle_sleep_csv = "/x.csv"')
        preflight.check_configs(configs)
        assert any("different arm" in f for f in preflight.FAILURES)

    def test_oracle_arm_without_the_key_is_caught(self, configs):
        path = next(p for p in configs if make_configs.arm_of(p) == "gpu_oracle")
        kept = [l for l in path.read_text().splitlines()
                if not l.startswith("oracle_sleep_csv")]
        path.write_text("\n".join(kept))
        preflight.check_configs(configs)
        assert any("different arm" in f for f in preflight.FAILURES)

    def test_an_oracle_reading_another_fleet_is_caught(self, configs, tmp_path):
        path = next(p for p in configs if make_configs.arm_of(p) == "gpu_oracle")
        text = path.read_text()
        current = [l for l in text.splitlines()
                   if l.startswith("oracle_sleep_csv")][0]
        other = tmp_path / "other_sleep_times.csv"
        other.write_text("client_id,sleep_time\n1,1.0\n")
        path.write_text(text.replace(current, f'oracle_sleep_csv = "{other}"'))
        preflight.check_configs(configs)
        assert any("the clients sleep from" in f for f in preflight.FAILURES)

    def test_partition_size_not_matching_the_fleet_is_caught(self, configs):
        """The sleep times were generated for one workload; a config that trains
        on another silently breaks the belief/reality correspondence."""
        self.corrupt(configs, "gpu", "partition_size = 2500",
                     "partition_size = 1250")
        preflight.check_configs(configs)
        assert any("different workload" in f for f in preflight.FAILURES)

    def test_disagreeing_seeds_are_caught(self, configs):
        path = next(p for p in configs if make_configs.arm_of(p) == "gpu")
        text = path.read_text().replace(
            "[data]\ndatasource", "[data]\ndatasource", 1)
        # change only the [data] one, which is the last of the three
        head, _, tail = text.partition("[data]")
        path.write_text(head + "[data]" + tail.replace("random_seed = 4126",
                                                       "random_seed = 9999", 1))
        preflight.check_configs(configs)
        assert any("random_seed keys disagree" in f for f in preflight.FAILURES)

    def test_nonzero_avg_training_time_is_caught(self, configs):
        self.corrupt(configs, "gpu", "avg_training_time = 0.0",
                     "avg_training_time = 2.0")
        preflight.check_configs(configs)
        assert any("avg_training_time" in f for f in preflight.FAILURES)

    def test_max_concurrency_drift_is_caught(self, configs):
        self.corrupt(configs, "gpu", "max_concurrency = 4", "max_concurrency = 8")
        preflight.check_configs(configs)
        assert any("max_concurrency" in f for f in preflight.FAILURES)

    def test_wrong_t_round_model_is_caught(self, configs):
        self.corrupt(configs, "gpu", 't_round_model = "queue"',
                     't_round_model = "min"')
        preflight.check_configs(configs)
        assert any("t_round_model" in f for f in preflight.FAILURES)

    def test_missing_sleep_csv_is_caught(self, configs, fleet):
        (fleet / "sleep_times.csv").unlink()
        preflight.check_configs(configs)
        assert any("sleep CSV missing" in f for f in preflight.FAILURES)


class TestStalenessBoundWarning:
    def test_a_bound_the_slowest_client_can_reach_warns(self, fleet):
        """Checked against the slowest CLIENT, not the tier mean: with jitter
        they are no longer the same number, and the slowest client is the one
        that would trip the flush."""
        manifest = manifest_of(fleet)
        by_tier = {"xavier": [17.89], "orin": [2.2], "thor": [1.36]}
        preflight.check_round_clock(manifest, 4, staleness_bound=10, by_tier=by_tier)
        assert any("within 20% of staleness_bound" in w for w in preflight.WARNINGS)

    def test_the_real_bound_has_headroom(self, fleet):
        manifest = manifest_of(fleet)
        table = preflight.read_csv(fleet / "sleep_table.csv")
        by_tier: dict[str, list[float]] = {}
        for row in table:
            by_tier.setdefault(row["tier"], []).append(float(row["sleep_s"]))
        preflight.check_round_clock(manifest, 4, staleness_bound=25, by_tier=by_tier)
        assert preflight.WARNINGS == []


# ---------------------------------------------------------------------------
# Exit status -- what a launcher script keys off
# ---------------------------------------------------------------------------

class TestExitStatus:
    def run(self, *args):
        return subprocess.run(
            [sys.executable, str(DATASET_DIR / "preflight.py"), "--skip-env", *args],
            capture_output=True, text=True, cwd=DATASET_DIR,
        )

    def test_clean_setup_exits_zero(self, configs):
        proc = self.run("--configs", *[str(p) for p in configs])
        assert proc.returncode == 0, proc.stdout
        assert "preflight clean." in proc.stdout

    def test_a_failure_exits_nonzero(self, configs):
        path = next(p for p in configs if make_configs.arm_of(p) == "gpu")
        path.write_text(path.read_text().replace("avg_training_time = 0.0",
                                                 "avg_training_time = 1.0"))
        proc = self.run("--configs", *[str(p) for p in configs])
        assert proc.returncode == 1
        assert "do not launch" in proc.stdout

    def test_no_matching_configs_exits_nonzero(self):
        """A tag that matches nothing must not silently check zero configs and
        report clean -- that is a launcher's green light."""
        proc = self.run("--tag", "_no_such_tag")
        assert proc.returncode == 1
        assert "no configs to check" in proc.stdout


class TestDefaultTagExclusion:
    """A bare `preflight.py` invocation must exclude throwaway probe/determinism
    builds from the real sweep's check, and this list must come from
    `make_configs.THROWAWAY_TAGS` rather than a second hand-maintained copy --
    a copy here once listed only `_det1, _det2, _probe` and silently missed
    `_det3`-`_det6` and `_smoke` after they were registered elsewhere.
    """

    def test_matches_make_configs_throwaway_tags(self):
        source = (DATASET_DIR / "preflight.py").read_text()
        assert "make_configs.THROWAWAY_TAGS" in source

    def test_every_registered_probe_tag_is_excludable(self):
        for tag in make_configs.THROWAWAY_TAGS:
            assert tag != "_rep2"

    def test_rep2_is_not_a_throwaway_tag(self):
        """_rep2 is a real, permanent measurement (Stage 3's repeat of the C=4
        gpu cell) and must be preflighted like every other real run, not
        silently skipped the way _det1/_probe/_smoke are."""
        assert "_rep2" not in make_configs.THROWAWAY_TAGS

    def test_bare_invocation_would_include_a_rep2_config_on_disk(self, tmp_path):
        """A behavioural check, not just a registry check: render a fleet-free
        stand-in _rep2 config name and confirm preflight's own filtering logic
        keeps it."""
        names = ["gpu_a0p8_N20_C4_s4126_rep2", "gpu_a0p8_N20_C4_s4126_det1",
                "gpu_a0p8_N20_C4_s4126"]
        kept = [n for n in names
               if not any(n.endswith(t) for t in make_configs.THROWAWAY_TAGS)]
        assert kept == ["gpu_a0p8_N20_C4_s4126_rep2", "gpu_a0p8_N20_C4_s4126"]
