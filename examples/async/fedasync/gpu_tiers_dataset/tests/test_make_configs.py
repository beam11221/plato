"""Tests for make_configs.py.

The generator's job is to make the arms of a cell differ in exactly the keys that
define an arm and in nothing else, and to make two runs never land on one port.
Both failures are silent at run time: the first produces a confounded comparison
that still looks like a clean one, the second hangs instead of crashing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import build_fleet
import make_configs
import make_fleet

DATASET_DIR = Path(__file__).resolve().parents[1]

ALL_C = sorted(make_configs.C_SLOT)
ALL_N = sorted(make_configs.N_SLOT)
ALL_ARMS = sorted(make_configs.ARMS)


@pytest.fixture
def fleets_built():
    missing = []
    for n in ALL_N:
        try:
            make_configs.fleet_dir(n)
        except SystemExit:
            missing.append(n)
    if missing:
        pytest.skip(f"fleets not built for N={missing}; run build_fleet.py")


@pytest.fixture
def cell(tmp_path, fleets_built):
    """Every arm of one cell, rendered."""
    paths = [
        make_configs.render(arm, 0.8, tmp_path, 4126, 20, 4)
        for arm in ALL_ARMS
    ]
    make_configs.validate(paths)
    return {make_configs.arm_of(p): tomllib.loads(p.read_text()) for p in paths}


# ---------------------------------------------------------------------------
# Run names and cell parsing
# ---------------------------------------------------------------------------

class TestNaming:
    def test_run_name(self):
        assert make_configs.run_name("gpu", 0.8, 20, 4, 4126) == \
            "gpu_a0p8_N20_C4_s4126"
        assert make_configs.run_name("gpu_oracle", 0.8, 30, 8, 4126, "_det1") == \
            "gpu_oracle_a0p8_N30_C8_s4126_det1"

    def test_alpha_tag_keeps_the_axis_visible(self):
        """`%g` would render 1.0 as a bare `a1` and lose the value."""
        assert make_configs.alpha_tag(1.0) == "a1p0"
        assert make_configs.alpha_tag(0.8) == "a0p8"

    @pytest.mark.parametrize("arm", ALL_ARMS)
    def test_cell_and_arm_round_trip(self, arm, tmp_path):
        """`gpu_oracle` contains an underscore, so splitting on the first one --
        what the older generators do -- would read the arm as `gpu`."""
        path = tmp_path / f"{make_configs.run_name(arm, 0.8, 20, 4, 4126)}.toml"
        assert make_configs.arm_of(path) == arm
        assert make_configs.cell_of(path) == "a0p8_N20_C4_s4126"

    def test_gpu_and_gpu_oracle_are_the_same_cell(self, tmp_path):
        a = tmp_path / "gpu_a0p8_N20_C4_s4126.toml"
        b = tmp_path / "gpu_oracle_a0p8_N20_C4_s4126.toml"
        assert make_configs.cell_of(a) == make_configs.cell_of(b)

    def test_unparseable_name_is_fatal(self, tmp_path):
        with pytest.raises(SystemExit, match="cannot parse a cell"):
            make_configs.cell_of(tmp_path / "nonsense.toml")


# ---------------------------------------------------------------------------
# The port plan
# ---------------------------------------------------------------------------

class TestPortPlan:
    def test_every_planned_run_has_a_unique_port(self):
        ports = {}
        for alpha in make_configs.ALPHA_SLOT:
            for arm in ALL_ARMS:
                for n in ALL_N:
                    for c in ALL_C:
                        for seed in make_configs.SEEDS:
                            key = (alpha, arm, n, c, seed)
                            port = make_configs.port_for(arm, alpha, n, c, seed)
                            assert port not in ports, (key, ports.get(port))
                            ports[port] = key
        assert len(ports) == (
            len(make_configs.ALPHA_SLOT) * len(ALL_ARMS) * len(ALL_N)
            * len(ALL_C) * len(make_configs.SEEDS)
        )

    def test_tagged_runs_cannot_collide_with_real_ones(self):
        real = {make_configs.port_for(a, alpha, n, c, s)
                for a in ALL_ARMS for alpha in make_configs.ALPHA_SLOT
                for n in ALL_N for c in ALL_C for s in make_configs.SEEDS}
        tagged = {make_configs.port_for(a, 0.8, n, c, s, tag)
                  for a in ALL_ARMS for n in ALL_N for c in ALL_C
                  for s in make_configs.SEEDS
                  for tag in make_configs.TAG_SLOT}
        assert not (real & tagged)

    def test_two_tags_of_one_config_get_different_ports(self):
        """_det1 and _det2 are the SAME config run twice. Sharing a port is a
        silent hang, and this is the pair most likely to be launched together."""
        one = make_configs.port_for("gpu", 0.8, 20, 4, 4126, "_det1")
        two = make_configs.port_for("gpu", 0.8, 20, 4, 4126, "_det2")
        assert one != two

    def test_every_tagged_run_has_a_unique_port(self):
        ports = {}
        for tag in make_configs.TAG_SLOT:
            for arm in ALL_ARMS:
                for n in ALL_N:
                    for c in ALL_C:
                        for seed in make_configs.SEEDS:
                            port = make_configs.port_for(arm, 0.8, n, c, seed, tag)
                            key = (tag, arm, n, c, seed)
                            assert port not in ports, (key, ports.get(port))
                            ports[port] = key

    def test_an_unregistered_tag_is_refused(self):
        with pytest.raises(SystemExit, match="no slot in the port plan"):
            make_configs.port_for("gpu", 0.8, 20, 4, 4126, "_whatever")

    def test_the_tag_block_fits_under_the_port_ceiling(self):
        """Registering _det3 and _det4 pushed the probe block past 65535 when
        each tag owned 2000 ports. The offset under a tag maxes at 913, so the
        block is 1000 wide."""
        offsets = [make_configs.port_for(a, 0.8, n, c, s, "_det1")
                   - make_configs.PROBE_PORT_BASE
                   for a in ALL_ARMS for n in ALL_N for c in ALL_C
                   for s in make_configs.SEEDS]
        assert max(offsets) < 1000

    def test_no_port_exceeds_the_16_bit_maximum(self):
        ports = [make_configs.port_for(a, alpha, n, c, s, tag)
                 for a in ALL_ARMS for alpha in make_configs.ALPHA_SLOT
                 for n in ALL_N for c in ALL_C for s in make_configs.SEEDS
                 for tag in ("", *make_configs.TAG_SLOT)]
        assert max(ports) <= 65535

    def test_block_does_not_overlap_other_programmes(self):
        """Blocks in use: surgical to 31015, separability n20 to 37325,
        n100 41000-50323. This programme starts at 51000."""
        ports = [make_configs.port_for(a, alpha, n, c, s, tag)
                 for a in ALL_ARMS for alpha in make_configs.ALPHA_SLOT
                 for n in ALL_N for c in ALL_C for s in make_configs.SEEDS
                 for tag in ("", *make_configs.TAG_SLOT)]
        assert min(ports) > 50323
        assert max(ports) < 65535

    @pytest.mark.parametrize("bad", [
        {"alpha": 0.5}, {"arm": "gpu_orcale"}, {"per_round": 3},
        {"total_clients": 50}, {"seed": 9999},
    ])
    def test_unregistered_values_are_refused_not_folded_in(self, bad):
        kwargs = dict(arm="gpu", alpha=0.8, total_clients=20, per_round=4,
                      seed=4126)
        kwargs.update(bad)
        with pytest.raises(SystemExit, match="no slot in the port plan"):
            make_configs.port_for(**kwargs)


# ---------------------------------------------------------------------------
# staleness_bound follows C
# ---------------------------------------------------------------------------

class TestStalenessBound:
    def test_c4_reproduces_the_historical_flat_value(self):
        """25 is what every finished cell at C=4 ran with; the rule must not
        move that cell while making the other concurrencies right."""
        assert make_configs.staleness_bound_for(4) == 25

    @pytest.mark.parametrize("c,expected", [
        (1, 7), (2, 13), (4, 25), (6, 38), (8, 50), (10, 63),
    ])
    def test_scales_with_c(self, c, expected):
        assert make_configs.staleness_bound_for(c) == expected

    def test_bound_clears_the_slowest_tier_at_every_c(self, fleets_built):
        """The bound must sit well above tau_hat of the SLOWEST CLIENT, not just
        the tier mean -- crossing it flushes every pending client into one
        aggregation and a round stops meaning one update (HANDOFF 4.9)."""
        import json
        for n in ALL_N:
            manifest = json.loads(
                (make_configs.fleet_dir(n) / "manifest.json").read_text()
            )
            slowest = manifest["tier_stats"]["xavier"]["max_s"]
            for c in ALL_C:
                t_round = manifest["E_T_s"] / c
                tau_hat = slowest / t_round
                bound = make_configs.staleness_bound_for(c)
                assert bound > 2.0 * tau_hat, (n, c, tau_hat, bound)


# ---------------------------------------------------------------------------
# The arms of a cell
# ---------------------------------------------------------------------------

class TestArmsOfACell:
    def test_all_five_arms_render(self, cell):
        assert set(cell) == set(ALL_ARMS)

    def test_arms_share_the_fleet(self, cell):
        truths = {
            arm: config["clients"]["simulation_distribution"]["csv_path"]
            for arm, config in cell.items()
        }
        assert len(set(truths.values())) == 1

    def test_arms_share_every_training_knob(self, cell):
        for key in ("rounds", "max_concurrency", "model_name", "epochs",
                    "batch_size", "target_accuracy"):
            values = {config["trainer"][key] for config in cell.values()}
            assert len(values) == 1, key
        for key in ("partition_size", "concentration", "random_seed"):
            values = {config["data"][key] for config in cell.values()}
            assert len(values) == 1, key

    def test_only_cpu_holds_a_different_tier_file(self, cell):
        beliefs = {arm: Path(config["gpu_tier"]["tiers_csv"]).name
                   for arm, config in cell.items()}
        assert beliefs["cpu"] == "tiers_cpu.csv"
        for arm in ("baseline", "gpu", "observed", "gpu_oracle"):
            assert beliefs[arm] == "tiers_estimated_15x.csv"

    def test_predictors_are_the_arm(self, cell):
        assert "staleness_predictor" not in cell["baseline"]["server"]
        assert cell["gpu"]["server"]["staleness_predictor"] == "gpu_tops"
        assert cell["cpu"]["server"]["staleness_predictor"] == "cpu"
        assert cell["observed"]["server"]["staleness_predictor"] == "observed"
        assert cell["gpu_oracle"]["server"]["staleness_predictor"] == "gpu_oracle"

    def test_baseline_carries_no_key_its_server_never_reads(self, cell):
        for key in ("polaris_alpha", "polaris_initial_gradient_bound",
                    "polaris_initial_staleness"):
            assert key not in cell["baseline"]["server"], key

    def test_only_gpu_oracle_holds_ground_truth(self, cell):
        for arm, config in cell.items():
            has_key = "oracle_sleep_csv" in config["gpu_tier"]
            assert has_key == (arm == "gpu_oracle"), arm

    def test_the_oracle_believes_exactly_what_the_clients_do(self, cell):
        config = cell["gpu_oracle"]
        assert (config["gpu_tier"]["oracle_sleep_csv"]
                == config["clients"]["simulation_distribution"]["csv_path"])

    def test_seed_is_written_to_all_three_places(self, cell):
        for arm, config in cell.items():
            assert config["clients"]["random_seed"] == 4126, arm
            assert config["server"]["random_seed"] == 4126, arm
            assert config["data"]["random_seed"] == 4126, arm

    def test_avg_training_time_is_present_and_zero(self, cell):
        """Absent, the first client report raises; non-zero, it is an additive
        constant identical for every client and the tier gap collapses toward 1."""
        for arm, config in cell.items():
            assert config["clients"]["avg_training_time"] == 0.0, arm

    def test_max_concurrency_equals_per_round(self, cell):
        for arm, config in cell.items():
            assert config["trainer"]["max_concurrency"] == \
                config["clients"]["per_round"], arm


class TestValidateCatchesDrift:
    def test_a_hand_edited_training_knob_is_caught(self, tmp_path, fleets_built):
        paths = [make_configs.render(arm, 0.8, tmp_path, 4126, 20, 4)
                 for arm in ("gpu", "cpu")]
        make_configs.validate(paths)                      # clean to begin with
        victim = next(p for p in paths if make_configs.arm_of(p) == "cpu")
        victim.write_text(victim.read_text().replace("batch_size = 64",
                                                     "batch_size = 128"))
        with pytest.raises(SystemExit, match="must be shared inside cell"):
            make_configs.validate(paths)

    def test_the_oracle_key_on_another_arm_is_caught(self, tmp_path, fleets_built):
        path = make_configs.render("gpu", 0.8, tmp_path, 4126, 20, 4)
        path.write_text(path.read_text() + '\noracle_sleep_csv = "/x.csv"\n')
        with pytest.raises(SystemExit, match="Only gpu_oracle"):
            make_configs.validate([path])

    def test_a_missing_oracle_key_is_caught(self, tmp_path, fleets_built):
        path = make_configs.render("gpu_oracle", 0.8, tmp_path, 4126, 20, 4)
        kept = [l for l in path.read_text().splitlines()
                if not l.startswith("oracle_sleep_csv")]
        path.write_text("\n".join(kept))
        with pytest.raises(SystemExit, match="Only gpu_oracle"):
            make_configs.validate([path])

    def test_an_oracle_pointed_at_another_fleet_is_caught(self, tmp_path, fleets_built):
        path = make_configs.render("gpu_oracle", 0.8, tmp_path, 4126, 20, 4)
        text = path.read_text()
        original = tomllib.loads(text)["gpu_tier"]["oracle_sleep_csv"]
        path.write_text(text.replace(
            f'oracle_sleep_csv = "{original}"',
            'oracle_sleep_csv = "/elsewhere/sleep_times.csv"',
        ))
        with pytest.raises(SystemExit, match="believe exactly what the clients do"):
            make_configs.validate([path])

    def test_duplicate_ports_are_caught(self, tmp_path, fleets_built):
        a = make_configs.render("gpu", 0.8, tmp_path, 4126, 20, 4)
        b = make_configs.render("cpu", 0.8, tmp_path, 4126, 20, 4)
        port_a = tomllib.loads(a.read_text())["server"]["port"]
        port_b = tomllib.loads(b.read_text())["server"]["port"]
        b.write_text(b.read_text().replace(f"port = {port_b}", f"port = {port_a}"))
        with pytest.raises(SystemExit, match="hang silently"):
            make_configs.validate([a, b])


# ---------------------------------------------------------------------------
# The derived numbers written into the header
# ---------------------------------------------------------------------------

class TestDerivedNumbers:
    def test_header_states_the_fleet_and_its_clock(self, tmp_path, fleets_built):
        path = make_configs.render("gpu", 0.8, tmp_path, 4126, 20, 4)
        header = path.read_text().split("[general]")[0]
        assert "n20_x10-o7-t3_xa15_p2500_j0p149_s4126" in header
        assert "E[T] = 7.9706 s" in header
        assert "1.992649" in header          # T_round = E[T]/4

    def test_unbuilt_fleet_is_a_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(make_configs, "FLEETS_DIR", tmp_path / "empty")
        with pytest.raises(SystemExit, match="build_fleet.py"):
            make_configs.fleet_dir(20)

    def test_unregistered_n_has_no_partition_size(self, tmp_path):
        with pytest.raises(SystemExit, match="no partition_size"):
            make_configs.render("gpu", 0.8, tmp_path, 4126, 50, 4)

    @pytest.mark.parametrize("n", ALL_N)
    def test_partition_sizes_hold_the_dataset_roughly_constant(self, n):
        total = n * make_configs.PARTITION_SIZE[n]
        assert abs(total - 50_000) <= 10, (n, total)

    def test_fleet_dir_matches_what_build_fleet_names(self, fleets_built):
        for n in ALL_N:
            counts = make_fleet.counts_for(n, make_configs.FLEET_MIX)
            expected = build_fleet.fleet_dir_name(
                counts, make_configs.ANCHOR_TIER, make_configs.ANCHOR_SLEEP,
                make_configs.PARTITION_SIZE[n], make_configs.JITTER_SIGMA,
                make_configs.FLEET_SEED,
            )
            assert make_configs.fleet_dir(n).name == expected


class TestExcludeProcessingTime:
    """Stage 3 Step 0's on/off switch, rendered into the config template.

    A run's own copy of its config (`config_used.toml`) is the only authority on
    whether it carried the patch, so the generator must be able to render EITHER
    value on request -- never bake in a silent default that a later invocation
    cannot override.
    """

    def test_default_follows_the_module_constant(self, tmp_path, fleets_built):
        path = make_configs.render("gpu", 0.8, tmp_path, 4126, 20, 4)
        rendered = tomllib.loads(path.read_text())["server"]
        assert rendered["exclude_processing_time_from_clock"] == \
            make_configs.EXCLUDE_PROCESSING_TIME

    def test_explicit_true_is_honoured(self, tmp_path, fleets_built):
        path = make_configs.render("gpu", 0.8, tmp_path, 4126, 20, 4,
                                   exclude_processing_time=True)
        rendered = tomllib.loads(path.read_text())["server"]
        assert rendered["exclude_processing_time_from_clock"] is True

    def test_explicit_false_is_honoured_even_if_the_default_is_true(
        self, tmp_path, fleets_built, monkeypatch
    ):
        monkeypatch.setattr(make_configs, "EXCLUDE_PROCESSING_TIME", True)
        path = make_configs.render("gpu", 0.8, tmp_path, 4126, 20, 4,
                                   exclude_processing_time=False)
        rendered = tomllib.loads(path.read_text())["server"]
        assert rendered["exclude_processing_time_from_clock"] is False

    def test_the_key_matches_what_runtime_patches_reads(self):
        """The generator and the reader must agree on the exact key name, or the
        patch silently never fires no matter what a config says."""
        import runtime_patches
        template = (make_configs.HERE / "make_configs.py").read_text()
        assert (f"{runtime_patches.CONFIG_KEY} = "
                "{exclude_processing_time}") in template

    def test_cli_default_matches_the_module_constant(self):
        args = make_configs.build_parser().parse_args(["--alphas", "0.8"])
        assert args.exclude_processing_time == make_configs.EXCLUDE_PROCESSING_TIME

    def test_cli_no_exclude_flag_turns_it_off(self):
        args = make_configs.build_parser().parse_args(
            ["--alphas", "0.8", "--no-exclude-processing-time"]
        )
        assert args.exclude_processing_time is False

    def test_cli_exclude_flag_turns_it_on_explicitly(self):
        args = make_configs.build_parser().parse_args(
            ["--alphas", "0.8", "--exclude-processing-time"]
        )
        assert args.exclude_processing_time is True
