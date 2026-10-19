"""Tests for the generated inputs: the fleet remap and the four arm configs."""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path

import pytest

import make_sumo_configs as make_configs
import remap_fleet

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
INPUT_DIR = EXPERIMENT_DIR / "input"
CONFIG_DIR = EXPERIMENT_DIR / "configs"


class TestBuildMapping:
    def test_pairs_vehicles_with_clients_of_the_same_tier(self):
        vehicles = {"veh_00": "xavier", "veh_01": "thor", "veh_02": "xavier"}
        clients = {1: "xavier", 2: "xavier", 3: "thor"}

        mapping = remap_fleet.build_mapping(vehicles, clients)

        assert mapping == {"veh_00": 1, "veh_02": 2, "veh_01": 3}

    def test_is_a_bijection(self):
        vehicles = {f"veh_{i:02d}": "xavier" for i in range(5)}
        clients = {i: "xavier" for i in range(1, 6)}

        mapping = remap_fleet.build_mapping(vehicles, clients)

        assert sorted(mapping.values()) == [1, 2, 3, 4, 5]

    def test_is_deterministic(self):
        vehicles = {"veh_02": "orin", "veh_00": "orin", "veh_01": "orin"}
        clients = {3: "orin", 1: "orin", 2: "orin"}

        first = remap_fleet.build_mapping(vehicles, clients)
        second = remap_fleet.build_mapping(dict(reversed(vehicles.items())), clients)

        assert first == second

    def test_refuses_a_tier_with_mismatched_counts(self):
        vehicles = {"veh_00": "xavier", "veh_01": "xavier"}
        clients = {1: "xavier"}

        with pytest.raises(SystemExit, match="compositions to match"):
            remap_fleet.build_mapping(vehicles, clients)

    def test_refuses_unknown_tier_names(self):
        vehicles = {"veh_00": "grace"}
        clients = {1: "xavier"}

        with pytest.raises(SystemExit, match="tier names differ"):
            remap_fleet.build_mapping(vehicles, clients)


@pytest.mark.skipif(
    not (INPUT_DIR / "mapping_n20.csv").exists(), reason="run build_inputs.sh n20"
)
class TestGeneratedFleet:
    """The three files describing the n20 fleet must agree with each other."""

    def _read(self, name: str) -> list[dict]:
        with (INPUT_DIR / name).open(newline="") as f:
            return list(csv.DictReader(f))

    def test_every_client_has_exactly_one_vehicle(self):
        rows = self._read("mapping_n20.csv")
        assert sorted(int(r["client_id"]) for r in rows) == list(range(1, 21))
        assert len({r["veh_id"] for r in rows}) == 20

    def test_the_mapping_preserves_tiers(self):
        clients = {
            int(r["client_id"]): r["tier_id"] for r in self._read("clients_n20.csv")
        }
        for row in self._read("mapping_n20.csv"):
            assert clients[int(row["client_id"])] == row["tier_id"]

    def test_the_composition_is_50_35_15(self):
        tiers = [r["tier_id"] for r in self._read("clients_n20.csv")]
        assert tiers.count("xavier") == 10
        assert tiers.count("orin") == 7
        assert tiers.count("thor") == 3

    def test_coverage_covers_every_client(self):
        covered = {int(r["client_id"]) for r in self._read("coverage_n20.csv")}
        assert covered == set(range(1, 21))

    def test_sumo_tier_column_is_gone_from_the_coverage_table(self):
        # tiers.csv is the only source of truth for hardware; leaving SUMO's own
        # column in would invite something downstream to read it.
        with (INPUT_DIR / "coverage_n20.csv").open(newline="") as f:
            assert "tier" not in next(csv.reader(f))


@pytest.mark.skipif(
    not (CONFIG_DIR / "gpu_n20.toml").exists(), reason="run build_inputs.sh n20"
)
class TestGeneratedConfigs:
    def _load(self, arm: str, cell: str = "n20") -> dict:
        with (CONFIG_DIR / f"{arm}_{cell}.toml").open("rb") as f:
            return tomllib.load(f)

    def test_arms_differ_only_in_the_permitted_keys(self):
        # make_configs.validate() raises on failure; calling it here pins the
        # rule against the files actually on disk.
        paths = [CONFIG_DIR / f"{arm}_n20.toml" for arm in make_configs.ARMS]
        make_configs.validate(paths)

    @pytest.mark.parametrize(
        "arm,predictor",
        [("gpu", "gpu_tops"), ("cpu", "cpu"), ("observed", "observed")],
    )
    def test_each_arm_declares_its_own_predictor(self, arm, predictor):
        assert self._load(arm)["server"]["staleness_predictor"] == predictor

    def test_the_baseline_carries_no_predictor_at_all(self):
        # fedasync_server.Server never reads the key; writing it would imply an
        # effect that does not exist.
        assert "staleness_predictor" not in self._load("baseline")["server"]

    def test_only_the_cpu_arm_believes_the_cpu_spec(self):
        for arm in ("baseline", "gpu", "observed"):
            assert self._load(arm)["gpu_tier"]["tiers_csv"].endswith(
                "tiers_estimated_15x.csv"
            )
        assert self._load("cpu")["gpu_tier"]["tiers_csv"].endswith("tiers_cpu.csv")

    def test_every_arm_sleeps_by_the_same_reality(self):
        csv_paths = {
            self._load(arm)["clients"]["simulation_distribution"]["csv_path"]
            for arm in make_configs.ARMS
        }
        assert len(csv_paths) == 1

    def test_ports_are_unique(self):
        ports = [self._load(arm)["server"]["port"] for arm in make_configs.ARMS]
        assert len(set(ports)) == len(ports)

    def test_the_settings_the_upload_gate_depends_on(self):
        clients = self._load("gpu")["clients"]
        assert clients["comm_simulation"] is True
        assert "compute_comm_time" not in clients
        assert clients["avg_training_time"] == 0.0

    def test_concurrency_matches_between_clients_and_trainer(self):
        config = self._load("gpu")
        assert config["clients"]["per_round"] == config["trainer"]["max_concurrency"]

    def test_the_anchor_is_fifteen_seconds(self):
        assert self._load("gpu")["gpu_tier"]["anchor_sleep"] == 15

    @pytest.mark.parametrize("arm", ["gpu", "cpu", "observed"])
    def test_polaris_keys_land_in_server_not_the_sub_table(self, arm):
        # HANDOFF section 4.2: a bare key written after
        # [server.staleness_weighting_function] belongs to that sub-table, and
        # the code then silently uses its defaults. The defaults equal these
        # values, so nothing would look wrong until someone changed one.
        server = self._load(arm)["server"]
        assert server["polaris_alpha"] == 10.0
        assert server["polaris_initial_gradient_bound"] == 0.5
        assert server["polaris_initial_staleness"] == 0.01
        assert set(server["staleness_weighting_function"]) == {"type", "a"}

    def test_the_baseline_carries_no_polaris_keys(self):
        server = self._load("baseline")["server"]
        assert not [key for key in server if key.startswith("polaris_")]

    def test_settings_match_the_surgical_experiment(self):
        """Everything except the anchor, the paths and the port is inherited."""
        surgical = EXPERIMENT_DIR.parent / "surgical" / "configs" / "gpu_xa100_gap15.toml"
        if not surgical.exists():
            pytest.skip("surgical configs not present")
        with surgical.open("rb") as f:
            reference = tomllib.load(f)
        ours = self._load("gpu")
        for section, key in (
            ("clients", "total_clients"),
            ("clients", "per_round"),
            ("data", "partition_size"),
            ("data", "concentration"),
            ("data", "random_seed"),
            ("trainer", "rounds"),
            ("trainer", "target_accuracy"),
            ("trainer", "max_concurrency"),
            ("server", "staleness_bound"),
            ("server", "minimum_clients_aggregated"),
            ("server", "mixing_hyperparameter"),
            ("server", "eval_interval"),
            ("gpu_tier", "t_round_model"),
        ):
            assert ours[section][key] == reference[section][key], f"{section}.{key}"
