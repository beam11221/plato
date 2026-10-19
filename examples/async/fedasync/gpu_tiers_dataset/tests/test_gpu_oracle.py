"""Tests for the gpu_oracle arm.

Two edits outside this folder are under test, and BOTH must be inert unless the
new config key is present -- `../surgical/`, `../separability/`,
`../separability_n100/` and `../sumo_close_loop/` all run this same code and are
being kept as the oracle branch for reference:

    gpu_tiers/gpu_tier_sleep.py          [gpu_tier] oracle_sleep_csv
    fedasync_polaris_vi_server.py        staleness_predictor = "gpu_oracle"

`TestDefaultOffIsByteIdentical` is the group that protects them.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DATASET_DIR = Path(__file__).resolve().parents[1]
FEDASYNC = DATASET_DIR.parent

# sys.path is set up by conftest.py, in an order that keeps THIS folder ahead of
# ../gpu_tiers/ -- both define a top-level `make_configs`.

gpu_tier_sleep = pytest.importorskip("gpu_tier_sleep")


def _import_server_module():
    """Import fedasync_polaris_vi_server with an argv PLATO's Config can parse.

    `plato/servers/base.py:382` declares `def start(self, port=Config().server.port)`
    and a default argument is evaluated at class-definition time, so importing
    anything that reaches plato.servers.base runs `Config()` -- and with it
    `parser.parse_args()` over the whole of `sys.argv`. Under pytest that argv is
    `['pytest', 'tests/', '-q']` and argparse exits 2, taking the test session
    with it. This is the same trap `fedasync_seeded.resolve_arm` documents, met
    from the other side.

    A minimal TOML is written and handed over via a temporary argv so the import
    succeeds; the tests below need only module constants and
    `check_oracle_wiring`'s injectable argument, none of which read Config.
    """
    import tempfile
    import textwrap

    scratch = Path(tempfile.mkdtemp())
    config = scratch / "minimal.toml"
    # base_path must point into the scratch directory: Config() creates it at
    # import time, and the default "./runtime" would litter the repo with a
    # directory and a server log on every test run.
    config.write_text(textwrap.dedent(f"""
        [general]
        base_path = "{scratch}"

        [clients]
        total_clients = 20
        per_round = 4
        [server]
        address = "127.0.0.1"
        port = 39999
        [data]
        partition_size = 2500
        [trainer]
        type = "basic"
        rounds = 1
        epochs = 1
        model_name = "resnet_18"
        [algorithm]
        type = "fedavg"
        [parameters]
        [results]
        types = "round, accuracy"
    """))

    saved = sys.argv
    sys.argv = [saved[0], "-c", str(config)]
    try:
        return pytest.importorskip("fedasync_polaris_vi_server")
    finally:
        sys.argv = saved


server_module = _import_server_module()

FLEET = DATASET_DIR / "fleets" / "n20_x10-o7-t3_xa15_p2500_j0p180_s4126"
PARTITION = 2500
EPOCHS = 1


@pytest.fixture(autouse=True)
def reset_module_caches():
    """gpu_tier_sleep caches every resolved config value in a module global.

    Each of these is a "resolved once per process" cache, which is right in a
    run and wrong in a test session, so every test starts from unresolved.
    """
    def clear():
        gpu_tier_sleep._computer = None
        gpu_tier_sleep._oracle_sleeps = None
        gpu_tier_sleep._predictor_scale = None
        gpu_tier_sleep._w_model_flops = None
        gpu_tier_sleep._comm_latency = None
        gpu_tier_sleep._comm_in_predictor = None
        gpu_tier_sleep._t_round_model = None
        gpu_tier_sleep._t_round_logged = False
    clear()
    yield
    clear()


@pytest.fixture
def fleet():
    if not FLEET.exists():
        pytest.skip(f"{FLEET} not built")
    return FLEET


@pytest.fixture
def true_sleeps(fleet):
    with (fleet / "sleep_times.csv").open(newline="") as fh:
        return {int(r["client_id"]): float(r["sleep_time"]) for r in csv.DictReader(fh)}


def configure(monkeypatch, fleet, *, oracle: bool):
    """Point gpu_tier_sleep at this fleet, with or without the oracle key.

    `gpu_tier_sleep` reads everything through `Config().config.gpu_tier`, so a
    SimpleNamespace standing in for that section exercises the real resolution
    code without needing PLATO's Config singleton or a TOML on disk.
    """
    gpu_tier = SimpleNamespace(
        tiers_csv=str(DATASET_DIR / "tiers" / "tiers_estimated_15x.csv"),
        clients_csv=str(fleet / "clients.csv"),
        anchor_tier="xavier",
        anchor_sleep=15,
        t_round_model="queue",
    )
    if oracle:
        gpu_tier.oracle_sleep_csv = str(fleet / "sleep_times.csv")

    fake = SimpleNamespace(
        config=SimpleNamespace(gpu_tier=gpu_tier),
        data=SimpleNamespace(partition_size=PARTITION),
        trainer=SimpleNamespace(epochs=EPOCHS),
    )
    monkeypatch.setattr("plato.config.Config", lambda *a, **k: fake)
    return gpu_tier


# ---------------------------------------------------------------------------
# Absent key -> the code path every existing folder runs
# ---------------------------------------------------------------------------

class TestDefaultOffIsByteIdentical:
    def test_no_key_means_no_oracle(self, monkeypatch, fleet):
        configure(monkeypatch, fleet, oracle=False)
        assert gpu_tier_sleep.oracle_sleeps() is None

    def test_gpu_arm_predicts_the_tier_mean_not_the_truth(
        self, monkeypatch, fleet, true_sleeps
    ):
        """The whole point of the split: gpu's belief is constant within a tier."""
        configure(monkeypatch, fleet, oracle=False)
        beliefs = {
            cid: gpu_tier_sleep.gpu_tops_predict_seconds(cid, PARTITION, EPOCHS)
            for cid in true_sleeps
        }
        assert {round(beliefs[c], 9) for c in range(1, 11)} == {15.0}     # xavier
        assert len({round(beliefs[c], 9) for c in range(11, 18)}) == 1    # orin
        assert len({round(beliefs[c], 9) for c in range(18, 21)}) == 1    # thor
        # ...and it is NOT the truth, which is what makes it an estimator.
        assert any(abs(beliefs[c] - true_sleeps[c]) > 1e-6 for c in true_sleeps)

    def test_absent_key_is_cached_without_touching_config_again(
        self, monkeypatch, fleet
    ):
        configure(monkeypatch, fleet, oracle=False)
        assert gpu_tier_sleep.oracle_sleeps() is None
        # The "resolved, and there is none" sentinel, not None.
        assert gpu_tier_sleep._oracle_sleeps is False
        assert gpu_tier_sleep.oracle_sleeps() is None

    def test_t_round_unchanged_without_the_key(self, monkeypatch, fleet):
        """E[T]/C over the datasheet tier times: 8.3231 / 4 = 2.080769 s, the
        number ../separability/README.md says the server must log."""
        configure(monkeypatch, fleet, oracle=False)
        cids = list(range(1, 21))
        sizes = {cid: PARTITION for cid in cids}
        t_round = gpu_tier_sleep.compute_pool_T_round_seconds(
            cids, sizes, EPOCHS, clients_per_round=4
        )
        assert t_round == pytest.approx(2.080769, abs=1e-6)


# ---------------------------------------------------------------------------
# Key present -> the oracle
# ---------------------------------------------------------------------------

class TestOracleBelief:
    def test_belief_equals_truth_for_every_client(
        self, monkeypatch, fleet, true_sleeps
    ):
        configure(monkeypatch, fleet, oracle=True)
        for cid, truth in true_sleeps.items():
            predicted = gpu_tier_sleep.gpu_tops_predict_seconds(cid, PARTITION, EPOCHS)
            assert predicted == pytest.approx(truth, abs=1e-9), cid

    def test_belief_varies_within_a_tier(self, monkeypatch, fleet):
        """Unlike gpu, the oracle distinguishes two units of the same model."""
        configure(monkeypatch, fleet, oracle=True)
        xavier = {
            round(gpu_tier_sleep.gpu_tops_predict_seconds(c, PARTITION, EPOCHS), 9)
            for c in range(1, 11)
        }
        assert len(xavier) == 10

    def test_t_round_is_computed_from_the_truth(self, monkeypatch, fleet, true_sleeps):
        """A half-oracle -- true t_client but tier-mean T_round -- would leave
        tau_hat wrong in the denominator. E[T] over the TRUE sleeps is 7.9180 s."""
        configure(monkeypatch, fleet, oracle=True)
        cids = sorted(true_sleeps)
        sizes = {cid: PARTITION for cid in cids}
        t_round = gpu_tier_sleep.compute_pool_T_round_seconds(
            cids, sizes, EPOCHS, clients_per_round=4
        )
        expected = sum(true_sleeps.values()) / len(true_sleeps) / 4
        assert t_round == pytest.approx(expected, abs=1e-9)
        assert t_round == pytest.approx(7.918013 / 4, abs=1e-6)
        # ...and it differs from the datasheet value the gpu arm uses.
        assert abs(t_round - 2.080769) > 1e-3

    def test_pool_mean_is_computed_from_the_truth(self, monkeypatch, fleet, true_sleeps):
        configure(monkeypatch, fleet, oracle=True)
        cids = sorted(true_sleeps)
        mean = gpu_tier_sleep.compute_pool_mean_T_seconds(
            cids, {cid: PARTITION for cid in cids}, EPOCHS
        )
        assert mean == pytest.approx(sum(true_sleeps.values()) / len(true_sleeps), abs=1e-9)

    def test_all_three_entry_points_agree(self, monkeypatch, fleet, true_sleeps):
        """The selector reads busy residuals from gpu_tops_predict_seconds and its
        denominator from the pool functions. If those held different beliefs the
        arm would be a half-oracle no table could describe."""
        configure(monkeypatch, fleet, oracle=True)
        cids = sorted(true_sleeps)
        sizes = {cid: PARTITION for cid in cids}
        per_client = [
            gpu_tier_sleep.gpu_tops_predict_seconds(c, PARTITION, EPOCHS) for c in cids
        ]
        assert gpu_tier_sleep.compute_pool_mean_T_seconds(
            cids, sizes, EPOCHS
        ) == pytest.approx(sum(per_client) / len(per_client), abs=1e-9)
        assert gpu_tier_sleep.compute_pool_T_round_seconds(
            cids, sizes, EPOCHS, clients_per_round=4
        ) == pytest.approx(
            gpu_tier_sleep.compute_t_round_from_times(per_client, 4), abs=1e-9
        )

    def test_oracle_ignores_partition_size_argument(self, monkeypatch, fleet, true_sleeps):
        """Truth is a measured number, not a formula: it does not rescale with a
        partition size passed at the call site. preflight.py is what checks the
        config's partition_size matches the fleet the oracle file came from."""
        configure(monkeypatch, fleet, oracle=True)
        a = gpu_tier_sleep.gpu_tops_predict_seconds(1, PARTITION, EPOCHS)
        b = gpu_tier_sleep.gpu_tops_predict_seconds(1, PARTITION * 2, EPOCHS)
        assert a == b == pytest.approx(true_sleeps[1], abs=1e-9)

    def test_missing_file_raises(self, monkeypatch, fleet, tmp_path):
        gpu_tier = configure(monkeypatch, fleet, oracle=True)
        gpu_tier.oracle_sleep_csv = str(tmp_path / "absent.csv")
        with pytest.raises(ValueError, match="does not exist"):
            gpu_tier_sleep.oracle_sleeps()

    def test_empty_file_raises(self, monkeypatch, fleet, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("client_id,sleep_time\n")
        gpu_tier = configure(monkeypatch, fleet, oracle=True)
        gpu_tier.oracle_sleep_csv = str(empty)
        with pytest.raises(ValueError, match="no rows"):
            gpu_tier_sleep.oracle_sleeps()

    def test_client_missing_from_oracle_file_raises(self, monkeypatch, fleet, tmp_path):
        """A fleet/oracle mismatch must not silently predict something plausible."""
        partial = tmp_path / "partial.csv"
        partial.write_text("client_id,sleep_time\n1,15.0\n")
        gpu_tier = configure(monkeypatch, fleet, oracle=True)
        gpu_tier.oracle_sleep_csv = str(partial)
        with pytest.raises(KeyError, match="not in the oracle sleep"):
            gpu_tier_sleep.gpu_tops_predict_seconds(2, PARTITION, EPOCHS)


# ---------------------------------------------------------------------------
# The arm / key cross-check
# ---------------------------------------------------------------------------

class TestOracleWiringGuard:
    def test_gpu_oracle_is_a_recognised_predictor(self):
        assert "gpu_oracle" in server_module.STALENESS_PREDICTORS
        assert server_module.resolve_staleness_predictor("gpu_oracle") == "gpu_oracle"
        assert server_module.resolve_staleness_predictor(" GPU_Oracle ") == "gpu_oracle"

    def test_gpu_oracle_uses_the_seconds_path(self):
        assert "gpu_oracle" in server_module.SECONDS_PREDICTORS
        assert "observed" not in server_module.SECONDS_PREDICTORS

    def test_existing_arms_still_resolve(self):
        for name in ("gpu_tops", "cpu", "observed"):
            assert server_module.resolve_staleness_predictor(name) == name
        assert server_module.resolve_staleness_predictor(None) == "observed"

    def test_unknown_arm_still_raises(self):
        with pytest.raises(ValueError):
            server_module.resolve_staleness_predictor("gpu_orcale")

    def test_oracle_arm_without_the_key_is_fatal(self):
        with pytest.raises(ValueError, match="requires"):
            server_module.check_oracle_wiring("gpu_oracle", oracle_csv=None)

    @pytest.mark.parametrize("arm", ["gpu_tops", "cpu", "observed"])
    def test_other_arms_with_the_key_are_fatal(self, arm):
        with pytest.raises(ValueError, match="only 'gpu_oracle'"):
            server_module.check_oracle_wiring(arm, oracle_csv="/some/sleep_times.csv")

    def test_matching_pairs_pass(self):
        server_module.check_oracle_wiring("gpu_oracle", oracle_csv="/x/sleep_times.csv")
        for arm in ("gpu_tops", "cpu", "observed"):
            server_module.check_oracle_wiring(arm, oracle_csv=None)
