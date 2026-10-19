"""Tests for fedasync_seeded.py -- the entry point.

The group that matters is `TestArgvSurvivesToChildren`. PLATO starts its clients
with `mp.Process(target=..., ...)` under spawn, and spawn snapshots `sys.argv` at
`proc.start()`, which happens long after this module is imported. An entry point
that consumed `--arm` from argv and did not put it back produced:

    server   ok, seeded, T_round correct
    clients  "[fedasync_seeded] --arm is required" x4, all dead
    result   the server waits forever for clients that never connect

No traceback, no non-zero exit, no failed run -- a hang. Only a real launch found
it, and only these tests keep it found.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

DATASET_DIR = Path(__file__).resolve().parents[1]

# fedasync_seeded imports PLATO at module level, which needs a config and a
# sanitised argv, so the whole module is exercised through subprocesses rather
# than imported here.
SOURCE = (DATASET_DIR / "fedasync_seeded.py").read_text()


# ---------------------------------------------------------------------------
# strip_arm, in isolation
# ---------------------------------------------------------------------------

def strip_arm(argv):
    """Run the real strip_arm without importing PLATO.

    The function is copied out by exec-ing just its source and the ARMS tuple it
    closes over; importing the module would pull in torch and PLATO for a pure
    argv test.
    """
    namespace: dict = {}
    start = SOURCE.index("POLARIS_ARMS = ")
    end = SOURCE.index("# =====", start)
    exec(SOURCE[start:end], namespace)  # noqa: S102 - the file under test
    return namespace["strip_arm"](argv)


class TestStripArm:
    @pytest.mark.parametrize("argv,arm", [
        (["run.py", "--arm", "gpu", "-c", "x.toml"], "gpu"),
        (["run.py", "--arm=gpu_oracle", "-c", "x.toml"], "gpu_oracle"),
        (["run.py", "-c", "x.toml", "--arm", "baseline"], "baseline"),
        (["run.py", "--arm", "cpu"], "cpu"),
        (["run.py", "--arm", "observed", "-c", "x.toml"], "observed"),
    ])
    def test_reads_every_form(self, argv, arm):
        assert strip_arm(argv)[0] == arm

    def test_removes_the_flag_and_keeps_the_rest(self):
        arm, cleaned = strip_arm(["run.py", "--arm", "gpu", "-c", "x.toml"])
        assert cleaned == ["run.py", "-c", "x.toml"]

    def test_does_not_mutate_the_input(self):
        argv = ["run.py", "--arm", "gpu", "-c", "x.toml"]
        original = list(argv)
        strip_arm(argv)
        assert argv == original, "strip_arm mutated sys.argv in place"

    def test_missing_arm_is_fatal(self):
        with pytest.raises(SystemExit, match="--arm is required"):
            strip_arm(["run.py", "-c", "x.toml"])

    def test_unknown_arm_is_fatal(self):
        """A typo must not silently become another arm of the same sweep."""
        with pytest.raises(SystemExit, match="not recognised"):
            strip_arm(["run.py", "--arm", "gpu_orcale", "-c", "x.toml"])

    def test_all_five_arms_are_accepted(self):
        for arm in ("baseline", "gpu", "cpu", "observed", "gpu_oracle"):
            assert strip_arm(["run.py", "--arm", arm])[0] == arm


# ---------------------------------------------------------------------------
# The regression that a real launch found
# ---------------------------------------------------------------------------

class TestArgvSurvivesToChildren:
    """`--arm` must be back in sys.argv by the time PLATO spawns a client."""

    SCRIPT = textwrap.dedent("""
        import sys
        # Exactly the sequence fedasync_seeded.py performs at module level.
        sys.path.insert(0, {dataset!r})
        source = open({entry!r}).read()

        import multiprocessing as mp

        ARMS = ("baseline", "gpu", "cpu", "observed", "gpu_oracle")
        ns = {{}}
        start = source.index("POLARIS_ARMS = ")
        end = source.index("# =====", start)
        exec(source[start:end], ns)

        ARM, cleaned = ns["strip_arm"](sys.argv)
        saved = list(sys.argv)
        sys.argv = cleaned
        # ... PLATO imports would happen here, parsing the cleaned argv ...
        assert "--arm" not in sys.argv, "argv was not cleaned for the import"
        sys.argv = saved            # THE LINE UNDER TEST

        def child():
            print("CHILD_ARGV", " ".join(sys.argv), flush=True)

        if __name__ == "__main__":
            mp.set_start_method("spawn", force=True)
            p = mp.Process(target=child); p.start(); p.join()
            print("CHILD_EXIT", p.exitcode, flush=True)
    """)

    def run(self, tmp_path, restore=True):
        script = tmp_path / "spawn_entry.py"
        body = self.SCRIPT.format(
            dataset=str(DATASET_DIR),
            entry=str(DATASET_DIR / "fedasync_seeded.py"),
        )
        if not restore:
            body = body.replace("sys.argv = saved            # THE LINE UNDER TEST",
                                "pass  # the bug: argv never restored")
        script.write_text(body)
        proc = subprocess.run(
            [sys.executable, str(script), "--arm", "gpu", "-c", "x.toml"],
            capture_output=True, text=True, cwd=tmp_path,
        )
        return proc

    def test_child_inherits_the_arm_flag(self, tmp_path):
        proc = self.run(tmp_path, restore=True)
        assert proc.returncode == 0, proc.stderr
        child_argv = [l for l in proc.stdout.splitlines()
                      if l.startswith("CHILD_ARGV")][0]
        assert "--arm gpu" in child_argv
        assert "CHILD_EXIT 0" in proc.stdout

    def test_without_the_restore_the_child_dies(self, tmp_path):
        """The bug, reproduced exactly as the failing launch showed it.

        The child does not merely lose the flag: re-importing the entry module
        under spawn re-runs `strip_arm`, which finds no `--arm` and raises, so
        the process is gone before it does any work. The parent sees only a
        non-zero child exit code it never checks -- hence the silent hang.
        """
        proc = self.run(tmp_path, restore=False)
        assert "CHILD_ARGV" not in proc.stdout, "the child should not get that far"
        assert "--arm is required" in proc.stderr
        exit_line = [l for l in proc.stdout.splitlines()
                     if l.startswith("CHILD_EXIT")][0]
        assert exit_line != "CHILD_EXIT 0"

    def test_the_entry_point_restores_argv(self):
        """Read the file itself, so the guarantee cannot be deleted quietly."""
        assert "sys.argv = _ARGV_WITH_ARM" in SOURCE
        restore_at = SOURCE.index("sys.argv = _ARGV_WITH_ARM")
        import_at = SOURCE.index("import fedasync_algorithm")
        assert restore_at > import_at, \
            "argv must be restored AFTER the PLATO imports, not before"


# ---------------------------------------------------------------------------
# Structure the seeding depends on
# ---------------------------------------------------------------------------

class TestModuleStructure:
    def test_seeding_runs_before_the_plato_imports(self):
        """torch must be seeded before anything initialises its RNG state."""
        seed_at = SOURCE.index("SEED_RECORD = seeding.seed_everything()")
        import_at = SOURCE.index("import fedasync_algorithm")
        assert seed_at < import_at

    def test_seeding_is_at_module_level_not_inside_main(self):
        """Inside main() it would run in the server only, and the client
        processes -- which do all the training -- would stay unseeded."""
        line = next(l for l in SOURCE.splitlines()
                    if l.strip().startswith("SEED_RECORD ="))
        assert not line.startswith(" "), "seed_everything() is indented"

    def test_arm_is_stripped_before_the_plato_imports(self):
        strip_at = SOURCE.index("ARM, _ARGV_WITHOUT_ARM = strip_arm")
        import_at = SOURCE.index("import fedasync_algorithm")
        assert strip_at < import_at

    def test_every_arm_maps_to_a_server(self):
        assert 'POLARIS_ARMS = ("gpu", "cpu", "observed", "gpu_oracle")' in SOURCE
        assert 'ARMS = ("baseline",) + POLARIS_ARMS' in SOURCE

    def test_the_callback_logs_the_seeding_record(self):
        """Seeding happens before Config installs the run's FileHandler, so the
        callback is the first place a client process can record that it ran."""
        assert "seeding.log_seeding()" in SOURCE
