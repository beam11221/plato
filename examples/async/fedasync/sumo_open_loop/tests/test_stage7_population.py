"""
Tests for Stage 7's slot allocator and identity policy.

The two tests this file exists for are `TestRecycleReset`'s: the stage prompt
calls the reset "the single thing most likely to be wrong and least likely to
announce itself", because if state leaks across a recycle then `p_recur = 0`
is quietly closed population and the whole cell measures nothing.

The fake context mirrors the real state layout exactly as
`fedasync_polaris_vi_selection.PolarisViClientSelectionStrategy.setup()`
builds it (numpy arrays indexed client_id - 1, priors 0.5 and 0.01).

The fake SERVER mirrors where `unexplored_clients` really lives: on the Polaris
AGGREGATION strategy, wrapped by `UploadGate` (`.inner`).  Stage 7's fake put
the set directly on the server -- the one place the real code never has it --
so the reset tests passed while every real recycle skipped the re-add
(STAGE7B_step1_siblings.md, section 4).  `test_stage7b_identity.py` repeats
the reset against the REAL classes; these fakes only keep this file free of
the Config singleton.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from stage7_population import (  # noqa: E402
    SIGMA,
    T_TIER,
    UNSEEN_GRADIENT_BOUND,
    UNSEEN_STALENESS,
    RecycleEvent,
    SlotAllocator,
    apply_recycle,
    partition_index,
    reset_slot_state,
    stable_digest,
    vehicle_epsilon,
    vehicle_sleep_s,
)

M = 8  # slots, for the fakes


def make_context(slots: int = M):
    """A stand-in for ServerContext carrying the polaris state layout."""
    return SimpleNamespace(
        state={
            "polaris": {
                "local_gradient_bounds": np.full(slots, UNSEEN_GRADIENT_BOUND, dtype=float),
                "local_stalenesses": np.full(slots, UNSEEN_STALENESS, dtype=float),
                "staleness_predictions": {},
            }
        }
    )


def make_server(slots: int = M, last_proxy=None):
    """Server -> UploadGate-like wrapper (.inner) -> Polaris aggregation strategy."""
    polaris_aggregation = SimpleNamespace(
        unexplored_clients=set(range(slots)), last_proxy=last_proxy
    )
    return SimpleNamespace(aggregation_strategy=SimpleNamespace(inner=polaris_aggregation))


def unexplored(server):
    return server.aggregation_strategy.inner.unexplored_clients


def seed_slot_with_history(context, server, slot_id: int) -> None:
    """Make the slot look like a client the server has watched for a while."""
    idx = slot_id - 1
    context.state["polaris"]["local_stalenesses"][idx] = 7.5
    context.state["polaris"]["local_gradient_bounds"][idx] = 3.25
    context.state["polaris"]["staleness_predictions"][slot_id] = 6.0
    unexplored(server).discard(idx)


def fill_then_recycle(allocator, outgoing="fEB.0", tier_out="xavier",
                      incoming="fWB.99", tier_in="thor"):
    """Occupy every slot, free the one `outgoing` holds, then let `incoming` in.

    Slots are handed out FIFO (see SlotAllocator.release), so a just-freed slot
    only comes back once the free list drains.  Filling first makes the freed
    slot the ONLY one available, which is both deterministic to assert on and
    the situation that actually matters: recycles happen when the allocator is
    near-full, which is exactly when M is doing any work.
    """
    slot, _ = allocator.assign(outgoing, tier_out)
    for i in range(1, allocator.slots):
        allocator.assign(f"filler.{i}", "orin")
    allocator.release(outgoing)
    new_slot, event = allocator.assign(incoming, tier_in)
    return slot, new_slot, event


# ---------------------------------------------------------------------------
# The two required assertions
# ---------------------------------------------------------------------------

class TestRecycleReset:
    def test_p_recur_zero_wipes_every_piece_of_slot_state(self):
        """After a recycle at p_recur = 0 the slot must look unseen."""
        context, server = make_context(), make_server()
        allocator = SlotAllocator(slots=M, p_recur=0.0)

        slot, _ = allocator.assign("fEB.0", "xavier")
        seed_slot_with_history(context, server, slot)
        for i in range(1, M):
            allocator.assign(f"filler.{i}", "orin")
        allocator.release("fEB.0")

        new_slot, event = allocator.assign("fWB.99", "thor")
        assert new_slot == slot, "the only free slot is the one just released"
        assert event.state_reset is True
        apply_recycle(context, event, server)

        idx = slot - 1
        polaris = context.state["polaris"]
        assert polaris["local_stalenesses"][idx] == UNSEEN_STALENESS
        assert polaris["local_gradient_bounds"][idx] == UNSEEN_GRADIENT_BOUND
        assert slot not in polaris["staleness_predictions"]
        assert idx in unexplored(server)

    def test_p_recur_zero_gives_the_current_unexplored_proxy(self):
        """Once an aggregation has set a proxy, a new identity gets THAT prior --
        the same G_i every other never-seen client carries -- not 0.5."""
        context, server = make_context(), make_server(last_proxy=9.25)
        allocator = SlotAllocator(slots=M, p_recur=0.0)
        slot, _ = allocator.assign("fEB.0", "xavier")
        seed_slot_with_history(context, server, slot)
        for i in range(1, M):
            allocator.assign(f"filler.{i}", "orin")
        allocator.release("fEB.0")
        _, event = allocator.assign("fWB.99", "thor")
        apply_recycle(context, event, server)
        assert context.state["polaris"]["local_gradient_bounds"][slot - 1] == 9.25
        assert slot - 1 in unexplored(server)

    def test_reset_refuses_a_polaris_server_without_the_set(self):
        """The Stage 7 failure mode must now be loud, not silent."""
        context = make_context()
        server = SimpleNamespace(unexplored_clients=set())  # the old, wrong place
        with pytest.raises(RuntimeError):
            reset_slot_state(context, 1, server)

    def test_p_recur_one_keeps_every_piece_of_slot_state(self):
        """The opposite: at p_recur = 1 the slot carries its history over."""
        context, server = make_context(), make_server()
        allocator = SlotAllocator(slots=M, p_recur=1.0)

        slot, _ = allocator.assign("fEB.0", "xavier")
        seed_slot_with_history(context, server, slot)
        for i in range(1, M):
            allocator.assign(f"filler.{i}", "orin")
        allocator.release("fEB.0")

        new_slot, event = allocator.assign("fWB.99", "thor")
        assert new_slot == slot
        assert event.state_reset is False
        apply_recycle(context, event, server)

        idx = slot - 1
        polaris = context.state["polaris"]
        assert polaris["local_stalenesses"][idx] == 7.5
        assert polaris["local_gradient_bounds"][idx] == 3.25
        assert polaris["staleness_predictions"][slot] == 6.0
        assert idx not in unexplored(server)


class TestTierFollowsTheVehicle:
    """Tier must follow the incoming vehicle, never the slot."""

    def test_tier_changes_with_the_new_occupant(self):
        allocator = SlotAllocator(slots=M, p_recur=0.0)
        slot, new_slot, event = fill_then_recycle(allocator)

        assert new_slot == slot
        assert allocator.tier_of_slot(slot) == "thor"
        assert event.tier_out == "xavier"
        assert event.tier_in == "thor"

    def test_tier_is_unchanged_by_p_recur(self):
        """Even when state is retained, the tier is the new vehicle's."""
        allocator = SlotAllocator(slots=M, p_recur=1.0)
        slot, new_slot, _event = fill_then_recycle(allocator)
        assert new_slot == slot
        assert allocator.tier_of_slot(slot) == "thor"


class TestAllocator:
    def test_assigns_distinct_slots(self):
        allocator = SlotAllocator(slots=3, p_recur=0.0)
        slots = {allocator.assign(f"v{i}", "orin")[0] for i in range(3)}
        assert slots == {1, 2, 3}

    def test_reports_exhaustion_rather_than_evicting(self):
        allocator = SlotAllocator(slots=2, p_recur=0.0)
        allocator.assign("a", "orin")
        allocator.assign("b", "orin")
        slot, event = allocator.assign("c", "orin")

        assert slot is None and event is None
        assert allocator.slot_exhausted_events == 1
        # The two holders keep their slots: nobody was evicted.
        assert allocator.occupancy == 2

    def test_reassigning_the_same_vehicle_is_idempotent(self):
        allocator = SlotAllocator(slots=2, p_recur=0.0)
        first, _ = allocator.assign("a", "thor")
        again, event = allocator.assign("a", "thor")
        assert again == first and event is None

    def test_first_use_of_a_virgin_slot_is_not_a_retain(self):
        """Nothing to carry over, so state_reset is True even at p_recur = 1."""
        allocator = SlotAllocator(slots=2, p_recur=1.0)
        _slot, event = allocator.assign("a", "orin")
        assert event.outgoing_veh_id is None
        assert event.state_reset is True

    def test_rejects_p_recur_outside_zero_one(self):
        with pytest.raises(ValueError, match="p_recur"):
            SlotAllocator(slots=2, p_recur=1.5)


class TestRecycleLogLine:
    def test_carries_every_field_the_prompt_requires(self):
        event = RecycleEvent(3, "fEB.1", "fWB.2", "orin", "thor", True)
        line = event.as_log_line()
        for field in ("slot_id=3", "outgoing_veh_id=fEB.1", "incoming_veh_id=fWB.2",
                      "tier_out=orin", "tier_in=thor", "state_reset=1"):
            assert field in line


# ---------------------------------------------------------------------------
# Determinism: the same vehicle must get the same sleep and the same shard in
# every process, or a spawned client disagrees with the server.
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_digest_is_stable_across_calls(self):
        assert stable_digest("eps", 4126, "fEB.7") == stable_digest("eps", 4126, "fEB.7")

    def test_digest_is_not_pythons_randomised_hash(self):
        """blake2b, so it cannot move with PYTHONHASHSEED between processes."""
        assert stable_digest("partition", "fEB.7") != hash("partition:fEB.7")

    def test_sleep_is_reproducible_per_vehicle(self):
        a = vehicle_sleep_s(4126, "fWB.512", "xavier")
        b = vehicle_sleep_s(4126, "fWB.512", "xavier")
        assert a == b

    def test_sleep_depends_on_the_vehicle_not_just_the_tier(self):
        a = vehicle_sleep_s(4126, "fWB.512", "xavier")
        b = vehicle_sleep_s(4126, "fWB.513", "xavier")
        assert a != b

    def test_sleep_is_the_tier_mean_times_exp_epsilon(self):
        eps = vehicle_epsilon(4126, "fEB.3")
        import math
        assert vehicle_sleep_s(4126, "fEB.3", "orin") == pytest.approx(
            T_TIER["orin"] * math.exp(eps)
        )

    def test_epsilon_matches_the_configured_sigma(self):
        """Statistical, over enough draws to be stable: sd ~= SIGMA."""
        import statistics as stats
        draws = [vehicle_epsilon(4126, f"fEB.{i}") for i in range(4000)]
        assert stats.mean(draws) == pytest.approx(0.0, abs=0.01)
        assert stats.stdev(draws) == pytest.approx(SIGMA, rel=0.05)

    def test_partition_follows_the_vehicle_and_is_in_range(self):
        for i in range(200):
            idx = partition_index(f"fWB.{i}", 25)
            assert 0 <= idx < 25
        assert partition_index("fWB.7", 25) == partition_index("fWB.7", 25)

    def test_partition_is_not_tied_to_the_slot(self):
        """Two vehicles that will share a slot must not share a partition."""
        a, b = "fEB.0", "fWB.99"
        assert partition_index(a, 25) != partition_index(b, 25)


class TestResetIsSafeOnMissingState:
    def test_no_crash_when_polaris_state_absent(self):
        context = SimpleNamespace(state={})
        reset_slot_state(context, 1, server=None)  # must not raise


class TestOpenCoverageTimeline:
    """The live driver must be the same computation as the gate's replay."""

    TRACE = (HERE.parent / "sumo_trace" / "e1_open_poisson_seed4126_range400"
             / "coverage_windows.csv")
    GATE_DIGEST = "d1002cd7c986a4f4a32f6b04383e99f8"

    @pytest.mark.skipif(not TRACE.exists(), reason="open trace not present")
    def test_reproduces_the_gates_occupancy_digest(self):
        from sumo_open_mobility import OpenCoverageTimeline

        timeline = OpenCoverageTimeline.from_csv(self.TRACE, slots=20, p_recur=0.0)
        timeline.advance_to(1e9)

        assert timeline.occupancy_digest() == self.GATE_DIGEST
        assert timeline.turned_away == 216
        assert len(timeline.allocator.recycles) == 10271

    @pytest.mark.skipif(not TRACE.exists(), reason="open trace not present")
    def test_digest_does_not_depend_on_p_recur(self):
        """Slot holding is residency-based, so the identity policy cannot move it."""
        from sumo_open_mobility import OpenCoverageTimeline

        digests = []
        for p in (0.0, 1.0):
            timeline = OpenCoverageTimeline.from_csv(self.TRACE, slots=20, p_recur=p)
            timeline.advance_to(1e9)
            digests.append(timeline.occupancy_digest())
        assert digests[0] == digests[1] == self.GATE_DIGEST

    @pytest.mark.skipif(not TRACE.exists(), reason="open trace not present")
    def test_advancing_in_steps_matches_advancing_at_once(self):
        """The clock can arrive in any increments; the stream must not care."""
        from sumo_open_mobility import OpenCoverageTimeline

        stepped = OpenCoverageTimeline.from_csv(self.TRACE, slots=20, p_recur=0.0)
        for t in range(0, 36_001, 500):
            stepped.advance_to(float(t))
        stepped.advance_to(1e9)
        assert stepped.occupancy_digest() == self.GATE_DIGEST


class TestDigestAtCommonTime:
    """The cross-run invariant, checked the way it actually has to be checked.

    Wave 1 exposed the trap: end-of-run digests differ between arms because
    each consumes a different amount of the trace in 4,000 rounds.  The
    invariant is about the STREAM, so it has to be compared at a common
    simulated time.
    """

    TRACE = (HERE.parent / "sumo_trace" / "e1_open_poisson_seed4126_range400"
             / "coverage_windows.csv")

    @pytest.mark.skipif(not TRACE.exists(), reason="open trace not present")
    def test_identity_policy_cannot_move_the_stream(self):
        from sumo_open_mobility import digest_at

        at_zero = digest_at(self.TRACE, 20, 6807.0, p_recur=0.0)
        at_one = digest_at(self.TRACE, 20, 6807.0, p_recur=1.0)
        assert at_zero == at_one

    @pytest.mark.skipif(not TRACE.exists(), reason="open trace not present")
    def test_matches_what_wave_1_actually_logged(self):
        """observed_open stopped at t = 6807 s; this is the digest it emitted."""
        from sumo_open_mobility import digest_at

        digest, recycles, turned_away = digest_at(self.TRACE, 20, 6807.0)
        assert digest == "8906824024866ff18ae02283e177bceb"
        assert (recycles, turned_away) == (1891, 39)

    @pytest.mark.skipif(not TRACE.exists(), reason="open trace not present")
    def test_a_later_time_is_a_strict_extension(self):
        """Digests at different times must differ — otherwise the check is vacuous."""
        from sumo_open_mobility import digest_at

        assert digest_at(self.TRACE, 20, 6807.0)[0] != digest_at(self.TRACE, 20, 7262.0)[0]


class TestSelectableSlotsGuard:
    """Regression for the clock deadlock that froze `cpu_open` for 14 hours.

    A slot occupied by a vehicle PLATO is withholding (still training, or
    reported-but-not-yet-merged) is NOT dispatchable.  A guard that asks "is
    any slot occupied?" therefore refuses to idle while the selector is handed
    an empty pool, nothing is dispatched, no report arrives, and the simulated
    clock stops forever.  `sumo_server._advance_clock_to_coverage` documents
    the same bug in the closed-loop cell (cpu arm, round 719, 2026-09-06).
    """

    @staticmethod
    def _selectable(occupied, training, pending_merge):
        """The predicate as `OpenPopulationMixin._selectable_slots` computes it."""
        return set(occupied) - set(training) - set(pending_merge)

    def test_occupied_but_all_withheld_is_not_selectable(self):
        """The exact state cpu_open froze in: slots held, none dispatchable."""
        assert self._selectable(occupied=[6, 18], training=[], pending_merge=[6, 18]) == set()

    def test_occupied_and_free_is_selectable(self):
        assert self._selectable(occupied=[6, 18], training=[], pending_merge=[6]) == {18}

    def test_training_slots_are_not_selectable(self):
        assert self._selectable(occupied=[1, 2, 3], training=[1, 2], pending_merge=[]) == {3}

    def test_empty_occupancy_is_not_selectable(self):
        assert self._selectable(occupied=[], training=[], pending_merge=[]) == set()
