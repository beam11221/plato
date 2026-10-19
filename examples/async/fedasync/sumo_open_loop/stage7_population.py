"""
Stage 7 — open population: the slot allocator, the per-vehicle sleep model, and
the identity policy (`p_recur`).  See prompts/STAGE7_open_population.md.

The problem this solves
-----------------------
PLATO needs a bounded set of integer client ids.  The open trace has 10,523
STRING ids (`fEB.0`, `fWB.12`), each of which crosses the RSU exactly once and
never returns.  So a vehicle cannot own a client id for the length of a run the
way it does in the closed-loop cell; ids have to be recycled.

That recycling is the experiment.  `p_recur` is the probability that a freed
slot keeps the server-side state its previous occupant built up:

    p_recur = 0   open population   — every recycle wipes the slot clean, so a
                                      reactive estimator never accumulates the
                                      ~30-70 observations it needs (Stage 3's k*)
    p_recur = 1   closed population — state persists across recycles, which
                                      reproduces the closed-loop cell on the
                                      SAME arrivals and the SAME windows

Why the hashes are blake2b and not `hash()`
-------------------------------------------
Python randomises `hash()` of a str per process via PYTHONHASHSEED.  Both the
sleep draw and the partition index have to agree across the server process and
every spawned client process, so both go through blake2b — the same reasoning,
and the same fix, as `gpu_tiers_dataset/client_seeded_trainer.py`'s
`derive_seed`.  Getting this wrong would give one vehicle two different sleeps
and two different data shards depending on which process asked.

What is deliberately NOT taken from the trace's config.yaml
-----------------------------------------------------------
`config.yaml` carries `t_bar_train_s: {xavier: 30, orin: 18, thor: 10}` with
`t_bar_train_provisional: true`.  Its tier ratio is 3 : 1.8 : 1; every result
from Stage 3 onward is on 15 : 1.923077 : 1.  Using it would make tier
separation 3x instead of 15x and nothing would compare to any earlier stage.
`T_TIER` below is the Stage 3-6 model and is the one that is used.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass, field

# Stage 3-6's training-time model.  NOT config.yaml's t_bar_train_s.
T_TIER: dict[str, float] = {"xavier": 15.0, "orin": 1.923077, "thor": 1.0}

# Per-vehicle lognormal jitter, identical to the closed-loop fleet builder
# (gpu_tiers_dataset/build_fleet.py): sleep = T_tier * exp(eps), eps ~ N(0, s).
SIGMA = 0.149166

# What a slot looks like to the server before anyone has been seen on it.
# These mirror the configs' polaris_initial_* keys and the defaults in
# fedasync_polaris_vi_selection.setup().
UNSEEN_STALENESS = 0.01
UNSEEN_GRADIENT_BOUND = 0.5


# ---------------------------------------------------------------------------
# Deterministic, process-stable derivations from a vehicle's string id
# ---------------------------------------------------------------------------

def stable_digest(*parts: object) -> int:
    """A process-stable 64-bit integer from the parts, joined with ':'."""
    key = ":".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def vehicle_epsilon(base_seed: int, veh_id: str) -> float:
    """eps ~ N(0, SIGMA^2) for one vehicle, fixed for the whole programme.

    Drawn from (base_seed, veh_id) so the same vehicle gets the same sleep in
    every run and every arm — the open-population equivalent of the closed
    fleet's `sleep_table.csv`, which cannot exist for 10,523 vehicles.
    """
    rng = random.Random(stable_digest("eps", base_seed, veh_id))
    return rng.gauss(0.0, SIGMA)


def vehicle_sleep_s(base_seed: int, veh_id: str, tier: str) -> float:
    """The simulated training time this vehicle would take, in seconds."""
    if tier not in T_TIER:
        raise KeyError(f"[Stage7] unknown tier {tier!r} for vehicle {veh_id!r}")
    return T_TIER[tier] * math.exp(vehicle_epsilon(base_seed, veh_id))


def partition_index(veh_id: str, slots: int) -> int:
    """Which of the `slots` data partitions this VEHICLE trains on.

    Keyed on the vehicle, never on the slot: a recycled slot must not hand a
    brand-new vehicle the previous occupant's data, or G_i would carry
    information across an identity boundary the stage is trying to seal.
    """
    if slots <= 0:
        raise ValueError(f"[Stage7] slots must be positive, got {slots}")
    return stable_digest("partition", veh_id) % slots


# ---------------------------------------------------------------------------
# The allocator
# ---------------------------------------------------------------------------

@dataclass
class RecycleEvent:
    """One slot changing hands.  Logged per occurrence so a run is auditable."""

    slot_id: int
    outgoing_veh_id: str | None
    incoming_veh_id: str
    tier_out: str | None
    tier_in: str
    state_reset: bool

    def as_log_line(self) -> str:
        return (
            f"[Stage7] recycle slot_id={self.slot_id} "
            f"outgoing_veh_id={self.outgoing_veh_id or '-'} "
            f"incoming_veh_id={self.incoming_veh_id} "
            f"tier_out={self.tier_out or '-'} tier_in={self.tier_in} "
            f"state_reset={int(self.state_reset)}"
        )


class SlotAllocator:
    """Maps the vehicles currently inside coverage onto `slots` PLATO ids.

    Slot ids are 1-based, matching PLATO's client ids.  A slot is either free
    or held by exactly one vehicle.  `assign` returns None when every slot is
    taken — the caller records a `slot_exhausted` event and the arriving
    vehicle never becomes a client, which is the failure mode `M` is sized to
    avoid.
    """

    def __init__(self, slots: int, p_recur: float, base_seed: int = 4126) -> None:
        if not 0.0 <= p_recur <= 1.0:
            raise ValueError(f"[Stage7] p_recur must be in [0, 1], got {p_recur}")
        self.slots = slots
        self.p_recur = p_recur
        self.base_seed = base_seed
        # Slot -> the vehicle holding it now, and the one that held it before.
        self.occupant: dict[int, str] = {}
        self.occupant_tier: dict[int, str] = {}
        self.previous_occupant: dict[int, str] = {}
        self.previous_tier: dict[int, str] = {}
        self.slot_of: dict[str, int] = {}
        self._free: list[int] = list(range(1, slots + 1))
        # Recycle decisions are seeded so a p_recur strictly between 0 and 1
        # would still be reproducible; at 0 and 1 the RNG is never consulted.
        self._rng = random.Random(stable_digest("recycle", base_seed, slots, p_recur))

        self.recycles: list[RecycleEvent] = []
        # Stage 11: optional OBSERVERS, called synchronously at the moment a
        # slot is assigned / released (None = Stages 7-10, unchanged).  They
        # never influence which slot a vehicle gets, so the assign/release
        # stream -- and the occupancy digest -- is the same with or without
        # them.  on_assign may return True/False to mark whether the new
        # occupant carries server-side state over (the event's state_reset).
        self.on_assign = None   # (slot_id, veh_id, tier) -> bool | None
        self.on_release = None  # (slot_id, veh_id) -> None
        self.slot_exhausted_events: int = 0
        self.assignments: int = 0
        self.peak_occupancy: int = 0

    # -- queries --------------------------------------------------------

    @property
    def occupancy(self) -> int:
        return len(self.occupant)

    def tier_of_slot(self, slot_id: int) -> str | None:
        """The tier of whoever holds the slot NOW.  Tier follows the vehicle."""
        return self.occupant_tier.get(slot_id)

    # -- mutation -------------------------------------------------------

    def assign(self, veh_id: str, tier: str) -> tuple[int | None, RecycleEvent | None]:
        """Give `veh_id` a slot.  Returns (slot_id, recycle_event_or_None).

        slot_id is None when the allocator is full; the caller must treat that
        as a `slot_exhausted` event rather than evicting anyone, because
        evicting a training vehicle would silently truncate the population in
        a second, harder-to-see way.
        """
        if veh_id in self.slot_of:
            return self.slot_of[veh_id], None
        if not self._free:
            self.slot_exhausted_events += 1
            return None, None

        slot_id = self._free.pop(0)
        outgoing = self.previous_occupant.get(slot_id)
        tier_out = self.previous_tier.get(slot_id)

        # p_recur is the probability the slot RETAINS the old state.
        if outgoing is None:
            keep_state = False  # first use of a virgin slot: nothing to retain
        elif self.p_recur >= 1.0:
            keep_state = True
        elif self.p_recur <= 0.0:
            keep_state = False
        else:
            keep_state = self._rng.random() < self.p_recur

        self.occupant[slot_id] = veh_id
        self.occupant_tier[slot_id] = tier
        self.slot_of[veh_id] = slot_id
        self.assignments += 1
        self.peak_occupancy = max(self.peak_occupancy, self.occupancy)

        if self.on_assign is not None:
            carried = self.on_assign(slot_id, veh_id, tier)
            if carried is not None:
                keep_state = bool(carried)

        event = RecycleEvent(
            slot_id=slot_id,
            outgoing_veh_id=outgoing,
            incoming_veh_id=veh_id,
            tier_out=tier_out,
            tier_in=tier,
            state_reset=not keep_state,
        )
        self.recycles.append(event)
        return slot_id, event

    def release(self, veh_id: str) -> int | None:
        """Free the slot `veh_id` holds; it becomes reusable by a later arrival."""
        slot_id = self.slot_of.pop(veh_id, None)
        if slot_id is None:
            return None
        if self.on_release is not None:
            self.on_release(slot_id, veh_id)  # before the slot is cleared
        self.previous_occupant[slot_id] = veh_id
        self.previous_tier[slot_id] = self.occupant_tier.get(slot_id)
        self.occupant.pop(slot_id, None)
        self.occupant_tier.pop(slot_id, None)
        # Append rather than insert: slots are reused round-robin, which keeps
        # a just-freed slot from being handed straight back to the next
        # arrival and makes recycles spread over all M slots.
        self._free.append(slot_id)
        return slot_id


# ---------------------------------------------------------------------------
# The reset — the single thing most likely to be wrong (see the stage prompt)
# ---------------------------------------------------------------------------

def unexplored_owner(server):
    """The object that actually holds `unexplored_clients`, or None.

    It is the Polaris AGGREGATION strategy (fedasync_polaris_vi_server.py:187),
    not the server, and the open server wraps that strategy in `UploadGate`.
    Stage 7 looked for it on the server, found nothing, and skipped the re-add
    silently: every recycled slot kept G_i = 0.5 instead of the unexplored
    proxy (STAGE7B_step1_siblings.md, section 4).  So walk `.inner` wrappers
    from the aggregation strategy down, and fail loudly in `reset_slot_state`
    if a Polaris server has none.
    """
    if server is None:
        return None
    candidate = getattr(server, "aggregation_strategy", None)
    for _ in range(10):
        if candidate is None:
            return None
        if getattr(candidate, "unexplored_clients", None) is not None:
            return candidate
        candidate = getattr(candidate, "inner", None)
    return None


def reset_slot_state(context, slot_id: int, server=None) -> None:
    """Wipe every piece of server-side state keyed to `slot_id`.

    The places a slot's history lives, found by reading
    fedasync_polaris_vi_selection.py and fedasync_polaris_vi_server.py:

      local_stalenesses[slot-1]      the reactive arm's tau_hat  -> unseen default
      local_gradient_bounds[slot-1]  G_i                          -> the CURRENT
                                     unexplored proxy (alpha x mean), i.e. the
                                     prior every other never-seen client carries
                                     now; UNSEEN_GRADIENT_BOUND only before the
                                     first aggregation has set any proxy
      staleness_predictions[slot]    a prediction still in flight -> dropped, or it
                                     would be scored against the NEW occupant's report
      unexplored_clients             "never yet reported" marker  -> slot re-added,
                                     so the next aggregation keeps it on the proxy

    That last set is what this codebase uses in place of a times_selected
    counter (fedasync_polaris_vi_server.py:288-293).  It lives on the
    aggregation strategy; see `unexplored_owner`.
    """
    polaris_state = context.state.setdefault("polaris", {})
    idx = slot_id - 1

    stalenesses = polaris_state.get("local_stalenesses")
    if stalenesses is not None and idx < len(stalenesses):
        stalenesses[idx] = UNSEEN_STALENESS

    owner = unexplored_owner(server)
    bounds = polaris_state.get("local_gradient_bounds")
    if bounds is not None and idx < len(bounds):
        proxy = getattr(owner, "last_proxy", None) if owner is not None else None
        bounds[idx] = UNSEEN_GRADIENT_BOUND if proxy is None else proxy

    predictions = polaris_state.get("staleness_predictions")
    if isinstance(predictions, dict):
        predictions.pop(slot_id, None)

    if owner is not None:
        owner.unexplored_clients.add(idx)
    elif bounds is not None and server is not None:
        # A Polaris server (it has G_i) whose unexplored set cannot be found is
        # exactly the Stage 7 defect.  Refuse to continue rather than run it again.
        raise RuntimeError(
            "[Stage7b] reset_slot_state: server has G_i state but no "
            "unexplored_clients was found on its aggregation strategy."
        )


def apply_recycle(context, event: RecycleEvent, server=None) -> None:
    """Log the recycle and, when it is not a retain, wipe the slot."""
    logging.info("%s", event.as_log_line())
    if event.state_reset:
        reset_slot_state(context, event.slot_id, server)
