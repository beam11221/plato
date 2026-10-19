"""
Stage 7b — the one notion of WHO is standing in a slot.

Stage 7 had three separate answers to "which vehicle is this?", and they
disagreed silently: the proactive predictors read the slot's row in the
closed-loop fleet file (H3), the sampler read the slot id, the per-round seed
read the slot id, while the sleep and the coverage verdict read the trace
vehicle.  See gpu_tiers_dataset/stage7_evidence/STAGE7B_step1_siblings.md.

This module replaces all of those with a single lookup.  At each moment a slot
has an **occupant identity** `(identity_id, tier, sleep_s, partition)`, and
everything that used to key on `client_id` reads it instead:

    p_recur = 0   identity is the TRACE VEHICLE.  Tier from the trace, sleep
                  from `vehicle_sleep_s`, partition `hash(veh_id) mod M`.
                  Every recycle is a new identity.

    p_recur = 1   identity is BOUND TO THE SLOT.  Slot k is one persistent
                  vehicle of the Stage 3-6 fleet: its tier from
                  `clients_n20r400.csv`, its sleep from
                  `client_sleep_times_n20r400.csv`, its partition k-1.  The
                  trace supplies only WHEN it is in coverage.  SUMO motion is
                  tier-agnostic (one vType), so giving a window a slot's tier
                  does not violate the physics.

What stays with the TRACE VEHICLE in both regimes: the coverage window, i.e.
whether an upload arrives.  That is mobility, not identity.

Only p_recur in {0, 1} is supported.  An intermediate value would need a rule
for which identity a recycled slot shows, which no stage has specified.
"""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from stage7_population import partition_index, stable_digest, vehicle_sleep_s

# Same modulus as gpu_tiers_dataset/client_seeded_trainer.py.
_SEED_MODULUS = 2**31 - 1


@dataclass(frozen=True)
class OccupantIdentity:
    """Who a slot's occupant IS, for everything except its coverage window."""

    identity_id: str
    tier: str
    sleep_s: float
    partition: int  # 0-based; the sampler is built with id partition + 1


def identity_seed(base_seed: int, identity_id: str, current_round: int | None) -> int:
    """Per-(identity, round) training seed.

    The key has exactly the format of `client_seeded_trainer.derive_seed`
    (`f"{base}:{id}:{round}"`), so at p_recur = 1, where identity_id is the
    slot number as a string, this returns the SAME seed the closed-loop cell
    used for that client.  At p_recur = 0 the identity is the vehicle.
    """
    if current_round is None:
        current_round = 0
    key = f"{int(base_seed)}:{identity_id}:{int(current_round)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


def load_fleet(clients_csv: str | Path, sleep_csv: str | Path) -> dict[int, tuple[str, float]]:
    """slot -> (tier, sleep_s) for the Stage 3-6 fleet."""
    tiers: dict[int, str] = {}
    with Path(clients_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            tiers[int(row["client_id"])] = row["tier_id"].strip()
    sleeps: dict[int, float] = {}
    with Path(sleep_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            sleeps[int(row["client_id"])] = float(row["sleep_time"])
    if set(tiers) != set(sleeps):
        raise ValueError(
            f"[Stage7b] fleet files disagree on client ids: "
            f"{sorted(set(tiers) ^ set(sleeps))}"
        )
    return {cid: (tiers[cid], sleeps[cid]) for cid in sorted(tiers)}


class ReturningPopulation:
    """Stage 11: partial recurrence (`p_return`).

    On every slot ALLOCATION (not on every query), draw u from the identity
    RNG.  If u < p_return and some commuter is not currently in coverage, the
    occupant is a commuter chosen uniformly among those; otherwise it is the
    trace vehicle of the window (a transient, exactly as at p_recur = 0).

    Commuters are the 20 fixed Stage 3-10 fleet identities: commuter k has
    fleet vehicle k's tier and sleep and partition k-1, identity id "C<k>".

    Both RNG streams are seeded from the PHYSICS seed only -- the identity
    sequence is part of the population, like the trace, so it is identical
    across arms and run seeds.  u is drawn on every allocation from its own
    stream, so the u sequence is identical at every p_return (common random
    numbers across the curve); the commuter pick has a separate stream.
    """

    def __init__(self, p_return: float, fleet: dict[int, tuple[str, float]], physics_seed: int) -> None:
        if not 0.0 <= p_return <= 1.0:
            raise ValueError(f"[Stage11] p_return must be in [0, 1], got {p_return}")
        self.p_return = p_return
        self.fleet = fleet
        self.commuters = {f"C{k}": OccupantIdentity(f"C{k}", tier, sleep, k - 1)
                          for k, (tier, sleep) in sorted(fleet.items())}
        self._u = random.Random(stable_digest("identity-u", physics_seed))
        self._pick = random.Random(stable_digest("identity-pick", physics_seed))
        self.slot_identity: dict[int, OccupantIdentity] = {}
        self.slot_kind: dict[int, str] = {}
        self.in_coverage: set[str] = set()
        self.allocations = self.commuter_allocations = self.exhaustions = 0
        self.kind_tiers = {"commuter": {}, "transient": {}}
        self._digest = hashlib.blake2b(digest_size=16)

    def assign(self, slot: int, veh_id: str, tier: str, base_seed: int, slots: int):
        """-> (identity, kind, exhausted)."""
        self.allocations += 1
        u = self._u.random()
        exhausted = False
        identity, kind = None, "transient"
        if u < self.p_return:
            free = sorted((c for c in self.commuters if c not in self.in_coverage),
                          key=lambda c: int(c[1:]))
            if free:
                cid = free[self._pick.randrange(len(free))]
                identity, kind = self.commuters[cid], "commuter"
                self.in_coverage.add(cid)
                self.commuter_allocations += 1
            else:
                exhausted = True
                self.exhaustions += 1
        if identity is None:
            identity = OccupantIdentity(veh_id, tier, vehicle_sleep_s(base_seed, veh_id, tier),
                                        partition_index(veh_id, slots))
        self.slot_identity[slot] = identity
        self.slot_kind[slot] = kind
        counts = self.kind_tiers[kind]
        counts[identity.tier] = counts.get(identity.tier, 0) + 1
        self._digest.update(f"{slot}|{veh_id}|{identity.identity_id}|{kind}\n".encode())
        return identity, kind, exhausted

    def release(self, slot: int) -> tuple[OccupantIdentity | None, str | None]:
        identity = self.slot_identity.pop(slot, None)
        kind = self.slot_kind.pop(slot, None)
        if kind == "commuter":
            self.in_coverage.discard(identity.identity_id)
        return identity, kind

    def digest(self) -> str:
        return self._digest.hexdigest()


class IdentityResolver:
    """Answers "who is the occupant of slot k right now?" for both regimes."""

    def __init__(
        self,
        timeline,
        base_seed: int,
        p_recur: float,
        slots: int,
        fleet: dict[int, tuple[str, float]] | None = None,
        p_return: float | None = None,
    ) -> None:
        # Stage 11: partial recurrence is a separate identity policy on top of
        # p_recur = 0 (p_recur keeps its meaning and code path).
        self.returning = None
        if p_return is not None:
            if p_recur != 0.0:
                raise ValueError("[Stage11] p_return needs p_recur = 0 (it replaces the transient-only policy).")
            if fleet is None:
                raise ValueError("[Stage11] p_return needs the Stage 3-10 fleet files (the commuter pool).")
            self.returning = ReturningPopulation(p_return, fleet, base_seed)
        if p_recur not in (0.0, 1.0):
            raise ValueError(
                f"[Stage7b] p_recur must be 0 or 1, got {p_recur}: no rule exists "
                "for which identity a partly-retained slot shows."
            )
        self.timeline = timeline
        self.base_seed = base_seed
        self.p_recur = p_recur
        self.slots = slots
        self.slot_bound = p_recur == 1.0

        if self.slot_bound:
            if fleet is None:
                raise ValueError("[Stage7b] p_recur = 1 needs the Stage 3-6 fleet files.")
            missing = [k for k in range(1, slots + 1) if k not in fleet]
            if missing:
                raise ValueError(f"[Stage7b] fleet has no entry for slot(s) {missing}.")
        self.fleet = fleet or {}

    def attach(self, allocator) -> None:
        """Stage 11: decide each allocation's identity at the moment it happens."""
        if self.returning is None:
            return
        # Chained by the server (sumo_open_server.install_returning_population),
        # which also moves the state; this default only keeps identities right.
        allocator.on_assign = lambda slot, veh, tier: (
            self.returning.assign(slot, veh, tier, self.base_seed, self.slots) and None)
        allocator.on_release = lambda slot, veh: self.returning.release(slot) and None

    def of_slot(self, slot: int) -> OccupantIdentity | None:
        """The identity occupying `slot` now, or None if the slot is empty."""
        veh_id = self.timeline.vehicle_in_slot(slot)
        if veh_id is None:
            return None
        if self.returning is not None:
            return self.returning.slot_identity.get(slot)
        if self.slot_bound:
            tier, sleep_s = self.fleet[slot]
            return OccupantIdentity(str(slot), tier, sleep_s, slot - 1)
        tier = self.timeline.tier_of_vehicle[veh_id]
        return OccupantIdentity(
            veh_id,
            tier,
            vehicle_sleep_s(self.base_seed, veh_id, tier),
            partition_index(veh_id, self.slots),
        )


def partition_fingerprint(sampler, datasource) -> str:
    """A short digest of a Dirichlet sampler's label proportions.

    The partition a sampler draws is a function of its per-class weights, and
    those are a function only of the id it was built with.  Logging this per
    dispatch lets the pre-flight prove from the LOG which partition trained:
    fingerprints must group by the occupant's partition, never by slot.
    Returns "na" for samplers without `sample_weights`.
    """
    weights = getattr(sampler, "sample_weights", None)
    if weights is None:
        return "na"
    targets = datasource.targets()
    per_class: dict[int, float] = {}
    for weight, target in zip(weights, targets):
        per_class.setdefault(int(target), float(weight))
    key = ",".join(f"{c}:{per_class[c]:.10f}" for c in sorted(per_class))
    return hashlib.blake2b(key.encode(), digest_size=6).hexdigest()
