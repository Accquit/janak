"""Negotiation protocol types (Phase 2B).

This module defines:

* :class:`NegotiationAction`       what a robot decided to do about a conflict
* :class:`ConflictProposal`        a typed message from a robot declaring its
                                   own action and priority for a conflict
* :class:`ConflictResponse`        a typed acknowledgment with the responder's
                                   own action and priority
* :class:`NegotiationRecord`       per-robot local state for an in-flight
                                   negotiation
* :func:`make_conflict_id`         deterministic conflict identifier used to
                                   correlate proposals and responses

The protocol is purely algorithmic — no natural language, no shared
state. A robot's decision is computed locally from the priorities it
observes; the message exchange is informational and used to confirm
that the two sides agree (or to to after a TTL).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from .conflicts import ConflictType


class NegotiationAction(Enum):
    """The action a robot has decided to take on a conflict."""

    PROCEED = "proceed"   # continue along the current plan
    YIELD = "yield"       # back off (wait or reroute)
    REROUTE = "reroute"   # explicitly reroute around the conflict
    WAIT = "wait"         # explicitly stand still for ``wait_ticks`` ticks


# A negotiation message lives for at most this many ticks before the
# robot proceeds with its unilateral decision.
DEFAULT_NEGOTIATION_TTL = 5


@dataclass(frozen=True)
class ConflictProposal:
    """A typed proposal sent by a robot about a single conflict."""

    conflict_id: str
    sender_id: str
    peer_id: str
    conflict_type: ConflictType
    timestep: int
    cell: Optional[Tuple[int, int]] = None
    edge: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    self_priority: float = 0.0
    self_action: NegotiationAction = NegotiationAction.PROCEED
    self_path_length: int = 0


@dataclass(frozen=True)
class ConflictResponse:
    """A typed acknowledgment sent by the peer of a :class:`ConflictProposal`."""

    conflict_id: str
    sender_id: str
    peer_id: str
    self_priority: float = 0.0
    self_action: NegotiationAction = NegotiationAction.PROCEED
    acknowledged: bool = True


@dataclass
class NegotiationRecord:
    """Per-robot local state for an in-flight negotiation.

    The robot is the ``owner_id``. The peer is identified by
    ``peer_id``. Resolution happens when both sides agree OR the TTL
    expires.
    """

    conflict_id: str
    peer_id: str
    conflict_type: ConflictType
    timestep: int
    cell: Optional[Tuple[int, int]] = None
    edge: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    initiated_tick: int = 0
    ttl: int = DEFAULT_NEGOTIATION_TTL
    self_priority: float = 0.0
    self_action: NegotiationAction = NegotiationAction.PROCEED
    peer_priority: Optional[float] = None
    peer_action: Optional[NegotiationAction] = None
    resolved: bool = False
    resolution: Optional[NegotiationAction] = None  # final action taken
    resolution_reason: str = ""


def make_conflict_id(
    conflict_type: ConflictType,
    robot_a: str,
    robot_b: str,
    timestep: int,
    cell: Optional[Tuple[int, int]] = None,
    edge: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> str:
    """Return a deterministic conflict identifier.

    The two robot ids are sorted so the same conflict appears under
    the same id regardless of which side observed it first.
    """
    a, b = sorted((robot_a, robot_b))
    if conflict_type is ConflictType.VERTEX:
        assert cell is not None, "vertex conflict requires a cell"
        return f"V:{a}:{b}:{timestep}:{cell[0]},{cell[1]}"
    assert edge is not None, "edge conflict requires an edge"
    (xa, ya), (xb, yb) = edge
    return f"E:{a}:{b}:{timestep}:{xa},{ya}->{xb},{yb}"


def record_for(
    conflict_id: str,
    negotiation_records: dict,
) -> Optional[NegotiationRecord]:
    """Convenience accessor used by tests and the agent."""
    return negotiation_records.get(conflict_id)