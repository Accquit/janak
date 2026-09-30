"""Decentralized coordination layer (Phase 2A).

This package adds the *first half* of decentralized fleet coordination:

- :mod:`communication`  P2P-style message bus and per-robot peer state.
- :mod:`reservations`   Space-time reservation table (per robot).
- :mod:`conflicts`      Local conflict prediction (vertex + edge).
- :mod:`agent`          :class:`RobotCoordination` wires the above to a
  robot.

Important architectural notes
-----------------------------
* The simulator remains a world clock; it does **not** plan, assign
  or negotiate. All decisions stay local to each robot.
* Each robot owns its own peer-state view, its own reservation
  table and its own conflict detector. There is **no** central
  authority holding those data structures.
* Conflict prediction only **logs** events. It does **not** wait,
  reroute, or alter robot behaviour. Resolution belongs to the next
  layer (Phase 2B).
"""

from .communication import (
    InProcessMessageBus,
    Message,
    MessageBus,
    MessageType,
    PeerStateRepository,
    RobotState,
)
from .reservations import Reservation, ReservationTable
from .conflicts import ConflictDetector, ConflictType, PredictedConflict
from .priority import PriorityInputs, compute_priority, is_higher_priority
from .negotiation import (
    ConflictProposal,
    ConflictResponse,
    NegotiationAction,
    NegotiationRecord,
    make_conflict_id,
)
from .spacetime import space_time_astar, spatial_path
from .resolver import ResolutionChoice, choose_yield_action
from .resilience import (
    DEADLOCK_WINDOW,
    HEARTBEAT_INTERVAL,
    PEER_FAILURE_TIMEOUT,
    SELF_ISOLATION_TIMEOUT,
    STALE_THRESHOLD,
)
from .agent import RobotCoordination
from .lifecycle import (
    PeerLiveness,
    PeerLivenessTracker,
    RobotLifecycle,
    RobotLifecycleState,
)

__all__ = [
    "InProcessMessageBus",
    "Message",
    "MessageBus",
    "MessageType",
    "PeerStateRepository",
    "RobotState",
    "Reservation",
    "ReservationTable",
    "ConflictDetector",
    "ConflictType",
    "PredictedConflict",
    "PriorityInputs",
    "compute_priority",
    "is_higher_priority",
    "ConflictProposal",
    "ConflictResponse",
    "NegotiationAction",
    "NegotiationRecord",
    "make_conflict_id",
    "space_time_astar",
    "spatial_path",
    "ResolutionChoice",
    "choose_yield_action",
    "DEADLOCK_WINDOW",
    "HEARTBEAT_INTERVAL",
    "PEER_FAILURE_TIMEOUT",
    "SELF_ISOLATION_TIMEOUT",
    "STALE_THRESHOLD",
    "RobotCoordination",
    "PeerLiveness",
    "PeerLivenessTracker",
    "RobotLifecycle",
    "RobotLifecycleState",
]