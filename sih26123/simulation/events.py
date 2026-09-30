"""In-memory event log.

The foundation uses a simple append-only log. Event types are a closed
``Enum`` so that downstream code (UI, dashboards, replay) can rely on a
stable set of names. No database yet; logs are kept in memory only.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(Enum):
    """The full set of events the simulator can emit."""

    SIMULATION_STARTED = "simulation_started"
    SIMULATION_ENDED = "simulation_ended"
    ROBOT_SPAWNED = "robot_spawned"
    ROBOT_STARTED = "robot_started"
    ROBOT_MOVED = "robot_moved"
    ROBOT_WAITING = "robot_waiting"
    ROBOT_IDLE = "robot_idle"
    ROBOT_ARRIVED = "robot_arrived"
    ROBOT_FAILED = "robot_failed"
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_PICKED_UP = "task_picked_up"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    OBSTACLE_DETECTED = "obstacle_detected"
    OBSTACLE_ADDED = "obstacle_added"
    OBSTACLE_REMOVED = "obstacle_removed"
    COLLISION_DETECTED = "collision_detected"

    # Phase 2A additions (coordination / conflict prediction).
    PEER_STATE_UPDATED = "peer_state_updated"
    CONFLICT_PREDICTED = "conflict_predicted"
    MESSAGE_BROADCAST = "message_broadcast"

    # Phase 2B additions (decentralized conflict resolution).
    CONFLICT_NEGOTIATION_STARTED = "conflict_negotiation_started"
    CONFLICT_NEGOTIATION_RESOLVED = "conflict_negotiation_resolved"
    ROBOT_YIELDED = "robot_yielded"
    ROBOT_REROUTED = "robot_rerouted"
    RESERVATION_CONFLICT = "reservation_conflict"

    # Phase 2C additions (failure and resilience).
    ROBOT_HEARTBEAT = "robot_heartbeat"
    ROBOT_FAILED_DETECTED = "robot_failed_detected"
    ROBOT_OFFLINE = "robot_offline"
    ROBOT_SAFE_HALT = "robot_safe_halt"
    PEER_STALE = "peer_stale"
    PEER_FAILED = "peer_failed"
    PEER_ISOLATED = "peer_isolated"
    RESERVATIONS_CLEARED = "reservations_cleared"
    DEADLOCK_DETECTED = "deadlock_detected"
    DEADLOCK_RESOLVED = "deadlock_resolved"
    TASK_RETURNED_TO_POOL = "task_returned_to_pool"
    TASK_REASSIGNED = "task_reassigned"
    TASK_POOL = "task_pool"


class Event:
    """A single recorded event.

    Constructor accepts arbitrary keyword arguments that are stored in
    the ``data`` dict. This lets callers write
    ``Event(tick, EventType.ROBOT_MOVED, robot_id="A")`` without
    manually building a dict.
    """

    __slots__ = ("tick", "event_type", "data")

    def __init__(self, tick: int, event_type: EventType, **data: Any) -> None:
        self.tick = int(tick)
        self.event_type = event_type
        self.data = dict(data)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Event(tick={self.tick}, type={self.event_type.value}, data={self.data})"


class EventLog:
    """Append-only in-memory event log.

    The log is owned by the simulator and queried by tests or the demo.
    """

    def __init__(self) -> None:
        self._events: List[Event] = []

    def record(self, tick: int, event_type: EventType, **data: Any) -> Event:
        event = Event(tick=tick, event_type=event_type, data=dict(data))
        self._events.append(event)
        return event

    def all(self) -> List[Event]:
        """Return a shallow copy of all recorded events."""
        return list(self._events)

    def of_type(self, event_type: EventType) -> List[Event]:
        """Return all events of a given type."""
        return [e for e in self._events if e.event_type is event_type]

    def last_of_type(self, event_type: EventType) -> Optional[Event]:
        for e in reversed(self._events):
            if e.event_type is event_type:
                return e
        return None

    def count(self, event_type: EventType) -> int:
        return sum(1 for e in self._events if e.event_type is event_type)

    def clear(self) -> None:
        self._events.clear()

    def extend(self, events: List[Event]) -> None:
        """Append an iterable of pre-built :class:`Event` objects."""
        for e in events:
            self._events.append(e)