"""P2P communication abstraction (Phase 2A).

This module provides:

* :class:`MessageType`     Closed enum of supported message kinds.
* :class:`Message`         Typed message envelope.
* :class:`RobotState`      Structured robot broadcast payload.
* :class:`MessageBus`      Abstract interface (transport-agnostic).
* :class:`InProcessMessageBus` Deterministic in-process bus used in
  tests and the headless demo.
* :class:`PeerStateRepository` Per-robot view of other robots' states.

The bus only transports messages. It does not interpret them and
does not make any decision. Replacing this with a UDP / TCP /
WebSocket / MQTT transport later only requires implementing the
:class:`MessageBus` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

CellCoord = Tuple[int, int]


class MessageType(Enum):
    """Closed set of message kinds the bus currently understands."""

    ROBOT_STATE = "robot_state"
    HEARTBEAT = "heartbeat"
    RESERVATION_UPDATE = "reservation_update"
    CONFLICT_PROPOSAL = "conflict_proposal"
    CONFLICT_RESPONSE = "conflict_response"
    PEER_FAILED = "peer_failed"
    TASK_REASSIGN_REQUEST = "task_reassign_request"


@dataclass
class Message:
    """A typed envelope passed through the bus.

    ``payload`` is intentionally typed as ``Any``: the bus does not
    interpret it. Callers narrow based on ``msg_type``.
    """

    sender_id: str
    timestamp: int
    msg_type: MessageType
    payload: Any


@dataclass(frozen=True)
class RobotState:
    """The structured state a robot periodically broadcasts.

    Fields are deliberately explicit (no dict blobs) so that future
    layers can rely on a stable schema.

    Attributes
        ``robot_id``:
            Sender's id.
        ``timestamp``:
            Tick at which the broadcast was produced.
        ``sequence``:
            Strictly monotonically increasing per-sender sequence
            number. Peers use it to detect dropped or reordered
            messages; together with ``timestamp`` it provides the
            basis for liveness tracking (Phase 2C).
        ``position``:
            ``(x, y)`` current cell.
        ``velocity``:
            ``(vx, vy)`` last applied velocity (zero if stationary).
        ``battery``:
            Remaining battery.
        ``task_id``:
            Assigned task id, or ``None`` if idle.
        ``task_priority``:
            Priority of the currently assigned task, or ``0`` if idle.
            Added in Phase 2B so peers can compute priority scores
            without inspecting the simulator's task list. Existing
            callers that omit this field get the default ``0``.
        ``status``:
            String form of :class:`simulation.robot.RobotStatus`.
        ``planned_path``:
            Remaining planned cells (next cell first). May be empty
            if the robot has no active path.
        ``intent``:
            Free-form string (e.g. ``"moving_to_pickup"``,
            ``"idle"``). Kept as a string for forward compatibility.
    """

    robot_id: str
    timestamp: int
    position: CellCoord
    velocity: Tuple[int, int]
    battery: float
    task_id: Optional[str]
    status: str
    planned_path: Tuple[CellCoord, ...]
    sequence: int = 0
    intent: str = "unknown"
    task_priority: int = 0


class MessageBus(ABC):
    """Abstract transport interface.

    The bus only moves messages around. Implementations must be
    deterministic for a given sequence of :meth:`publish` calls.
    """

    @abstractmethod
    def register_peer(self, peer_id: str) -> None:
        """Register a peer so its inbox can receive messages."""

    @abstractmethod
    def publish(self, msg: Message) -> None:
        """Append a message to the outbox for later delivery."""

    @abstractmethod
    def deliver(self) -> int:
        """Move every queued outbox message into recipient inboxes.

        A message published by ``sender_id`` is delivered to every
        other registered peer in publish order. Returns the number
        of messages distributed.
        """

    @abstractmethod
    def drain_inbox(self, peer_id: str) -> List[Message]:
        """Return and clear the inbox of ``peer_id``."""

    @abstractmethod
    def inbox_size(self, peer_id: str) -> int:
        """Return the number of messages currently in the inbox."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all outbox and inbox state."""

    @abstractmethod
    def peer_ids(self) -> List[str]:
        """Return the list of registered peer ids."""


class InProcessMessageBus(MessageBus):
    """Deterministic in-process FIFO bus.

    Behaviour:

    * ``publish`` appends to outbox in call order.
    * ``deliver`` moves outbox messages into each recipient's inbox
      in publish order, skipping the sender.
    * ``drain_inbox`` returns and empties the inbox.

    Determinism is preserved because we only iterate the (already
    ordered) outbox and per-peer inboxes in insertion order.

    Phase 2C fault injection (used by deterministic failure tests):
    a per-peer ``drop_mask`` lets the simulator drop messages destined
    for a peer, simulating communication failure. The mask is
    per-peer-id and persists across calls until explicitly cleared.
    """

    def __init__(self, peer_ids: Optional[Iterable[str]] = None) -> None:
        self._outbox: Deque[Message] = deque()
        self._inboxes: Dict[str, Deque[Message]] = {}
        # Phase 2C: per-recipient drop flag. If True, this recipient
        # never receives any published messages (deterministic
        # communication-failure simulation).
        self._drop_mask: Dict[str, bool] = {}

        if peer_ids is not None:
            for pid in peer_ids:
                self.register_peer(pid)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_peer(self, peer_id: str) -> None:
        self._inboxes.setdefault(peer_id, deque())
        self._drop_mask.setdefault(peer_id, False)

    def peer_ids(self) -> List[str]:
        return list(self._inboxes.keys())

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def publish(self, msg: Message) -> None:
        if msg.sender_id not in self._inboxes:
            # Auto-register unknown senders so the demo never crashes on
            # forgotten registration. Tests can detect this via
            # ``peer_ids()``.
            self.register_peer(msg.sender_id)
        self._outbox.append(msg)

    def is_registered(self, peer_id: str) -> bool:
        return peer_id in self._inboxes

    # ------------------------------------------------------------------
    # Phase 2C: communication-failure injection
    # ------------------------------------------------------------------
    def drop_messages_to(self, peer_id: str, drop: bool = True) -> None:
        """Set the drop flag for messages destined for ``peer_id``.

        When ``drop=True`` the bus still consumes the message from the
        outbox (so ``deliver`` returns the same count semantics) but
        does NOT push it into the recipient's inbox. This simulates a
        deterministic communication failure for that peer.
        """
        self._drop_mask[peer_id] = bool(drop)

    def clear_drop_mask(self) -> None:
        """Clear every peer's drop flag (default: pass-through)."""
        for pid in self._drop_mask:
            self._drop_mask[pid] = False

    def is_dropping_to(self, peer_id: str) -> bool:
        return self._drop_mask.get(peer_id, False)

    def deliver(self) -> int:
        delivered = 0
        while self._outbox:
            msg = self._outbox.popleft()
            for peer_id, inbox in self._inboxes.items():
                if peer_id == msg.sender_id:
                    continue
                if self._drop_mask.get(peer_id, False):
                    # Simulate dropped message: consume it but do not
                    # enqueue.
                    continue
                inbox.append(msg)
                delivered += 1
        return delivered

    def drain_inbox(self, peer_id: str) -> List[Message]:
        inbox = self._inboxes.get(peer_id)
        if not inbox:
            return []
        messages = list(inbox)
        inbox.clear()
        return messages

    def inbox_size(self, peer_id: str) -> int:
        return len(self._inboxes.get(peer_id, ()))

    def reset(self) -> None:
        self._outbox.clear()
        for inbox in self._inboxes.values():
            inbox.clear()


class PeerStateRepository:
    """A robot's local view of other robots' latest states.

    Each robot owns its own repository. There is intentionally no
    global authoritative state object.
    """

    def __init__(self, self_id: str, stale_threshold_ticks: int = 3) -> None:
        self.self_id = self_id
        self.stale_threshold_ticks = max(1, int(stale_threshold_ticks))
        # peer_id -> (timestamp, state)
        self._states: Dict[str, Tuple[int, RobotState]] = {}

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update(self, state: RobotState) -> bool:
        """Store a peer's state if it is strictly newer than what we have.

        Returns ``True`` if the state was stored, ``False`` if it was
        older than or equal to an existing record (and ignored).
        """
        existing = self._states.get(state.robot_id)
        if existing is not None and state.timestamp <= existing[0]:
            return False
        self._states[state.robot_id] = (state.timestamp, state)
        return True

    def forget(self, peer_id: str) -> None:
        self._states.pop(peer_id, None)

    def reset(self) -> None:
        self._states.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get(self, peer_id: str) -> Optional[RobotState]:
        entry = self._states.get(peer_id)
        return entry[1] if entry else None

    def all(self) -> Dict[str, RobotState]:
        return {pid: state for pid, (_, state) in self._states.items()}

    def is_fresh(self, peer_id: str, current_tick: int) -> bool:
        entry = self._states.get(peer_id)
        if entry is None:
            return False
        ts = entry[0]
        return (current_tick - ts) <= self.stale_threshold_ticks

    def fresh_peers(self, current_tick: int) -> List[str]:
        return [pid for pid in self._states if self.is_fresh(pid, current_tick)]

    def __contains__(self, peer_id: str) -> bool:
        return peer_id in self._states

    def __len__(self) -> int:
        return len(self._states)