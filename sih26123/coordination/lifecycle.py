"""Phase 2C failure and resilience primitives.

This module owns:

* :class:`RobotLifecycle`  per-robot state machine (ACTIVE / OFFLINE /
  SAFE_HALT / FAILED).
* :class:`RobotLifecycleState`  enum of the lifecycle states.
* :class:`PeerLiveness`  snapshot of one peer's freshness + the
  status the local robot has assigned to that peer (fresh / stale /
  failed / unknown).
* :class:`PeerLivenessTracker`  per-robot bookkeeping: for every peer
  it tracks the last received broadcast's tick and sequence number,
  the peer's current status, and the last known occupied cell.

The lifecycle is **purely deterministic** and **decentralised**:
each robot owns its own copy of the tracker. No global coordinator.
The tracker is reset only by the simulator at the start of every
tick (the same lifecycle the conflict dedup set has in Phase 2B.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class RobotLifecycleState(Enum):
    """Per-robot lifecycle state used by every other peer."""

    ACTIVE = "active"           # broadcasting normally, accepting tasks
    STALE = "stale"              # silent; only its last cell is trusted
    SAFE_HALT = "safe_halt"       # self-isolated, not making autonomous
                                 # progress that depends on stale peer
                                 # beliefs
    OFFLINE = "offline"           # declared failed by peers (>=40 ticks
                                 # without a valid broadcast)
    FAILED = "failed"             # self-failure detected (battery /
                                 # internal error)


@dataclass
class RobotLifecycle:
    """Owning robot's local view of its own lifecycle.

    The owning robot decides its own state. Peers do NOT write to this
    object; they observe the broadcast ``RobotState.status`` and update
    their own :class:`PeerLivenessTracker`.
    """

    state: RobotLifecycleState = RobotLifecycleState.ACTIVE
    # Tick at which the robot entered SAFE_HALT (for SELF_ISOLATION_TIMEOUT).
    safe_halt_since_tick: int = -1
    # Tick at which the robot was declared FAILED locally (e.g. battery).
    failed_since_tick: int = -1


@dataclass(frozen=True)
class PeerLiveness:
    """Local view of one peer's liveness."""

    peer_id: str
    status: RobotLifecycleState = RobotLifecycleState.ACTIVE
    last_seen_tick: int = -1
    last_seen_sequence: int = -1
    last_known_position: Optional[tuple] = None


class PeerLivenessTracker:
    """Per-robot bookkeeping for peer freshness and status.

    Reset at the start of each tick by the simulator; updated every time
    a peer broadcasts. The owning robot decides nothing about a peer
    based on this directly — the tracker only feeds the staleness and
    failure-detection code in :mod:`coordination.agent` and the simulator.
    """

    def __init__(self) -> None:
        self._peers: Dict[str, PeerLiveness] = {}
        self._last_received: Dict[str, tuple[int, int]] = {}

    def reset(self) -> None:
        """Clear all per-tick bookkeeping. Does NOT drop peers."""
        self._peers.clear()
        self._last_received.clear()

    def record_heartbeat(
        self,
        peer_id: str,
        tick: int,
        sequence: int,
        position: Optional[tuple],
    ) -> bool:
        """Record a heartbeat from ``peer_id`` at ``tick`` / ``sequence``.

        Returns ``True`` if the heartbeat is newer than the previously
        recorded one (strict-greater on tick, then on sequence).
        Returns ``False`` for duplicates or stale messages.
        """
        if not isinstance(peer_id, str) or not peer_id:
            return False
        if type(tick) is not int or type(sequence) is not int:
            return False
        if tick < 0 or sequence < 0:
            return False
        if position is not None:
            if (
                not isinstance(position, (tuple, list))
                or len(position) != 2
                or any(type(value) is not int for value in position)
            ):
                return False
            position = tuple(position)
        previous = self._last_received.get(peer_id, (-1, -1))
        if (tick, sequence) <= previous:
            return False
        self._last_received[peer_id] = (tick, sequence)
        previous_info = self._peers.get(peer_id)
        if position is None and previous_info is not None:
            position = previous_info.last_known_position
        self._peers[peer_id] = PeerLiveness(
            peer_id=peer_id,
            status=RobotLifecycleState.ACTIVE,
            last_seen_tick=tick,
            last_seen_sequence=sequence,
            last_known_position=position,
        )
        return True

    def get(self, peer_id: str) -> Optional[PeerLiveness]:
        return self._peers.get(peer_id)

    def all(self) -> Dict[str, PeerLiveness]:
        """Return a snapshot of all currently-known peers."""
        return dict(self._peers)

    def forget(self, peer_id: str) -> bool:
        """Forget a peer entirely (used for offline / cleanup)."""
        existed = peer_id in self._peers
        self._peers.pop(peer_id, None)
        self._last_received.pop(peer_id, None)
        return existed

    def stale_peers(self, current_tick: int, threshold: int) -> list:
        """Return peers whose last_seen_tick is older than current_tick - threshold."""
        if threshold < 1:
            threshold = 1
        out = []
        for pid, info in self._peers.items():
            if info.status is not RobotLifecycleState.ACTIVE:
                continue
            if info.last_seen_tick < 0:
                continue
            if (current_tick - info.last_seen_tick) >= threshold:
                out.append(pid)
        return out

    def failed_peers(self, current_tick: int, threshold: int) -> list:
        """Return peers whose last_seen_tick is older than current_tick - threshold."""
        if threshold < 1:
            threshold = 1
        out = []
        for pid, info in self._peers.items():
            if info.last_seen_tick < 0:
                continue
            if (current_tick - info.last_seen_tick) >= threshold:
                out.append(pid)
        return out

    def mark_failed(self, peer_id: str) -> bool:
        """Mark a peer as OFFLINE in the local view.

        Returns ``True`` if the peer's status changed (or was newly
        absent).
        """
        prev = self._peers.get(peer_id)
        if prev is not None and prev.status == RobotLifecycleState.OFFLINE:
            return False
        last_seen_tick = prev.last_seen_tick if prev is not None else -1
        last_seen_seq = prev.last_seen_sequence if prev is not None else -1
        last_known_pos = prev.last_known_position if prev is not None else None
        self._peers[peer_id] = PeerLiveness(
            peer_id=peer_id,
            status=RobotLifecycleState.OFFLINE,
            last_seen_tick=last_seen_tick,
            last_seen_sequence=last_seen_seq,
            last_known_position=last_known_pos,
        )
        return True

    def mark_stale(self, peer_id: str) -> bool:
        """Mark a silent peer stale while retaining its last known cell."""
        prev = self._peers.get(peer_id)
        if prev is None or prev.status is not RobotLifecycleState.ACTIVE:
            return False
        self._peers[peer_id] = PeerLiveness(
            peer_id=peer_id,
            status=RobotLifecycleState.STALE,
            last_seen_tick=prev.last_seen_tick,
            last_seen_sequence=prev.last_seen_sequence,
            last_known_position=prev.last_known_position,
        )
        return True

    def __contains__(self, peer_id: str) -> bool:
        return peer_id in self._peers

    def __len__(self) -> int:
        return len(self._peers)
