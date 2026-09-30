"""Per-robot coordination orchestrator (Phase 2A + Phase 2B).

:class:`RobotCoordination` is the *local* counterpart of a network
agent. It is owned by a single robot and:

* Drains the robot's inbox from the message bus, updating its
  :class:`PeerStateRepository` and refreshing peer reservations.
* Rebuilds the robot's own reservations from its current state.
* Runs :class:`ConflictDetector` and emits predicted-conflict events.
* (Phase 2B) Negotiates each predicted conflict locally, deciding
  whether to PROCEED or to YIELD (WAIT / REROUTE), broadcasts a
  :class:`ConflictProposal` and emits the negotiation outcome events.
* Publishes the robot's own :class:`RobotState` to the bus.

The orchestrator is **decentralised**: it only reads from the bus,
only writes its own state, and only emits events. It does not let
other robots choose for it. Decisions are made from local inputs â€”
the peer's priority comes from the latest :class:`RobotState`
broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .communication import (
    Message,
    MessageBus,
    MessageType,
    PeerStateRepository,
    RobotState,
)
from .conflicts import ConflictDetector, ConflictType, PredictedConflict
from .negotiation import (
    ConflictProposal,
    NegotiationAction,
    NegotiationRecord,
    make_conflict_id,
)
from .priority import PriorityInputs, _clip01, compute_priority, is_higher_priority
from .reservations import ReservationTable
from .spacetime import space_time_astar, spatial_path
from .lifecycle import (
    PeerLiveness,
    PeerLivenessTracker,
    RobotLifecycle,
    RobotLifecycleState,
)
from .resilience import (
    DEADLOCK_WINDOW,
    HEARTBEAT_INTERVAL,
    PEER_FAILURE_TIMEOUT,
    SELF_ISOLATION_TIMEOUT,
    STALE_THRESHOLD,
)

CellCoord = tuple[int, int]


@dataclass
class RobotCoordination:
    """The local coordination module of a single robot.

    Parameters
        ``robot_id``:
            The id of the owning robot.
        ``bus``:
            The shared :class:`MessageBus`.
        ``peer_states``:
            A :class:`PeerStateRepository` for this robot.
        ``reservations``:
            A :class:`ReservationTable` for this robot (own + peer).
        ``conflict_detector``:
            A :class:`ConflictDetector` bound to the above table.
        ``broadcast_period``:
            Broadcast at most once every N ticks. ``1`` means every
            tick. ``0`` disables broadcasting.
        ``last_broadcast_tick``:
            Tick of the most recent broadcast (``-1`` if none yet).
        ``negotiation_records``:
            Local map of in-flight negotiations keyed by conflict id.
            The owner uses these to track decisions; the contents are
            intentionally per-robot (no global authority).
    """

    robot_id: str
    bus: MessageBus
    peer_states: PeerStateRepository
    reservations: ReservationTable
    conflict_detector: ConflictDetector
    broadcast_period: int = 1
    last_broadcast_tick: int = field(default=-1)
    negotiation_records: Dict[str, NegotiationRecord] = field(default_factory=dict)

    # Phase 2B.2: per-tick set of conflict_ids that this robot has
    # already negotiated in the current tick. Used to deduplicate
    # processing of the same conflict across the first-round
    # (step 7) and second-round (step 9b) negotiations within
    # one tick. Reset by the simulator at the start of each tick.
    processed_conflict_ids_this_tick: set = field(default_factory=set)

    # Phase 2C: monotonically increasing sequence number for outgoing
    # heartbeats. Reset by the owning robot's tick() when the
    # simulator calls ``process_inbox`` (the bus still uses FIFO
    # ordering so out-of-order delivery is detectable).
    sequence: int = 0

    # Phase 2C: lifecycle of this robot (ACTIVE / SAFE_HALT /
    # OFFLINE / FAILED) plus the ticks at which it entered the
    # latter three states.
    lifecycle: "RobotLifecycle" = field(default_factory=lambda: __import__(
        "sih26123.coordination.lifecycle",
        fromlist=["RobotLifecycle"],
    ).RobotLifecycle())

    # Phase 2C: bookkeeping for peer freshness / status. Reset by
    # the simulator at the start of each tick.
    peer_liveness: "PeerLivenessTracker" = field(default_factory=lambda: __import__(
        "sih26123.coordination.lifecycle",
        fromlist=["PeerLivenessTracker"],
    ).PeerLivenessTracker())

    # Phase 2C: monotonic tick at which the owning robot last received
    # at least one valid peer heartbeat. Used to drive the
    # SELF_ISOLATION_TIMEOUT self-isolation check.
    last_peer_heartbeat_tick: int = field(default=-1)

    # ------------------------------------------------------------------
    # Inbox
    # ------------------------------------------------------------------
    def process_inbox(self) -> int:
        """Drain the inbox and update peer state + reservations.

        Also records incoming :class:`ConflictProposal` messages for
        transparency. The owner does **not** change its own decision
        based on the peer's proposal â€” every decision is local. The
        recorded information is useful for future deadlock detection.

        Returns the number of state messages processed.

        Phase 2C also updates the ``peer_liveness`` tracker from each
        successfully-decoded peer ``ROBOT_STATE`` and bumps
        ``last_peer_heartbeat_tick`` whenever at least one peer
        broadcast was received this tick.
        """
        messages = self.bus.drain_inbox(self.robot_id)
        processed = 0
        peer_received_this_call = False
        for msg in messages:
            if msg.msg_type is MessageType.ROBOT_STATE:
                state = msg.payload
                if isinstance(state, RobotState):
                    if state.robot_id == self.robot_id:
                        continue  # never from self
                    if self.peer_states.update(state):
                        self.reservations.update_peer_from_state(
                            peer_id=state.robot_id,
                            peer_position=state.position,
                            peer_path=list(state.planned_path),
                            peer_timestamp=state.timestamp,
                            horizon=self.conflict_detector.lookahead_horizon,
                        )
                        # Phase 2C: track peer freshness for staleness,
                        # isolation, and failure detection.
                        self.peer_liveness.record_heartbeat(
                            peer_id=state.robot_id,
                            tick=state.timestamp,
                            sequence=state.sequence,
                            position=state.position,
                        )
                        processed += 1
                        peer_received_this_call = True
            elif msg.msg_type is MessageType.HEARTBEAT:
                # Phase 2C: bare heartbeat with no payload, used for
                # liveness-only signals (no reservation update).
                info = msg.payload if isinstance(msg.payload, dict) else {}
                pid = info.get("peer_id", msg.sender_id)
                tick_v = info.get("tick", msg.timestamp)
                seq_v = info.get("sequence", 0)
                pos_v = info.get("position")
                if pid == self.robot_id:
                    continue
                self.peer_liveness.record_heartbeat(
                    peer_id=pid,
                    tick=tick_v,
                    sequence=seq_v,
                    position=pos_v,
                )
                peer_received_this_call = True
            elif msg.msg_type is MessageType.CONFLICT_PROPOSAL:
                proposal = msg.payload
                if isinstance(proposal, ConflictProposal):
                    self._record_proposal(proposal)
        if peer_received_this_call:
            self.last_peer_heartbeat_tick = self.last_peer_heartbeat_tick
            # The simulator updates last_peer_heartbeat_tick externally
            # by passing the current tick in (see Simulator.tick step
            # 4b). Local recording is left here as a hook.
        return processed

    # ------------------------------------------------------------------
    # Own reservations + conflict detection
    # ------------------------------------------------------------------
    def update_own_reservations(
        self,
        current_pos: CellCoord,
        path: list[CellCoord],
        current_tick: int,
    ) -> None:
        """Rebuild own reservations from current robot state."""
        self.reservations.update_own_for_path(
            current_pos=current_pos,
            path=path,
            start_tick=current_tick,
            horizon=self.conflict_detector.lookahead_horizon,
        )

    def predict_conflicts(self, current_tick: int) -> List[PredictedConflict]:
        """Run the conflict detector and return predicted conflicts."""
        return self.conflict_detector.predict(current_tick)

    # ------------------------------------------------------------------
    # Negotiation (Phase 2B)
    # ------------------------------------------------------------------
    def run_negotiation(
        self,
        robot,
        warehouse,
        current_tick: int,
        conflicts: List[PredictedConflict],
        max_task_priority: int,
    ) -> List:
        """Run a local negotiation for each predicted conflict.

        Returns events (``CONFLICT_NEGOTIATION_STARTED``,
        ``CONFLICT_NEGOTIATION_RESOLVED``, ``ROBOT_YIELDED``,
        ``ROBOT_REROUTED``, ``RESERVATION_CONFLICT``). The robot's
        path may be updated; the simulator must call this **before**
        :meth:`Robot.step` so that motion follows the new path.
        """
        from ..simulation.events import Event, EventType

        events: List = []
        if not conflicts:
            return events

        max_task_priority = max(1, int(max_task_priority))

        # An idle robot (no current task) has nothing to protect. It
        # does not negotiate: it stays put and other robots must
        # reroute around it. This avoids spurious yield decisions driven
        # by a zero-priority idle robot (which would otherwise always
        # rank lower than any active peer).
        if robot.current_task is None:
            return events

        for conflict in conflicts:
            peer_id = (
                conflict.robot_b
                if conflict.robot_a == self.robot_id
                else conflict.robot_a
            )
            if peer_id == self.robot_id:
                continue

            conflict_id = make_conflict_id(
                conflict.conflict_type,
                conflict.robot_a,
                conflict.robot_b,
                conflict.timestep,
                conflict.cell,
                conflict.edge,
            )

            # Phase 2B.2 dedup: skip if this exact conflict_id was
            # already processed in the current tick (by this robot
            # in an earlier round, or by the peer robot). The
            # conflict_id fully identifies the conflict (robot pair +
            # cell + timestep + type), so a dedup hit means the
            # underlying negotiation is already done for this tick.
            if conflict_id in self.processed_conflict_ids_this_tick:
                continue

            self_priority = self._compute_self_priority(
                robot, current_tick, max_task_priority,
            )

            peer_state = self.peer_states.get(peer_id)
            peer_priority = self._compute_peer_priority(
                peer_state, max_task_priority,
            )

            # If the robot is *already* at the conflict cell, give it
            # implicit priority over the arriving robot. The arriving
            # robot should reroute instead.
            if conflict.cell is not None and robot.position == conflict.cell:
                self_priority += 1.0

            # Local decision â€” no waiting on a response.
            self_proceeds = is_higher_priority(
                self_priority, peer_priority, self.robot_id, peer_id,
            )
            self_action = (
                NegotiationAction.PROCEED if self_proceeds else NegotiationAction.YIELD
            )

            record = NegotiationRecord(
                conflict_id=conflict_id,
                peer_id=peer_id,
                conflict_type=conflict.conflict_type,
                timestep=conflict.timestep,
                cell=conflict.cell,
                edge=conflict.edge,
                initiated_tick=current_tick,
                self_priority=self_priority,
                self_action=self_action,
                peer_priority=peer_priority,
                peer_action=None,
                resolved=True,
                resolution=self_action,
                resolution_reason="local_priority_decision",
            )
            self.negotiation_records[conflict_id] = record

            # Phase 2B.2: mark this conflict_id as processed in
            # the current tick so the second-round negotiation
            # (and any subsequent negotiation rounds in this
            # tick) dedup against this one.
            self.processed_conflict_ids_this_tick.add(conflict_id)

            # 1. Negotiation started.
            events.append(Event(
                current_tick,
                EventType.CONFLICT_NEGOTIATION_STARTED,
                robot_id=self.robot_id,
                peer_id=peer_id,
                conflict_id=conflict_id,
                conflict_type=conflict.conflict_type.value,
                timestep=conflict.timestep,
                cell=list(conflict.cell) if conflict.cell is not None else None,
            ))

            # 2. Reservation conflict (always emitted when both robots
            #    have predicted overlap).
            events.append(Event(
                current_tick,
                EventType.RESERVATION_CONFLICT,
                robot_id=self.robot_id,
                peer_id=peer_id,
                conflict_id=conflict_id,
                timestep=conflict.timestep,
                cell=list(conflict.cell) if conflict.cell is not None else None,
            ))

            # 3. If we yield, choose WAIT or REROUTE.
            #    But: if the robot is already in a negotiation-driven
            #    WAITING period, do NOT yield again â€” that would extend
            #    the wait indefinitely (the conflict timestep keeps
            #    moving forward with the existing reservations). The
            #    robot's current wait covers all subsequent conflicts.
            if not self_proceeds and robot.waiting_until_tick is None:
                events.extend(self._execute_yield(
                    robot, warehouse, current_tick, conflict, conflict_id,
                    peer_id, max_task_priority,
                ))
            elif not self_proceeds:
                # Already waiting. Skip the yield but keep the local
                # decision recorded.
                self_action_record = self_action
            else:
                self_action_record = self_action

            # 4. Resolved.
            events.append(Event(
                current_tick,
                EventType.CONFLICT_NEGOTIATION_RESOLVED,
                robot_id=self.robot_id,
                peer_id=peer_id,
                conflict_id=conflict_id,
                self_action=self_action.value,
                self_priority=round(self_priority, 4),
                peer_priority=round(peer_priority, 4),
            ))

            # 5. Broadcast the proposal (informational exchange).
            self._send_proposal(
                conflict,
                conflict_id,
                self_priority,
                self_action,
                remaining_path_length=len(robot.remaining_path()),
            )

        return events

    # ------------------------------------------------------------------
    # Outbox
    # ------------------------------------------------------------------
    def should_broadcast(self, current_tick: int) -> bool:
        if self.broadcast_period <= 0:
            return False
        if self.last_broadcast_tick < 0:
            return True  # never broadcast yet
        return (current_tick - self.last_broadcast_tick) >= self.broadcast_period

    def publish_state(
        self,
        current_tick: int,
        position: CellCoord,
        velocity: tuple[int, int],
        battery: float,
        task_id: Optional[str],
        status: str,
        planned_path: list[CellCoord],
        intent: str = "unknown",
        task_priority: int = 0,
    ) -> bool:
        """Publish this robot's :class:`RobotState` to the bus.

        Returns ``True`` if a message was published, ``False`` if the
        broadcast period suppressed it.

        Phase 2C: bumps the local ``sequence`` counter so peers can
        detect dropped or reordered heartbeats.
        """
        if not self.should_broadcast(current_tick):
            return False
        self.sequence += 1
        state = RobotState(
            robot_id=self.robot_id,
            timestamp=current_tick,
            sequence=self.sequence,
            position=tuple(position),
            velocity=tuple(velocity),
            battery=float(battery),
            task_id=task_id,
            status=str(status),
            planned_path=tuple(tuple(c) for c in planned_path),
            intent=intent,
            task_priority=int(task_priority),
        )
        msg = Message(
            sender_id=self.robot_id,
            timestamp=current_tick,
            msg_type=MessageType.ROBOT_STATE,
            payload=state,
        )
        self.bus.publish(msg)
        self.last_broadcast_tick = current_tick
        return True

    # ------------------------------------------------------------------
    # Phase 2C: heartbeat helpers
    # ------------------------------------------------------------------
    def should_send_heartbeat(self, current_tick: int) -> bool:
        """Return True if this robot should send a heartbeat this tick.

        Defaults to the standard broadcast cadence
        (HEARTBEAT_INTERVAL). Robots that have failed or are SAFE_HALT
        do not broadcast.
        """
        if self.lifecycle.state != RobotLifecycleState.ACTIVE:
            return False
        return self.should_broadcast(current_tick)

    def send_heartbeat(self, current_tick: int, position: CellCoord) -> bool:
        """Send a bare heartbeat with no reservation payload.

        Useful when the robot has nothing new to publish but peers
        need a liveness signal. Returns ``True`` if a message was sent.
        """
        if self.lifecycle.state != RobotLifecycleState.ACTIVE:
            return False
        self.sequence += 1
        msg = Message(
            sender_id=self.robot_id,
            timestamp=current_tick,
            msg_type=MessageType.HEARTBEAT,
            payload={
                "peer_id": self.robot_id,
                "tick": current_tick,
                "sequence": self.sequence,
                "position": tuple(position),
            },
        )
        self.bus.publish(msg)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_self_priority(
        self,
        robot,
        current_tick: int,
        max_task_priority: int,
    ) -> float:
        task_priority = 0
        deadline = None
        if robot.current_task is not None:
            task_priority = int(getattr(robot.current_task, "priority", 0) or 0)
            deadline = getattr(robot.current_task, "deadline", None)

        total_path_length = max(len(robot.path), 1)
        remaining_path_length = len(robot.remaining_path())

        inputs = PriorityInputs(
            robot_id=self.robot_id,
            task_priority=task_priority,
            max_task_priority=max_task_priority,
            deadline=deadline,
            current_tick=current_tick,
            battery=float(robot.battery),
            battery_capacity=float(robot.battery_capacity),
            waiting_time=int(getattr(robot, "waiting_time", 0) or 0),
            remaining_path_length=remaining_path_length,
            expected_path_length=total_path_length,
            max_horizon=self.conflict_detector.lookahead_horizon,
        )
        return compute_priority(inputs)

    @staticmethod
    def _compute_peer_priority(peer_state, max_task_priority: int) -> float:
        """Estimate the peer's priority from its latest broadcast.

        Uses whatever fields the peer's :class:`RobotState` exposes
        (``task_priority`` and ``battery``) and treats the others as
        unknown. If the peer's state is unknown entirely we default to
        ``0`` â€” the cautious choice that means "let the other robot
        proceed if it can". This avoids pre-emptively yielding to a
        ghost peer.
        """
        if peer_state is None:
            return 0.0
        # Use the same priority formula as ``_compute_self_priority``,
        # but feed it with whatever fields we know about the peer.
        # Deadline / waiting-time / route are unknown to us, so we
        # default them to neutral values.
        battery_capacity = max_task_priority * 100  # arbitrary nominal
        inputs = PriorityInputs(
            robot_id=getattr(peer_state, "robot_id", ""),
            task_priority=int(getattr(peer_state, "task_priority", 0) or 0),
            max_task_priority=max_task_priority,
            deadline=None,
            current_tick=int(getattr(peer_state, "timestamp", 0) or 0),
            battery=float(getattr(peer_state, "battery", 0.0) or 0.0),
            battery_capacity=battery_capacity,
            waiting_time=0,
            remaining_path_length=0,
            expected_path_length=1,
            max_horizon=1,
        )
        return compute_priority(inputs)

    def _execute_yield(
        self,
        robot,
        warehouse,
        current_tick: int,
        conflict,
        conflict_id: str,
        peer_id: str,
        max_task_priority: int,
    ) -> List:
        from ..simulation.events import Event, EventType
        from ..simulation.robot import RobotStatus
        from .resolver import choose_yield_action

        events: List = []
        destination = robot.destination
        reroute_path: Optional[List[CellCoord]] = None
        if destination is not None and destination != robot.position:
            reroute_path = self._reroute(
                robot, destination, warehouse, current_tick,
            )

        old_remaining = len(robot.remaining_path())
        # If a reroute was found, drop the leading start cell when
        # reporting the cost (the cost is "extra cells beyond start").
        if reroute_path is not None:
            # The first cell of the spatial path equals the start; the
            # extra cost is len-1 minus the start cell.
            spatial_cells = max(0, len(reroute_path) - 1)
        else:
            spatial_cells = None
        choice = choose_yield_action(
            conflict_timestep=conflict.timestep,
            current_tick=current_tick,
            reroute_path_length=spatial_cells,
            old_remaining_length=old_remaining,
        )

        if choice.action == NegotiationAction.REROUTE and reroute_path is not None:
            # Drop the leading start cell so Robot.set_path applies its
            # own stripping rule consistently.
            spatial = list(reroute_path)
            if spatial and spatial[0] == robot.position:
                spatial = spatial[1:]
            robot.set_path(spatial, destination)
            self.update_own_reservations(
                robot.position, robot.remaining_path(), current_tick,
            )
            events.append(Event(
                current_tick,
                EventType.ROBOT_REROUTED,
                robot_id=self.robot_id,
                peer_id=peer_id,
                conflict_id=conflict_id,
                reroute_cost=choice.reroute_cost,
                wait_cost=choice.wait_cost,
                reason=choice.reason,
                new_path_length=len(robot.remaining_path()),
            ))
        else:
            robot.status = RobotStatus.WAITING
            robot.waiting_until_tick = current_tick + choice.wait_ticks
            events.append(Event(
                current_tick,
                EventType.ROBOT_YIELDED,
                robot_id=self.robot_id,
                peer_id=peer_id,
                conflict_id=conflict_id,
                wait_ticks=choice.wait_ticks,
                reason=choice.reason,
            ))

        return events

    def _reroute(
        self,
        robot,
        destination: CellCoord,
        warehouse,
        current_tick: int,
    ) -> Optional[List[CellCoord]]:
        """Run space-time A* to find a path that avoids peer reservations.

        The search horizon is at least the lookahead horizon plus the
        Manhattan distance from the current cell to the destination â€”
        this guarantees the search is wide enough to reach the goal
        even when the lookahead itself is short.
        """

        def is_blocked(x: int, y: int, t: int) -> bool:
            if not (0 <= x < warehouse.width and 0 <= y < warehouse.height):
                return True
            if not warehouse.is_traversable(x, y):
                return True
            bucket = self.reservations.robots_at((x, y), t)
            return any(rid != self.robot_id for rid in bucket)

        sx, sy = robot.position
        gx, gy = destination
        manhattan = abs(sx - gx) + abs(sy - gy)
        max_timestep = (
            current_tick
            + max(self.conflict_detector.lookahead_horizon, manhattan + 4)
        )
        path = space_time_astar(
            start=robot.position,
            goal=destination,
            start_tick=current_tick,
            width=warehouse.width,
            height=warehouse.height,
            is_cell_traversable=warehouse.is_traversable,
            is_blocked_at=is_blocked,
            max_timestep=max_timestep,
        )
        if path is None:
            return None
        return spatial_path(path)

    def _send_proposal(
        self,
        conflict,
        conflict_id: str,
        self_priority: float,
        self_action: NegotiationAction,
        remaining_path_length: int,
    ) -> None:
        peer_id = (
            conflict.robot_b
            if conflict.robot_a == self.robot_id
            else conflict.robot_a
        )
        proposal = ConflictProposal(
            conflict_id=conflict_id,
            sender_id=self.robot_id,
            peer_id=peer_id,
            conflict_type=conflict.conflict_type,
            timestep=conflict.timestep,
            cell=conflict.cell,
            edge=conflict.edge,
            self_priority=self_priority,
            self_action=self_action,
            self_path_length=remaining_path_length,
        )
        self.bus.publish(Message(
            sender_id=self.robot_id,
            timestamp=0,
            msg_type=MessageType.CONFLICT_PROPOSAL,
            payload=proposal,
        ))

    # ------------------------------------------------------------------
    # Recording peer proposals (informational)
    # ------------------------------------------------------------------
    def _record_proposal(self, proposal: ConflictProposal) -> None:
        if proposal.peer_id != self.robot_id:
            return
        record = self.negotiation_records.get(proposal.conflict_id)
        if record is None:
            record = NegotiationRecord(
                conflict_id=proposal.conflict_id,
                peer_id=proposal.sender_id,
                conflict_type=ConflictType.VERTEX,
                timestep=0,
                peer_priority=proposal.self_priority,
                peer_action=proposal.self_action,
            )
            self.negotiation_records[proposal.conflict_id] = record
        else:
            record.peer_priority = proposal.self_priority
            record.peer_action = proposal.self_action

    # ==================================================================
    # Phase 2C: heartbeat / staleness / failure / safe-halt helpers
    # ==================================================================
    def record_heartbeat_tick(self, current_tick: int) -> None:
        """Mark that this robot just received at least one valid peer
        broadcast at ``current_tick``.
        """
        self.last_peer_heartbeat_tick = current_tick

    def mark_peer_stale(self, current_tick: int) -> list:
        """Sweep peers older than STALE_THRESHOLD and clear their
        forward reservations. Emits ``PEER_STALE`` and
        ``RESERVATIONS_CLEARED`` events for each dropped peer.
        """
        from ..simulation.events import Event, EventType
        events: list = []
        stale_ids = self.peer_liveness.stale_peers(current_tick, STALE_THRESHOLD)
        for pid in stale_ids:
            self.peer_states.forget(pid)
            cleared = self.reservations.clear_robot(pid)
            events.append(Event(
                current_tick,
                EventType.PEER_STALE,
                robot_id=self.robot_id,
                peer_id=pid,
                since_tick=current_tick - STALE_THRESHOLD,
            ))
            events.append(Event(
                current_tick,
                EventType.RESERVATIONS_CLEARED,
                robot_id=self.robot_id,
                peer_id=pid,
                cells_cleared=cleared,
            ))
        return events

    def declare_peer_failed(self, peer_id: str, current_tick: int) -> list:
        """Force-declare ``peer_id`` OFFLINE (>= PEER_FAILURE_TIMEOUT)."""
        from ..simulation.events import Event, EventType
        events: list = []
        cleared = self.reservations.clear_robot(peer_id)
        self.peer_liveness.mark_failed(peer_id)
        events.append(Event(
            current_tick,
            EventType.PEER_FAILED,
            robot_id=self.robot_id,
            peer_id=peer_id,
            since_tick=current_tick - PEER_FAILURE_TIMEOUT,
        ))
        events.append(Event(
            current_tick,
            EventType.RESERVATIONS_CLEARED,
            robot_id=self.robot_id,
            peer_id=peer_id,
            cells_cleared=cleared,
        ))
        return events

    def enter_safe_halt(self, current_tick: int) -> list:
        """Force this robot into SAFE_HALT (self-isolation)."""
        from ..simulation.events import Event, EventType
        events: list = []
        if self.lifecycle.state == RobotLifecycleState.SAFE_HALT:
            return events
        self.lifecycle.state = RobotLifecycleState.SAFE_HALT
        self.lifecycle.safe_halt_since_tick = current_tick
        events.append(Event(
            current_tick,
            EventType.ROBOT_SAFE_HALT,
            robot_id=self.robot_id,
            since_tick=current_tick - SELF_ISOLATION_TIMEOUT,
        ))
        return events

    def exit_safe_halt(self, current_tick: int) -> list:
        """Resume ACTIVE after a peer heartbeat is received."""
        from ..simulation.events import Event, EventType
        events: list = []
        if self.lifecycle.state != RobotLifecycleState.SAFE_HALT:
            return events
        self.lifecycle.state = RobotLifecycleState.ACTIVE
        self.lifecycle.safe_halt_since_tick = -1
        events.append(Event(
            current_tick,
            EventType.ROBOT_HEARTBEAT,
            robot_id=self.robot_id,
            tag='recovered',
        ))
        return events

    def detect_deadlock(self, current_tick: int) -> Optional[str]:
        """Return the conflict_id of a wait-for cycle in this robot's
        negotiation_records, or ``None``. Cycles of length 1 (self-loops)
        are ignored; cycles of length > 1 are deadlock candidates.
        """
        waits_for: Dict[str, set] = {}
        for cid, rec in self.negotiation_records.items():
            if rec.peer_id == self.robot_id:
                continue
            if rec.resolution is None:
                continue
            waits_for.setdefault(self.robot_id, set()).add(rec.peer_id)
        WHITE, GREY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {pid: WHITE for pid in waits_for}

        def dfs(node: str) -> bool:
            colour[node] = GREY
            for neighbour in waits_for.get(node, ()):  # type: ignore[arg-type]
                if colour.get(neighbour, WHITE) == GREY:
                    return True
                if colour.get(neighbour, WHITE) == WHITE and dfs(neighbour):
                    return True
            colour[node] = BLACK
            return False

        for pid in waits_for:
            if colour[pid] == WHITE:
                if dfs(pid):
                    candidates = sorted(
                        cid for cid, rec in self.negotiation_records.items()
                        if rec.resolution is not None and rec.peer_id != self.robot_id
                    )
                    return candidates[0] if candidates else None
        return None

    def resolve_deadlock(self, conflict_id: str, current_tick: int) -> list:
        """Resolve the deadlock by clearing this robot's record of
        ``conflict_id``. Other robots' reservations are preserved.
        """
        from ..simulation.events import Event, EventType
        events: list = []
        events.append(Event(
            current_tick,
            EventType.DEADLOCK_RESOLVED,
            conflict_id=conflict_id,
        ))
        self.negotiation_records.pop(conflict_id, None)
        return events
