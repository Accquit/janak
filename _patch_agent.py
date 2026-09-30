"""Append Phase 2C methods to RobotCoordination."""

import pathlib

p = pathlib.Path('sih26123/coordination/agent.py')
text = p.read_text(encoding='utf-8')

# Anchor: the closing of _record_proposal (last line of the class).
anchor = "            record.peer_priority = proposal.self_priority\n            record.peer_action = proposal.self_action\n"

new_block = """            record.peer_priority = proposal.self_priority
            record.peer_action = proposal.self_action

    # ==================================================================
    # Phase 2C: heartbeat / staleness / failure / safe-halt helpers
    # ==================================================================
    def record_heartbeat_tick(self, current_tick: int) -> None:
        \"\"\"Mark that this robot just received at least one valid peer
        broadcast at ``current_tick``.
        \"\"\"
        self.last_peer_heartbeat_tick = current_tick

    def mark_peer_stale(self, current_tick: int) -> list:
        \"\"\"Sweep peers older than STALE_THRESHOLD and clear their
        forward reservations. Emits ``PEER_STALE`` and
        ``RESERVATIONS_CLEARED`` events for each dropped peer.
        \"\"\"
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
        \"\"\"Force-declare ``peer_id`` OFFLINE (>= PEER_FAILURE_TIMEOUT).\"\"\"
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
        \"\"\"Force this robot into SAFE_HALT (self-isolation).\"\"\"
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
        \"\"\"Resume ACTIVE after a peer heartbeat is received.\"\"\"
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
        \"\"\"Return the conflict_id of a wait-for cycle in this robot's
        negotiation_records, or ``None``. Cycles of length 1 (self-loops)
        are ignored; cycles of length > 1 are deadlock candidates.
        \"\"\"
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
        \"\"\"Resolve the deadlock by clearing this robot's record of
        ``conflict_id``. Other robots' reservations are preserved.
        \"\"\"
        from ..simulation.events import Event, EventType
        events: list = []
        events.append(Event(
            current_tick,
            EventType.DEADLOCK_RESOLVED,
            conflict_id=conflict_id,
        ))
        self.negotiation_records.pop(conflict_id, None)
        return events
"""

if anchor not in text:
    raise SystemExit("anchor missing")

text = text.replace(anchor, new_block, 1)
p.write_text(text, encoding='utf-8')
print("OK")
