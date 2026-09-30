import pathlib
p = pathlib.Path('sih26123/simulation/simulator.py')
text = p.read_text(encoding='utf-8')
old = """        # Phase 2C 4b: stale-peer sweep. Any peer whose last valid
        # heartbeat is older than STALE_THRESHOLD ticks has its
        # forward reservations cleared and is marked STALE in the
        # local view.
        from ..coordination.resilience import STALE_THRESHOLD, SELF_ISOLATION_TIMEOUT
        for robot in coordination_robots:
            events.extend(robot.coordination.mark_peer_stale(self.tick_count))"""
new = """        # Phase 2C 4b: stale-peer sweep. Any peer whose last valid
        # heartbeat is older than STALE_THRESHOLD ticks has its
        # forward reservations cleared and is marked STALE in the
        # local view.
        from ..coordination.resilience import (
            PEER_FAILURE_TIMEOUT, SELF_ISOLATION_TIMEOUT, STALE_THRESHOLD,
        )
        for robot in coordination_robots:
            events.extend(robot.coordination.mark_peer_stale(self.tick_count))

        # Phase 2C 4b.5: peer-failure sweep. Any peer whose last
        # valid heartbeat is older than PEER_FAILURE_TIMEOUT ticks is
        # declared OFFLINE; its forward reservations are released.
        failed_peers: Dict[str, List] = {}
        for robot in coordination_robots:
            liveness = robot.coordination.peer_liveness
            for pid, info in liveness.all().items():
                if info.status == "offline":
                    continue
                last = info.last_seen_tick
                if last < 0:
                    continue
                if (self.tick_count - last) >= PEER_FAILURE_TIMEOUT:
                    failed_peers.setdefault(robot.robot_id, []).append(pid)
        for robot_id, peer_ids in failed_peers.items():
            robot = next(r for r in self.robots if r.robot_id == robot_id)
            for pid in peer_ids:
                events.extend(
                    robot.coordination.declare_peer_failed(pid, self.tick_count)
                )"""
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print('OK')
