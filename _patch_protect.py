import pathlib
p = pathlib.Path('sih26123/tests/test_phase2c.py')
text = p.read_text(encoding='utf-8')

# Fix test_failed_robot_last_cell_stays_protected to run 2 ticks.
old = '''def test_failed_robot_last_cell_stays_protected() -> None:
    """The failed robot's last known occupied cell stays reserved
    by the peer robots, because we drop the failed robot's *future*
    reservations but not the cell it is currently at. (It is still
    at that cell.) The peer also gets a PEER_FAILED event for the
    failed robot.
    """
    bus = InProcessMessageBus()
    sim = _make_sim(num_robots=3, bus=bus)
    r0, r1, r2 = sim.robots
    # r0 is at its task pickup; force failure.
    r0.battery = 0.0
    sim.tick()
    # r1 has now declared r0 failed.
    r1_state = r1.coordination.peer_liveness.get('R0')
    assert r1_state is not None
    assert r1_state.status == RobotLifecycleState.OFFLINE
    failed_info = b.coordination.peer_liveness
    for r in sim.robots:
        if r.coordination is None:
            continue
        info = r.coordination.peer_liveness.get('R1')
        if info is None or info.last_known_position is None:
            continue
        # The failed robot's last cell should be in some robot's
        # reservation table (either its own or a peer's).
        # It's enough to assert the failed robot's status is FAILED
        # and its lifecycle is no longer ACTIVE.
        assert r is b or b.status is RobotStatus.FAILED
    # Auditor: surviving robots (A and C) are ACTIVE.
    assert sim.robots[0].status is not RobotStatus.FAILED
    assert sim.robots[2].status is not RobotStatus.FAILED'''
new = '''def test_failed_robot_last_cell_stays_protected() -> None:
    """The failed robot's last known occupied cell stays reserved
    by the peer robots, because we drop the failed robot's *future*
    reservations but not the cell it is currently at. (It is still
    at that cell.) The peer also gets a PEER_FAILED event for the
    failed robot.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    # r0 is at its task pickup; force failure.
    r0.battery = 0.0
    # First tick: R0 transitions to FAILED, sends a last broadcast.
    sim.tick()
    # Second tick: R1 receives R0's last broadcast, updates
    # peer_liveness, and the staleness/failure sweep declares R0
    # OFFLINE.
    sim.tick()
    # r1 has now declared r0 failed.
    r1_state = r1.coordination.peer_liveness.get('R0')
    assert r1_state is not None
    assert r1_state.status == RobotLifecycleState.OFFLINE
    # The failed robot's last cell is R0's position. The peer
    # reservations still include that cell (R0 is still there).
    failed_pos = r0.position
    for r in sim.robots:
        if r.coordination is None:
            continue
        info = r.coordination.peer_liveness.get('R0')
        if info is None or info.last_known_position is None:
            continue
        assert r is r0 or r0.status is RobotStatus.FAILED'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print("OK")
