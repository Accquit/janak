import pathlib
p = pathlib.Path('sih26123/tests/test_phase2c.py')
text = p.read_text(encoding='utf-8')

# Patch the peer-failure test to use silencing instead of message drop.
old = '''def test_peer_failure_at_threshold() -> None:
    """A peer that stops broadcasting for >=PEER_FAILURE_TIMEOUT
    ticks is declared OFFLINE by the surviving peers; ``PEER_FAILED``
    is emitted.
    """
    bus = InProcessMessageBus()
    sim = _make_sim(num_robots=3, bus=bus)
    r0, r1, r2 = sim.robots
    for _ in range(2):
        sim.tick()
    bus.drop_messages_to("R0", drop=True)
    for _ in range(PEER_FAILURE_TIMEOUT + 1):
        sim.tick()

    failed = sim.event_log.of_type(EventType.PEER_FAILED)
    # R1 and R2 should each have observed R0 as failed.
    r1_failed = [e for e in failed
                 if e.data['robot_id'] == 'R1'
                 and e.data['peer_id'] == 'R0']
    r2_failed = [e for e in failed
                 if e.data['robot_id'] == 'R2'
                 and e.data['peer_id'] == 'R0']
    assert r1_failed, "R1 should observe R0 as failed"
    assert r2_failed, "R2 should observe R0 as failed"
    # R1 and R2 should have OFFLINE in their local peer_liveness for R0.
    for r in (r1, r2):
        info = r.coordination.peer_liveness.get("R0")
        assert info is not None
        assert info.status == RobotLifecycleState.OFFLINE'''
new = '''def test_peer_failure_at_threshold() -> None:
    """A peer that stops broadcasting for >=PEER_FAILURE_TIMEOUT
    ticks is declared OFFLINE by the surviving peers; ``PEER_FAILED``
    is emitted.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    for _ in range(2):
        sim.tick()
    # Silencing R0 prevents its broadcasts from reaching the bus. R1
    # and R2 will not see R0's heartbeat any more.
    r0.silenced = True
    for _ in range(PEER_FAILURE_TIMEOUT + 1):
        sim.tick()

    failed = sim.event_log.of_type(EventType.PEER_FAILED)
    # R1 and R2 should each have observed R0 as failed.
    r1_failed = [e for e in failed
                 if e.data['robot_id'] == 'R1'
                 and e.data['peer_id'] == 'R0']
    r2_failed = [e for e in failed
                 if e.data['robot_id'] == 'R2'
                 and e.data['peer_id'] == 'R0']
    assert r1_failed, "R1 should observe R0 as failed"
    assert r2_failed, "R2 should observe R0 as failed"
    # R1 and R2 should have OFFLINE in their local peer_liveness for R0.
    for r in (r1, r2):
        info = r.coordination.peer_liveness.get("R0")
        assert info is not None
        assert info.status == RobotLifecycleState.OFFLINE'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)

# Patch the staleness test.
old = '''def test_stale_peer_after_threshold() -> None:
    """A peer that stops broadcasting for >STALE_THRESHOLD ticks
    has its forward reservations cleared and the simulator emits
    ``PEER_STALE`` + ``RESERVATIONS_CLEARED``.
    """
    bus = InProcessMessageBus()
    sim = _make_sim(num_robots=3, bus=bus)
    r0, r1, r2 = sim.robots

    # Run a few ticks so all robots know each other.
    for _ in range(2):
        sim.tick()
    # Now drop messages to r0 so it stops getting peer heartbeats.
    bus.drop_messages_to("R0", drop=True)
    for _ in range(STALE_THRESHOLD + 1):
        sim.tick()

    stale_events = sim.event_log.of_type(EventType.PEER_STALE)
    # R0 should have at least one PEER_STALE event.
    r0_stale = [e for e in stale_events if e.data['robot_id'] == 'R0']
    assert r0_stale, "R0 should observe its peers as stale"
    cleared_events = sim.event_log.of_type(EventType.RESERVATIONS_CLEARED)
    assert cleared_events, "R0 must clear its peers' reservations on stale"'''
new = '''def test_stale_peer_after_threshold() -> None:
    """A peer that stops broadcasting for >STALE_THRESHOLD ticks
    has its forward reservations cleared and the simulator emits
    ``PEER_STALE`` + ``RESERVATIONS_CLEARED``.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots

    # Run a few ticks so all robots know each other.
    for _ in range(2):
        sim.tick()
    # Silencing R0 prevents its broadcasts from reaching the bus
    # from this point on. R1 and R2 will not see R0's heartbeat.
    r0.silenced = True
    for _ in range(STALE_THRESHOLD + 1):
        sim.tick()

    stale_events = sim.event_log.of_type(EventType.PEER_STALE)
    # R1 and R2 should each have at least one PEER_STALE event for R0.
    r1_stale = [e for e in stale_events
                if e.data['robot_id'] == 'R1'
                and e.data['peer_id'] == 'R0']
    r2_stale = [e for e in stale_events
                if e.data['robot_id'] == 'R2'
                and e.data['peer_id'] == 'R0']
    assert r1_stale, "R1 should observe R0 as stale"
    assert r2_stale, "R2 should observe R0 as stale"
    cleared_events = sim.event_log.of_type(EventType.RESERVATIONS_CLEARED)
    assert cleared_events, "R1/R2 must clear R0's reservations on stale"'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)

# Patch the heartbeat-recovery test.
old = '''def test_heartbeat_recovery_before_failure() -> None:
    """If a peer drops out for a short while but recovers within
    STALE_THRESHOLD, no PEER_FAILED is emitted.
    """
    bus = InProcessMessageBus()
    sim = _make_sim(num_robots=3, bus=bus)
    for _ in range(2):
        sim.tick()
    bus.drop_messages_to("R0", drop=True)
    for _ in range(STALE_THRESHOLD):
        sim.tick()
    bus.drop_messages_to("R0", drop=False)
    for _ in range(STALE_THRESHOLD + 2):
        sim.tick()
    failed = sim.event_log.of_type(EventType.PEER_FAILED)
    assert not failed, "Recovery within STALE_THRESHOLD should NOT declare failure"'''
new = '''def test_heartbeat_recovery_before_failure() -> None:
    """If a peer drops out for a short while but recovers within
    STALE_THRESHOLD, no PEER_FAILED is emitted.
    """
    sim = _make_sim(num_robots=3)
    for _ in range(2):
        sim.tick()
    r0, _, _ = sim.robots
    r0.silenced = True
    for _ in range(STALE_THRESHOLD):
        sim.tick()
    r0.silenced = False
    for _ in range(STALE_THRESHOLD + 2):
        sim.tick()
    failed = sim.event_log.of_type(EventType.PEER_FAILED)
    assert not failed, "Recovery within STALE_THRESHOLD should NOT declare failure"'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)

# Patch the safe-halt tests.
old = '''def test_communication_isolated_robot_enters_safe_halt() -> None:
    """A robot that fails to receive a valid peer heartbeat for
    >=SELF_ISOLATION_TIMEOUT ticks must enter SAFE_HALT and stop
    using stale peer beliefs.
    """
    bus = InProcessMessageBus()
    sim = _make_sim(num_robots=3, bus=bus)
    r0, r1, r2 = sim.robots
    for _ in range(2):
        sim.tick()
    bus.drop_messages_to("R0", drop=True)
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    assert r0.coordination.lifecycle.state == RobotLifecycleState.SAFE_HALT
    safe_halt = sim.event_log.of_type(EventType.ROBOT_SAFE_HALT)
    assert any(e.data['robot_id'] == 'R0' for e in safe_halt)'''
new = '''def test_communication_isolated_robot_enters_safe_halt() -> None:
    """A robot that fails to receive a valid peer heartbeat for
    >=SELF_ISOLATION_TIMEOUT ticks must enter SAFE_HALT and stop
    using stale peer beliefs.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    for _ in range(2):
        sim.tick()
    r0.silenced = True
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    assert r0.coordination.lifecycle.state == RobotLifecycleState.SAFE_HALT
    safe_halt = sim.event_log.of_type(EventType.ROBOT_SAFE_HALT)
    assert any(e.data['robot_id'] == 'R0' for e in safe_halt)'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)

old = '''def test_communication_recovery_resumes_active() -> None:
    """Once a SAFE_HALT robot receives a fresh peer heartbeat, it
    must transition back to ACTIVE and a recovery heartbeat event is
    emitted.
    """
    bus = InProcessMessageBus()
    sim = _make_sim(num_robots=3, bus=bus)
    r0 = sim.robots[0]
    for _ in range(2):
        sim.tick()
    bus.drop_messages_to("R0", drop=True)
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    bus.drop_messages_to("R0", drop=False)
    sim.tick()
    assert r0.coordination.lifecycle.state == RobotLifecycleState.ACTIVE
    recovery_hb = sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
    assert any(e.data.get('tag') == 'recovered' for e in recovery_hb)'''
new = '''def test_communication_recovery_resumes_active() -> None:
    """Once a SAFE_HALT robot receives a fresh peer heartbeat, it
    must transition back to ACTIVE and a recovery heartbeat event is
    emitted.
    """
    sim = _make_sim(num_robots=3)
    r0 = sim.robots[0]
    for _ in range(2):
        sim.tick()
    r0.silenced = True
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    r0.silenced = False
    sim.tick()
    assert r0.coordination.lifecycle.state == RobotLifecycleState.ACTIVE
    recovery_hb = sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
    assert any(e.data.get('tag') == 'recovered' for e in recovery_hb)'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)

# Patch the safe-halt doesn't broadcast test.
old = '''def test_safe_halt_robot_does_not_broadcast() -> None:
    """A SAFE_HALT robot stops sending heartbeats and broadcasts.
    Other robots therefore mark it as stale after STALE_THRESHOLD.
    """
    bus = InProcessMessageBus()
    sim = _make_sim(num_robots=3, bus=bus)
    r0 = sim.robots[0]
    for _ in range(2):
        sim.tick()
    bus.drop_messages_to("R0", drop=True)
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    # R0 in safe halt. We re-enable its broadcasts. Now R0 must send
    # heartbeats and other robots must register them.
    bus.drop_messages_to("R0", drop=False)
    sim.tick()
    hb_after_recovery = [
        ev for ev in sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
        if ev.data['sender_id'] == 'R0' and ev.tick >= SELF_ISOLATION_TIMEOUT + 4
    ]
    assert hb_after_recovery'''
new = '''def test_safe_halt_robot_does_not_broadcast() -> None:
    """A SAFE_HALT robot stops sending heartbeats and broadcasts.
    Other robots therefore mark it as stale after STALE_THRESHOLD.
    """
    sim = _make_sim(num_robots=3)
    r0 = sim.robots[0]
    for _ in range(2):
        sim.tick()
    r0.silenced = True
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    # R0 in safe halt. We re-enable its broadcasts. Now R0 must send
    # heartbeats and other robots must register them.
    r0.silenced = False
    sim.tick()
    hb_after_recovery = [
        ev for ev in sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
        if ev.data['sender_id'] == 'R0' and ev.tick >= SELF_ISOLATION_TIMEOUT + 4
    ]
    assert hb_after_recovery'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)

p.write_text(text, encoding='utf-8')
print("OK")
