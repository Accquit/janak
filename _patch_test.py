import pathlib
p = pathlib.Path('sih26123/tests/test_phase2c.py')
text = p.read_text(encoding='utf-8')
old = '''def test_peer_failure_at_threshold() -> None:
    """A peer that stops broadcasting for >=PEER_FAILURE_TIMEOUT
    ticks is declared OFFLINE; ``PEER_FAILED`` is emitted.
    """
    bus = InProcessMessageBus()
    sim = _make_sim(num_robots=3, bus=bus)
    for _ in range(2):
        sim.tick()
    bus.drop_messages_to("R0", drop=True)
    for _ in range(PEER_FAILURE_TIMEOUT + 1):
        sim.tick()

    failed = sim.event_log.of_type(EventType.PEER_FAILED)
    r0_failed = [e for e in failed if e.data['robot_id'] == 'R0']
    assert r0_failed, "R0 should observe its peers as failed"
    # R0 should have OFFLINE in its local peer_liveness for R1 and R2.
    r0_coord = r0.coordination
    for pid in ("R1", "R2"):
        info = r0_coord.peer_liveness.get(pid)
        assert info is not None
        assert info.status == RobotLifecycleState.OFFLINE'''
new = '''def test_peer_failure_at_threshold() -> None:
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
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print("OK")
