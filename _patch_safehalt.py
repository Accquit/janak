import pathlib
p = pathlib.Path('sih26123/tests/test_phase2c.py')
text = p.read_text(encoding='utf-8')

# Fix the SAFE_HALT test to also drop messages to R0 (not just silence).
old = '''def test_communication_isolated_robot_enters_safe_halt() -> None:
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
new = '''def test_communication_isolated_robot_enters_safe_halt() -> None:
    """A robot that fails to receive a valid peer heartbeat for
    >=SELF_ISOLATION_TIMEOUT ticks must enter SAFE_HALT and stop
    using stale peer beliefs.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    for _ in range(2):
        sim.tick()
    # Silence R0 (no broadcasts) and drop messages TO R0
    # (R0 cannot hear peers anymore).
    r0.silenced = True
    sim.message_bus.drop_messages_to("R0", drop=True)
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    assert r0.coordination.lifecycle.state == RobotLifecycleState.SAFE_HALT
    safe_halt = sim.event_log.of_type(EventType.ROBOT_SAFE_HALT)
    assert any(e.data['robot_id'] == 'R0' for e in safe_halt)'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print("OK")
