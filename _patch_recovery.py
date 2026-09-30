import pathlib
p = pathlib.Path('sih26123/tests/test_phase2c.py')
text = p.read_text(encoding='utf-8')

# Fix the recovery test to also clear the drop mask.
old = '''def test_communication_recovery_resumes_active() -> None:
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
new = '''def test_communication_recovery_resumes_active() -> None:
    """Once a SAFE_HALT robot receives a fresh peer heartbeat, it
    must transition back to ACTIVE and a recovery heartbeat event is
    emitted.
    """
    sim = _make_sim(num_robots=3)
    r0 = sim.robots[0]
    for _ in range(2):
        sim.tick()
    # Silence R0 and drop messages TO R0 (full isolation).
    r0.silenced = True
    sim.message_bus.drop_messages_to("R0", drop=True)
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    # Re-enable communication: stop silencing and clear the drop
    # mask so R0 can hear peers again.
    r0.silenced = False
    sim.message_bus.clear_drop_mask()
    sim.tick()
    assert r0.coordination.lifecycle.state == RobotLifecycleState.ACTIVE
    recovery_hb = sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
    assert any(e.data.get('tag') == 'recovered' for e in recovery_hb)'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print("OK")
