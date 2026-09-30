import pathlib
p = pathlib.Path('sih26123/tests/test_phase2c.py')
text = p.read_text(encoding='utf-8')
old = """    # Re-enable communication: stop silencing and clear the drop
    # mask so R0 can hear peers again.
    r0.silenced = False
    sim.message_bus.clear_drop_mask()
    sim.tick()
    assert r0.coordination.lifecycle.state == RobotLifecycleState.ACTIVE
    recovery_hb = sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
    assert any(e.data.get('tag') == 'recovered' for e in recovery_hb)"""
new = """    # Re-enable communication: stop silencing and clear the drop
    # mask so R0 can hear peers again.
    r0.silenced = False
    sim.message_bus.clear_drop_mask()
    # Two ticks so R0 can both (a) receive a fresh broadcast at
    # step 3 and (b) update its last_peer_heartbeat_tick and exit
    # SAFE_HALT at step 4d.
    sim.tick()
    sim.tick()
    assert r0.coordination.lifecycle.state == RobotLifecycleState.ACTIVE
    recovery_hb = sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
    assert any(e.data.get('tag') == 'recovered' for e in recovery_hb)"""
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print("OK")
