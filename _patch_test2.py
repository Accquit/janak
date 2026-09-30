import pathlib
p = pathlib.Path('sih26123/tests/test_phase2c.py')
text = p.read_text(encoding='utf-8')

# Patch the robot failure with task test.
old = '''def test_robot_fails_with_task_returns_to_pool() -> None:
    """A robot that fails while carrying a task must return the task
    to PENDING and emit TASK_RETURNED_TO_POOL.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    t0 = r0.current_task
    assert t0 is not None and t0.status is TaskStatus.IN_PROGRESS
    # Force failure mid-task.
    r0.battery = 0.0
    sim.tick()
    assert r0.status is RobotStatus.FAILED
    assert r0.current_task is None
    assert t0.status is TaskStatus.PENDING
    assert t0.assigned_robot is None
    returned = sim.event_log.of_type(EventType.TASK_RETURNED_TO_POOL)
    assert any(e.data['task_id'] == t0.task_id
               and e.data['previous_robot'] == 'R0' for e in returned)'''
new = '''def test_robot_fails_with_task_returns_to_pool() -> None:
    """A robot that fails while carrying a task must return the task
    to PENDING and emit TASK_RETURNED_TO_POOL.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    t0 = r0.current_task
    assert t0 is not None
    # Step once so the task transitions from ASSIGNED to IN_PROGRESS.
    sim.tick()
    assert t0.status is TaskStatus.IN_PROGRESS, (
        f"task should be IN_PROGRESS, was {t0.status}"
    )
    # Force failure mid-task.
    r0.battery = 0.0
    sim.tick()
    assert r0.status is RobotStatus.FAILED
    assert r0.current_task is None
    assert t0.status is TaskStatus.PENDING
    assert t0.assigned_robot is None
    returned = sim.event_log.of_type(EventType.TASK_RETURNED_TO_POOL)
    assert any(e.data['task_id'] == t0.task_id
               and e.data['previous_robot'] == 'R0' for e in returned)'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print('OK')
