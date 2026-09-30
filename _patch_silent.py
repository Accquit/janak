import pathlib
p = pathlib.Path('sih26123/simulation/simulator.py')
text = p.read_text(encoding='utf-8')
old = '''        if self.message_bus is not None:
            for robot in coordination_robots:
                task_id = robot.current_task.task_id if robot.current_task else None
                task_priority = (
                    int(getattr(robot.current_task, "priority", 0) or 0)
                    if robot.current_task is not None
                    else 0
                )
                published = robot.coordination.publish_state('''
new = '''        if self.message_bus is not None:
            for robot in coordination_robots:
                # Phase 2C test hook: silenced robots must NOT broadcast.
                if getattr(robot, "silenced", False):
                    continue
                task_id = robot.current_task.task_id if robot.current_task else None
                task_priority = (
                    int(getattr(robot.current_task, "priority", 0) or 0)
                    if robot.current_task is not None
                    else 0
                )
                published = robot.coordination.publish_state('''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print("OK")
