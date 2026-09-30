import pathlib
p = pathlib.Path('sih26123/simulation/simulator.py')
text = p.read_text(encoding='utf-8')
old = """    def _update_tasks(self) -> List[Event]:
        \"\"\"Drive task lifecycle from observed robot positions.

        Robots make their own phase transitions; here we just observe
        the resulting state and reflect it on the task object.

        Phase 2C: when a robot enters FAILED while holding an active
        task, that task is returned to PENDING (returned to the task
        pool) so it can be reassigned to a surviving robot. The robot's
        ``current_task`` is cleared without teleporting it.
        \"\"\"
        events: List[Event] = []
        for robot in self.robots:
            task = robot.current_task
            if task is None:
                continue
            # Phase 2C: failed-robot task return-to-pool.
            if robot.status is RobotStatus.FAILED:
                events.extend(self._return_failed_task_to_pool(robot, task))
                continue"""
new = """    def _update_tasks(self) -> List[Event]:
        \"\"\"Drive task lifecycle from observed robot positions.

        Robots make their own phase transitions; here we just observe
        the resulting state and reflect it on the task object.

        Phase 2C: when a robot enters FAILED while holding an active
        task, that task is returned to PENDING (returned to the task
        pool) so it can be reassigned to a surviving robot. The robot's
        ``current_task`` is cleared without teleporting it. We also
        mark the robot's local lifecycle as FAILED so other peers
        observe it through broadcasts and the staleness/failure
        path stays consistent.
        \"\"\"
        events: List[Event] = []
        for robot in self.robots:
            task = robot.current_task
            if task is None:
                continue
            # Phase 2C: failed-robot task return-to-pool.
            if robot.status is RobotStatus.FAILED:
                if robot.coordination is not None and \\
                        robot.coordination.lifecycle.state != \\
                        RobotLifecycleState.FAILED:
                    robot.coordination.lifecycle.state = \\
                        RobotLifecycleState.FAILED
                    robot.coordination.lifecycle.failed_since_tick = \\
                        self.tick_count
                events.extend(self._return_failed_task_to_pool(robot, task))
                continue"""
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print("OK")
