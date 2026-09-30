import pathlib
p = pathlib.Path('sih26123/simulation/simulator.py')
text = p.read_text(encoding='utf-8')
old = '''                if published:
                    if not getattr(robot, "silenced", False):
                        events.append(Event(
                            self.tick_count,
                            EventType.MESSAGE_BROADCAST,
                            sender_id=robot.robot_id,
                            msg_type="robot_state",
                        ))
                        # Phase 2C: also emit the semantic heartbeat
                        # event so failure / staleness tests can subscribe
                        # to a dedicated channel rather than parsing
                        # broadcast payloads.
                        events.append(Event(
                            self.tick_count,
                            EventType.ROBOT_HEARTBEAT,
                            sender_id=robot.robot_id,
                            sequence=robot.coordination.sequence,
                        ))'''
new = '''                if published and not getattr(robot, "silenced", False):
                    events.append(Event(
                        self.tick_count,
                        EventType.MESSAGE_BROADCAST,
                        sender_id=robot.robot_id,
                        msg_type="robot_state",
                    ))
                    # Phase 2C: also emit the semantic heartbeat
                    # event so failure / staleness tests can subscribe
                    # to a dedicated channel rather than parsing
                    # broadcast payloads.
                    events.append(Event(
                        self.tick_count,
                        EventType.ROBOT_HEARTBEAT,
                        sender_id=robot.robot_id,
                        sequence=robot.coordination.sequence,
                    ))'''
assert old in text, "anchor missing"
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print("OK")
