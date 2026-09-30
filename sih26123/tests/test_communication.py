"""Tests for the Phase 2A communication layer."""

from __future__ import annotations

from typing import List

import pytest

from sih26123.coordination.communication import (
    InProcessMessageBus,
    Message,
    MessageBus,
    MessageType,
    PeerStateRepository,
    RobotState,
)


# ----------------------------------------------------------------------
# RobotState
# ----------------------------------------------------------------------
def test_robot_state_carries_all_required_fields() -> None:
    state = RobotState(
        robot_id="A",
        timestamp=10,
        position=(5, 7),
        velocity=(1, 0),
        battery=72.5,
        task_id="T1",
        status="moving",
        planned_path=((6, 7), (7, 7), (8, 7)),
        intent="moving_to_pickup",
    )
    assert state.robot_id == "A"
    assert state.timestamp == 10
    assert state.position == (5, 7)
    assert state.velocity == (1, 0)
    assert state.battery == 72.5
    assert state.task_id == "T1"
    assert state.status == "moving"
    assert state.planned_path == ((6, 7), (7, 7), (8, 7))
    assert state.intent == "moving_to_pickup"


# ----------------------------------------------------------------------
# InProcessMessageBus
# ----------------------------------------------------------------------
def test_message_bus_is_abstract() -> None:
    assert issubclass(InProcessMessageBus, MessageBus)


def test_register_peer_initialises_inbox() -> None:
    bus = InProcessMessageBus()
    bus.register_peer("A")
    bus.register_peer("B")
    assert set(bus.peer_ids()) == {"A", "B"}
    assert bus.inbox_size("A") == 0


def test_publish_queues_in_outbox() -> None:
    bus = InProcessMessageBus(["A", "B"])
    msg = Message(sender_id="A", timestamp=1, msg_type=MessageType.ROBOT_STATE, payload=RobotState(
        robot_id="A", timestamp=1, position=(0, 0), velocity=(0, 0), battery=100.0,
        task_id=None, status="idle", planned_path=(),
    ))
    bus.publish(msg)
    # Outbox is delivered only after ``deliver``.
    assert bus.inbox_size("B") == 0
    delivered = bus.deliver()
    assert delivered == 1
    assert bus.inbox_size("B") == 1
    assert bus.inbox_size("A") == 0  # sender is excluded


def test_drain_inbox_returns_and_clears() -> None:
    bus = InProcessMessageBus(["A", "B"])
    msg = Message(sender_id="A", timestamp=1, msg_type=MessageType.ROBOT_STATE, payload="payload")
    bus.publish(msg)
    bus.deliver()
    drained = bus.drain_inbox("B")
    assert len(drained) == 1
    assert drained[0].payload == "payload"
    assert bus.inbox_size("B") == 0


def test_publish_order_is_preserved_across_peers() -> None:
    """Each peer must receive messages in the publish order."""
    bus = InProcessMessageBus(["A", "B", "C"])
    for ts in range(5):
        bus.publish(Message(sender_id="A", timestamp=ts, msg_type=MessageType.ROBOT_STATE, payload=f"m{ts}"))
    bus.deliver()
    drained_b = bus.drain_inbox("B")
    drained_c = bus.drain_inbox("C")
    assert [m.payload for m in drained_b] == ["m0", "m1", "m2", "m3", "m4"]
    assert [m.payload for m in drained_c] == ["m0", "m1", "m2", "m3", "m4"]


def test_sender_does_not_receive_own_message() -> None:
    bus = InProcessMessageBus(["A", "B"])
    bus.publish(Message(sender_id="A", timestamp=0, msg_type=MessageType.ROBOT_STATE, payload="x"))
    bus.deliver()
    assert bus.inbox_size("A") == 0
    assert bus.inbox_size("B") == 1


def test_reset_clears_everything() -> None:
    bus = InProcessMessageBus(["A", "B"])
    bus.publish(Message(sender_id="A", timestamp=0, msg_type=MessageType.ROBOT_STATE, payload="x"))
    bus.deliver()
    bus.reset()
    assert bus.inbox_size("A") == 0
    assert bus.inbox_size("B") == 0


def test_unknown_sender_is_auto_registered() -> None:
    bus = InProcessMessageBus(["A"])
    # Auto-register on publish should not crash.
    bus.publish(Message(sender_id="Z", timestamp=0, msg_type=MessageType.ROBOT_STATE, payload="x"))
    bus.deliver()
    assert bus.inbox_size("A") == 1


def test_publish_before_register_does_not_raise() -> None:
    bus = InProcessMessageBus()
    bus.publish(Message(sender_id="A", timestamp=0, msg_type=MessageType.ROBOT_STATE, payload="x"))
    bus.deliver()  # No registered peers, so nothing is distributed.


# ----------------------------------------------------------------------
# PeerStateRepository
# ----------------------------------------------------------------------
def test_peer_state_repository_starts_empty() -> None:
    repo = PeerStateRepository(self_id="A")
    assert len(repo) == 0
    assert repo.get("B") is None
    assert "B" not in repo


def test_update_stores_state() -> None:
    repo = PeerStateRepository(self_id="A")
    s = RobotState(robot_id="B", timestamp=1, position=(0, 0), velocity=(0, 0), battery=100.0,
                   task_id=None, status="idle", planned_path=())
    assert repo.update(s) is True
    assert repo.get("B") == s
    assert "B" in repo


def test_update_ignores_stale_state() -> None:
    repo = PeerStateRepository(self_id="A")
    s1 = RobotState(robot_id="B", timestamp=5, position=(1, 0), velocity=(1, 0), battery=90.0,
                    task_id=None, status="moving", planned_path=((2, 0),))
    s2 = RobotState(robot_id="B", timestamp=5, position=(9, 9), velocity=(0, 0), battery=1.0,
                    task_id=None, status="idle", planned_path=())
    s3 = RobotState(robot_id="B", timestamp=4, position=(0, 0), velocity=(0, 0), battery=1.0,
                    task_id=None, status="idle", planned_path=())
    assert repo.update(s1) is True
    assert repo.update(s2) is False  # equal timestamp
    assert repo.update(s3) is False  # older timestamp
    assert repo.get("B").position == (1, 0)


def test_update_with_newer_timestamp_replaces() -> None:
    repo = PeerStateRepository(self_id="A")
    s_old = RobotState(robot_id="B", timestamp=1, position=(0, 0), velocity=(0, 0), battery=100.0,
                       task_id=None, status="idle", planned_path=())
    s_new = RobotState(robot_id="B", timestamp=2, position=(1, 0), velocity=(1, 0), battery=90.0,
                       task_id=None, status="moving", planned_path=())
    repo.update(s_old)
    assert repo.update(s_new) is True
    assert repo.get("B").timestamp == 2
    assert repo.get("B").position == (1, 0)


def test_is_fresh_with_stale_threshold() -> None:
    repo = PeerStateRepository(self_id="A", stale_threshold_ticks=2)
    s = RobotState(robot_id="B", timestamp=10, position=(0, 0), velocity=(0, 0), battery=100.0,
                   task_id=None, status="idle", planned_path=())
    repo.update(s)
    assert repo.is_fresh("B", 10)
    assert repo.is_fresh("B", 11)
    assert repo.is_fresh("B", 12)
    assert not repo.is_fresh("B", 13)
    assert not repo.is_fresh("B", 100)


def test_fresh_peers_listing() -> None:
    repo = PeerStateRepository(self_id="A", stale_threshold_ticks=1)
    for rid, ts in [("B", 5), ("D", 3)]:
        repo.update(RobotState(robot_id=rid, timestamp=ts, position=(0, 0), velocity=(0, 0),
                               battery=100.0, task_id=None, status="idle", planned_path=()))
    fresh = repo.fresh_peers(current_tick=5)
    assert set(fresh) == {"B"}  # D is at ts=3, stale at tick 5


def test_forget_removes_a_peer() -> None:
    repo = PeerStateRepository(self_id="A")
    repo.update(RobotState(robot_id="B", timestamp=1, position=(0, 0), velocity=(0, 0), battery=100.0,
                           task_id=None, status="idle", planned_path=()))
    repo.forget("B")
    assert repo.get("B") is None


def test_reset_clears_repo() -> None:
    repo = PeerStateRepository(self_id="A")
    repo.update(RobotState(robot_id="B", timestamp=1, position=(0, 0), velocity=(0, 0), battery=100.0,
                           task_id=None, status="idle", planned_path=()))
    repo.reset()
    assert len(repo) == 0