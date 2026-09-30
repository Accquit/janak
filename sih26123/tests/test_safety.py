"""P1 safety regressions for movement, reservations, and recovery."""

from __future__ import annotations

from sih26123.coordination.agent import RobotCoordination
from sih26123.coordination.communication import (
    InProcessMessageBus,
    Message,
    MessageType,
    PeerStateRepository,
    RobotState,
)
from sih26123.coordination.conflicts import ConflictDetector
from sih26123.coordination.lifecycle import (
    PeerLivenessTracker,
    RobotLifecycle,
    RobotLifecycleState,
)
from sih26123.coordination.reservations import ReservationTable
from sih26123.coordination.resilience import STALE_THRESHOLD
from sih26123.perception.simulated_sensor import SimulatedPerception, WorldView
from sih26123.simulation.events import EventType
from sih26123.simulation.robot import Robot, RobotStatus
from sih26123.simulation.simulator import Simulator
from sih26123.simulation.warehouse import CellType, Warehouse


def _robot(
    robot_id: str,
    position: tuple[int, int],
    warehouse: Warehouse,
    bus: InProcessMessageBus | None = None,
    coordinated: bool = False,
) -> Robot:
    perception = SimulatedPerception(
        world_view=WorldView(warehouse=warehouse, robots=[], tick=0),
        sensor_range=5,
        self_id=robot_id,
    )
    coordination = None
    if coordinated:
        if bus is None:
            bus = InProcessMessageBus()
        reservations = ReservationTable(owner_id=robot_id, horizon=8)
        coordination = RobotCoordination(
            robot_id=robot_id,
            bus=bus,
            peer_states=PeerStateRepository(self_id=robot_id),
            reservations=reservations,
            conflict_detector=ConflictDetector(
                owner_id=robot_id,
                reservations=reservations,
                lookahead_horizon=8,
            ),
            lifecycle=RobotLifecycle(),
            peer_liveness=PeerLivenessTracker(),
        )
        # These tests target the final safety boundary, independently of
        # the P2 negotiation policy.
        coordination.predict_conflicts = lambda current_tick: []
        coordination.run_negotiation = lambda **kwargs: []
    return Robot(
        robot_id=robot_id,
        start_position=position,
        perception=perception,
        battery_capacity=200.0,
        coordination=coordination,
    )


def test_swap_moves_are_blocked() -> None:
    warehouse = Warehouse(width=5, height=5)
    a = _robot("A", (1, 2), warehouse)
    b = _robot("B", (2, 2), warehouse)
    a.set_path([(2, 2)], (2, 2))
    b.set_path([(1, 2)], (1, 2))
    sim = Simulator(warehouse, [a, b], [])

    sim.tick()

    assert a.position == (1, 2)
    assert b.position == (2, 2)
    assert sim.event_log.of_type(EventType.COLLISION_DETECTED) == []


def test_failed_robot_cell_remains_physically_occupied() -> None:
    warehouse = Warehouse(width=5, height=5)
    a = _robot("A", (1, 1), warehouse)
    failed = _robot("F", (2, 1), warehouse)
    failed.status = RobotStatus.FAILED
    a.set_path([(2, 1)], (2, 1))
    sim = Simulator(warehouse, [a, failed], [])

    sim.tick()

    assert a.position == (1, 1)
    assert failed.position == (2, 1)
    assert sim.event_log.of_type(EventType.COLLISION_DETECTED) == []


def test_coordinated_move_requires_own_arrival_reservation() -> None:
    warehouse = Warehouse(width=5, height=5)
    bus = InProcessMessageBus()
    robot = _robot("A", (1, 1), warehouse, bus, coordinated=True)
    robot.set_path([(2, 1)], (2, 1))
    robot.coordination.update_own_reservations = lambda **kwargs: None
    sim = Simulator(warehouse, [robot], [], message_bus=bus)

    sim.tick()

    assert robot.position == (1, 1)
    waits = sim.event_log.of_type(EventType.ROBOT_WAITING)
    assert any(event.data.get("safety_reason") == "missing_own_reservation" for event in waits)
    direct_events = robot.step(1, warehouse)
    assert robot.position == (1, 1)
    assert any(
        event.data.get("safety_reason") == "missing_own_reservation"
        for event in direct_events
    )


def test_peer_reservation_blocks_coordinated_move() -> None:
    warehouse = Warehouse(width=5, height=5)
    bus = InProcessMessageBus()
    robot = _robot("A", (1, 1), warehouse, bus, coordinated=True)
    robot.set_path([(2, 1)], (2, 1))
    robot.coordination.reservations.add((2, 1), 1, "B")
    sim = Simulator(warehouse, [robot], [], message_bus=bus)

    sim.tick()

    assert robot.position == (1, 1)
    waits = sim.event_log.of_type(EventType.ROBOT_WAITING)
    assert any(event.data.get("safety_reason") == "peer_reservation" for event in waits)


def test_reverse_edge_reservation_blocks_coordinated_move() -> None:
    warehouse = Warehouse(width=5, height=5)
    bus = InProcessMessageBus()
    robot = _robot("A", (1, 1), warehouse, bus, coordinated=True)
    robot.set_path([(2, 1)], (2, 1))
    # B's reservation describes (2,1)->(1,1) during this tick.
    robot.coordination.reservations.add((2, 1), 0, "B")
    robot.coordination.reservations.add((1, 1), 1, "B")
    sim = Simulator(warehouse, [robot], [], message_bus=bus)

    sim.tick()

    assert robot.position == (1, 1)
    waits = sim.event_log.of_type(EventType.ROBOT_WAITING)
    assert any(
        event.data.get("safety_reason") == "peer_reverse_edge_reservation"
        for event in waits
    )


def test_stale_peer_last_cell_is_reserved_and_blocks_entry() -> None:
    warehouse = Warehouse(width=5, height=5)
    bus = InProcessMessageBus()
    robot = _robot("A", (1, 1), warehouse, bus, coordinated=True)
    robot.set_path([(2, 1)], (2, 1))
    robot.coordination.peer_liveness.record_heartbeat(
        peer_id="B", tick=0, sequence=1, position=(2, 1)
    )
    sim = Simulator(warehouse, [robot], [], message_bus=bus)
    sim.tick_count = 15

    sim.tick()

    assert robot.position == (1, 1)
    assert robot.coordination.peer_liveness.get("B").status is RobotLifecycleState.STALE
    assert robot.coordination.reservations.has_robot((2, 1), 16, "B")
    waits = sim.event_log.of_type(EventType.ROBOT_WAITING)
    assert any(event.data.get("safety_reason") == "peer_reservation" for event in waits)
    robot.coordination.mark_peer_stale(24)
    assert robot.coordination.reservations.has_robot((2, 1), 32, "B")


def test_failed_peer_clears_forward_path_but_keeps_last_cell_reserved() -> None:
    warehouse = Warehouse(width=5, height=5)
    robot = _robot("A", (1, 1), warehouse, coordinated=True)
    coordination = robot.coordination
    coordination.peer_liveness.record_heartbeat(
        peer_id="B", tick=0, sequence=1, position=(2, 1)
    )
    coordination.reservations.update_peer_from_state(
        peer_id="B",
        peer_position=(2, 1),
        peer_path=[(3, 1)],
        peer_timestamp=0,
        horizon=8,
    )

    coordination.declare_peer_failed("B", current_tick=5)

    assert coordination.peer_liveness.get("B").status is RobotLifecycleState.OFFLINE
    assert coordination.reservations.has_robot((2, 1), 13, "B")
    assert not coordination.reservations.has_robot((3, 1), 6, "B")
    coordination.mark_peer_stale(current_tick=20)
    assert coordination.reservations.has_robot((2, 1), 28, "B")


def test_safe_halt_blocks_direct_step_and_recovery_resumes() -> None:
    warehouse = Warehouse(width=5, height=5)
    bus = InProcessMessageBus()
    robot = _robot("A", (1, 1), warehouse, bus, coordinated=True)
    robot.set_path([(2, 1)], (2, 1))
    robot.coordination.lifecycle.state = RobotLifecycleState.SAFE_HALT

    robot.step(0, warehouse)

    assert robot.position == (1, 1)
    assert robot.status is RobotStatus.WAITING

    robot.coordination.lifecycle.state = RobotLifecycleState.ACTIVE
    robot.coordination.update_own_reservations(
        current_pos=robot.position,
        path=robot.remaining_path(),
        current_tick=1,
    )
    robot.step(1, warehouse)
    assert robot.position == (2, 1)


def test_space_time_wait_is_preserved_as_a_tick() -> None:
    warehouse = Warehouse(width=5, height=5)
    robot = _robot("A", (1, 1), warehouse)
    robot.set_path([(1, 1), (1, 1), (2, 1)], (2, 1))

    robot.step(0, warehouse)

    assert robot.position == (1, 1)
    assert robot.next_path_cell() == (2, 1)
    robot.step(1, warehouse)
    assert robot.position == (2, 1)


def test_liveness_rejects_stale_sequence_at_same_tick() -> None:
    tracker = PeerLivenessTracker()
    assert tracker.record_heartbeat("B", 5, 2, (2, 1))
    assert not tracker.record_heartbeat("B", 5, 1, (3, 1))
    assert tracker.get("B").last_known_position == (2, 1)
    assert tracker.record_heartbeat("B", 5, 3, (4, 1))
    assert tracker.get("B").last_known_position == (4, 1)


def test_heartbeat_without_known_occupied_cell_does_not_refresh_liveness() -> None:
    warehouse = Warehouse(width=5, height=5)
    bus = InProcessMessageBus(["A", "B"])
    robot = _robot("A", (1, 1), warehouse, bus, coordinated=True)
    bus.publish(Message(
        sender_id="B",
        timestamp=1,
        msg_type=MessageType.HEARTBEAT,
        payload={"peer_id": "B", "tick": 1, "sequence": 1, "position": None},
    ))
    bus.deliver()

    robot.coordination.process_inbox(current_tick=2)

    assert robot.coordination.last_peer_heartbeat_tick == -1
    assert robot.coordination.peer_liveness.get("B") is None


def test_old_peer_state_cannot_refresh_safe_halt_recovery() -> None:
    warehouse = Warehouse(width=5, height=5)
    bus = InProcessMessageBus(["A", "B"])
    robot = _robot("A", (1, 1), warehouse, bus, coordinated=True)
    old_state = RobotState(
        robot_id="B",
        timestamp=0,
        position=(2, 1),
        velocity=(0, 0),
        battery=100.0,
        task_id=None,
        status="idle",
        planned_path=(),
        sequence=1,
    )
    bus.publish(Message(
        sender_id="B",
        timestamp=0,
        msg_type=MessageType.ROBOT_STATE,
        payload=old_state,
    ))
    bus.deliver()

    robot.coordination.process_inbox(current_tick=STALE_THRESHOLD)

    assert robot.coordination.last_peer_heartbeat_tick == -1
    robot.coordination.mark_peer_stale(STALE_THRESHOLD)
    assert robot.coordination.peer_liveness.get("B").status is RobotLifecycleState.STALE
    assert robot.coordination.reservations.has_robot(
        (2, 1), STALE_THRESHOLD + 8, "B",
    )


def test_static_wall_is_checked_at_movement_boundary() -> None:
    grid = [[CellType.FREE] * 5 for _ in range(5)]
    grid[1][2] = CellType.OBSTACLE
    warehouse = Warehouse.with_grid(grid)
    robot = _robot("A", (1, 1), warehouse)
    robot.set_path([(2, 1)], (2, 1))

    robot.step(0, warehouse)

    assert robot.position == (1, 1)
    assert robot.status is RobotStatus.WAITING


def test_non_adjacent_path_step_cannot_teleport_robot() -> None:
    warehouse = Warehouse(width=5, height=5)
    robot = _robot("A", (1, 1), warehouse)
    robot.set_path([(3, 1)], (3, 1))

    events = robot.step(0, warehouse)

    assert robot.position == (1, 1)
    assert any(event.data.get("safety_reason") == "invalid_transition" for event in events)
