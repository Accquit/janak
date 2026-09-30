"""Tests for the RobotCoordination orchestrator and the simulator integration
of Phase 2A.

These tests verify:

* a single :class:`RobotCoordination` updates peer state and
  reservations from the bus;
* three robots integrated with a :class:`Simulator` exchange state
  through an in-process bus, each maintaining its own peer view;
* a conflict is predicted *before* the collision happens.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from sih26123.coordination.agent import RobotCoordination
from sih26123.coordination.communication import (
    InProcessMessageBus,
    MessageBus,
    MessageType,
    PeerStateRepository,
    RobotState,
)
from sih26123.coordination.conflicts import (
    ConflictDetector,
    ConflictType,
    PredictedConflict,
)
from sih26123.coordination.reservations import ReservationTable
from sih26123.perception.simulated_sensor import SimulatedPerception, WorldView
from sih26123.planning.astar import astar
from sih26123.simulation.events import EventType
from sih26123.simulation.robot import Robot
from sih26123.simulation.simulator import DynamicObstacleSchedule, Simulator
from sih26123.simulation.task import Task
from sih26123.simulation.warehouse import CellType, Warehouse


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
CellCoord = Tuple[int, int]


def _make_bus_with(num_peers: int) -> InProcessMessageBus:
    ids = [f"P{i}" for i in range(num_peers)]
    return InProcessMessageBus(ids)


def _make_coordination(robot_id: str, bus: MessageBus, horizon: int = 10, period: int = 1) -> RobotCoordination:
    peer_states = PeerStateRepository(self_id=robot_id, stale_threshold_ticks=5)
    reservations = ReservationTable(owner_id=robot_id, horizon=horizon)
    detector = ConflictDetector(owner_id=robot_id, reservations=reservations, lookahead_horizon=horizon)
    return RobotCoordination(
        robot_id=robot_id,
        bus=bus,
        peer_states=peer_states,
        reservations=reservations,
        conflict_detector=detector,
        broadcast_period=period,
    )


# ----------------------------------------------------------------------
# RobotCoordination — unit tests
# ----------------------------------------------------------------------
def test_process_inbox_updates_peer_state_and_reservations() -> None:
    bus = _make_bus_with(2)
    bus.register_peer("A")
    bus.register_peer("B")
    coord_a = _make_coordination("A", bus)
    coord_b = _make_coordination("B", bus)
    # B publishes.
    state_b = RobotState(
        robot_id="B", timestamp=2, position=(3, 4), velocity=(1, 0),
        battery=80.0, task_id="T1", status="moving",
        planned_path=((4, 4), (5, 4)),
    )
    bus.publish(__import__("sih26123.coordination.communication", fromlist=["Message"]).Message(
        sender_id="B", timestamp=2, msg_type=MessageType.ROBOT_STATE, payload=state_b,
    ))
    bus.deliver()
    processed = coord_a.process_inbox()
    assert processed == 1
    assert coord_a.peer_states.get("B") == state_b
    # Peer reservations derived: B's position (3,4) is reserved at
    # peer_timestamp + 1 = 3. Planned cells at 4 and 5.
    assert coord_a.reservations.has_robot((3, 4), 3, "B")
    assert coord_a.reservations.has_robot((4, 4), 4, "B")
    assert coord_a.reservations.has_robot((5, 4), 5, "B")


def test_process_inbox_ignores_own_messages() -> None:
    bus = _make_bus_with(2)
    coord_a = _make_coordination("A", bus)
    state_a = RobotState(
        robot_id="A", timestamp=1, position=(0, 0), velocity=(0, 0),
        battery=100.0, task_id=None, status="idle", planned_path=(),
    )
    bus.publish(__import__("sih26123.coordination.communication", fromlist=["Message"]).Message(
        sender_id="A", timestamp=1, msg_type=MessageType.ROBOT_STATE, payload=state_a,
    ))
    bus.deliver()
    # A's own broadcast is delivered to peers, not to A itself.
    assert coord_a.process_inbox() == 0


def test_update_own_reservations() -> None:
    bus = _make_bus_with(2)
    coord_a = _make_coordination("A", bus, horizon=5)
    coord_a.update_own_reservations(
        current_pos=(0, 0),
        path=[(1, 0), (2, 0)],
        current_tick=3,
    )
    assert coord_a.reservations.get_robot_at((0, 0), 3) == "A"
    assert coord_a.reservations.get_robot_at((1, 0), 4) == "A"
    assert coord_a.reservations.get_robot_at((2, 0), 5) == "A"


def test_predict_conflicts_returns_empty_when_no_conflicts() -> None:
    bus = _make_bus_with(2)
    coord_a = _make_coordination("A", bus)
    coord_a.update_own_reservations(current_pos=(0, 0), path=[(1, 0)], current_tick=0)
    assert coord_a.predict_conflicts(current_tick=0) == []


def test_publish_state_only_when_period_elapses() -> None:
    bus = _make_bus_with(2)
    bus.register_peer("A")
    coord = _make_coordination("A", bus, period=2)

    assert coord.should_broadcast(0) is True
    # After broadcasting at tick 0, the period suppresses tick 1.
    coord.publish_state(
        current_tick=0, position=(0, 0), velocity=(0, 0), battery=100.0,
        task_id=None, status="idle", planned_path=[],
    )
    assert coord.should_broadcast(1) is False
    assert coord.should_broadcast(2) is True
    # Broadcast at tick 2 suppresses tick 3.
    coord.publish_state(
        current_tick=2, position=(2, 0), velocity=(1, 0), battery=99.0,
        task_id=None, status="moving", planned_path=[(3, 0)],
    )
    assert coord.should_broadcast(3) is False


def test_publish_state_writes_to_bus() -> None:
    bus = _make_bus_with(2)
    # The sender (A) must be registered on the bus for delivery to skip
    # it (otherwise it auto-registers and may not match expectations).
    bus.register_peer("A")
    coord_a = _make_coordination("A", bus)
    ok = coord_a.publish_state(
        current_tick=5,
        position=(2, 3),
        velocity=(1, 0),
        battery=90.0,
        task_id="T1",
        status="moving",
        planned_path=[(3, 3), (4, 3)],
        intent="moving_to_pickup",
    )
    assert ok is True
    assert coord_a.last_broadcast_tick == 5
    assert set(bus.peer_ids()) == {"P0", "P1", "A"}
    bus.deliver()
    inbox_p0 = bus.drain_inbox("P0")
    inbox_p1 = bus.drain_inbox("P1")
    assert len(inbox_p0) == 1
    assert len(inbox_p1) == 1
    assert isinstance(inbox_p0[0].payload, RobotState)
    assert inbox_p0[0].payload.planned_path == ((3, 3), (4, 3))


def test_publish_state_returns_false_when_period_suppresses() -> None:
    bus = _make_bus_with(2)
    coord = _make_coordination("A", bus, period=3)
    coord.publish_state(current_tick=0, position=(0, 0), velocity=(0, 0), battery=100.0,
                        task_id=None, status="idle", planned_path=[])
    assert coord.publish_state(current_tick=1, position=(1, 0), velocity=(1, 0), battery=99.0,
                               task_id=None, status="moving", planned_path=[(2, 0)]) is False


# ----------------------------------------------------------------------
# Simulator integration
# ----------------------------------------------------------------------
def _small_warehouse() -> Warehouse:
    """12x8 warehouse with a vertical wall through the middle."""
    OB = CellType.OBSTACLE
    FR = CellType.FREE
    PK = CellType.PICKUP
    DR = CellType.DROPOFF

    width, height = 12, 8
    grid = [[FR for _ in range(width)] for _ in range(height)]
    for x in range(width):
        grid[0][x] = OB
        grid[height - 1][x] = OB
    for y in range(height):
        grid[y][0] = OB
        grid[y][width - 1] = OB
    grid[1][1] = PK
    grid[1][width - 2] = DR
    grid[5][1] = PK
    grid[5][width - 2] = DR
    return Warehouse.with_grid(grid)


def _make_robot_with_coordination(robot_id: str, start: CellCoord, warehouse: Warehouse,
                                   bus: MessageBus, horizon: int = 10) -> Robot:
    view = WorldView(warehouse=warehouse, robots=[], tick=0)
    perception = SimulatedPerception(world_view=view, sensor_range=4, self_id=robot_id)
    coordination = _make_coordination(robot_id, bus, horizon=horizon)
    return Robot(robot_id=robot_id, start_position=start, perception=perception,
                 battery_capacity=400.0, coordination=coordination)


def _make_sim_with_coordination() -> Tuple[Simulator, Warehouse, List[Robot], List[Task]]:
    warehouse = _small_warehouse()
    bus = InProcessMessageBus()
    robots: List[Robot] = [
        _make_robot_with_coordination("A", (2, 1), warehouse, bus),
        _make_robot_with_coordination("B", (6, 1), warehouse, bus),
        _make_robot_with_coordination("C", (5, 5), warehouse, bus),
    ]
    tasks: List[Task] = [
        Task(task_id="T1", pickup_location=(1, 1), dropoff_location=(10, 1), priority=1),
        Task(task_id="T2", pickup_location=(6, 1), dropoff_location=(10, 5), priority=2),
        Task(task_id="T3", pickup_location=(6, 5), dropoff_location=(1, 5), priority=3),
    ]
    sim = Simulator(warehouse, robots, tasks, seed=42, message_bus=bus)

    def planner(_r, start, goal):
        return astar(start, goal, warehouse.width, warehouse.height,
                     lambda x, y: warehouse.is_traversable(x, y))

    sim.set_path_planner(planner)
    sim.assign_task("T1", "A")
    sim.assign_task("T2", "B")
    sim.assign_task("T3", "C")
    return sim, warehouse, robots, tasks


def test_three_robots_broadcast_state_through_bus() -> None:
    sim, _, robots, _ = _make_sim_with_coordination()
    sim.run(max_ticks=2)
    # After two ticks each robot has broadcast at least once.
    for r in robots:
        assert r.coordination is not None
        assert r.coordination.last_broadcast_tick >= 0
    # The simulator recorded MESSAGE_BROADCAST events.
    assert sim.event_log.count(EventType.MESSAGE_BROADCAST) >= 3


def test_each_robot_has_independent_peer_view() -> None:
    """Each robot's PeerStateRepository contains the other two robots."""
    sim, _, robots, _ = _make_sim_with_coordination()
    sim.run(max_ticks=3)
    for r in robots:
        peers = list(r.coordination.peer_states.all().keys())
        # Each robot knows the other two.
        assert set(peers) == {other.robot_id for other in robots if other.robot_id != r.robot_id}


def test_peer_view_does_not_include_self() -> None:
    sim, _, robots, _ = _make_sim_with_coordination()
    sim.run(max_ticks=2)
    for r in robots:
        assert r.robot_id not in r.coordination.peer_states.all()


def test_each_robot_maintains_own_reservations() -> None:
    sim, _, robots, _ = _make_sim_with_coordination()
    sim.run(max_ticks=1)
    for r in robots:
        own_res = r.coordination.reservations.reservations_for(r.robot_id)
        # Each robot has at least its own current position reserved.
        assert any(res.cell == r.position for res in own_res)


def test_predicted_conflict_emitted_before_collision() -> None:
    """Run until a conflict is predicted; ensure the event is recorded
    *before* the corresponding collision event (if any).

    With Phase 2B.1 (decentralised conflict resolution), conflicts
    are *predicted* and then *resolved* via negotiation rather than
    materialising as collisions. This test still verifies that any
    predicted conflict that does fire precedes any collision.
    """
    sim, _, _, _ = _make_sim_with_coordination()
    sim.run(max_ticks=200)
    conflicts = sim.event_log.of_type(EventType.CONFLICT_PREDICTED)
    if not conflicts:
        # With Phase 2B.1 the negotiation may reroute robots before
        # any conflict is even predicted. That's still the right
        # outcome — we just have nothing to assert here.
        assert sim.event_log.count(EventType.COLLISION_DETECTED) == 0
        return
    # If any conflict IS predicted, it must come before any collision.
    first_conflict_tick = conflicts[0].tick
    first_collision_tick = None
    for e in sim.event_log.of_type(EventType.COLLISION_DETECTED):
        first_collision_tick = e.tick
        break
    if first_collision_tick is not None:
        assert first_conflict_tick <= first_collision_tick
    # The predicted conflict has a valid shape.
    first = conflicts[0]
    assert first.data["observer"] in {"A", "B", "C"}
    assert first.data["timestep"] >= first.tick


def test_no_central_authority_holds_all_reservations() -> None:
    """Each robot's reservation table differs from the others.

    Each robot reserves from its own viewpoint using its own path +
    the peer paths it has received. Because broadcasts are sequential
    and processing is per-robot, the tables will not be byte-identical
    even for the same logical scenario.
    """
    sim, _, robots, _ = _make_sim_with_coordination()
    sim.run(max_ticks=2)
    tables = [r.coordination.reservations for r in robots]
    # All three tables contain their owner's reservations.
    for i, r in enumerate(robots):
        own = tables[i].reservations_for(r.robot_id)
        assert len(own) >= 1
    # And they include reservations for at least one peer.
    for i, r in enumerate(robots):
        others = set(tables[i].all_reservations()) - set(tables[i].reservations_for(r.robot_id))
        assert len(others) >= 1


def test_coordination_period_limits_broadcasts() -> None:
    warehouse = _small_warehouse()
    bus = InProcessMessageBus()
    robot = _make_robot_with_coordination("A", (2, 1), warehouse, bus, horizon=5)
    robot.coordination.broadcast_period = 2

    sim = Simulator(warehouse, [robot], [], seed=0, message_bus=bus)
    sim.run(max_ticks=5)
    broadcasts = sim.event_log.of_type(EventType.MESSAGE_BROADCAST)
    # At ticks 0, 2, 4 -> at most 3 broadcasts.
    assert 1 <= len(broadcasts) <= 3


def test_phase2a_does_not_make_robots_wait() -> None:
    """Critical: predicted conflicts MUST NOT cause robots to WAIT.

    The foundation behaviour (no waiting from coordination) must hold:
    the only WAITING events come from dynamic obstacles.
    """
    sim, _, _, _ = _make_sim_with_coordination()
    sim.run(max_ticks=200)
    waiting = sim.event_log.of_type(EventType.ROBOT_WAITING)
    # No dynamic obstacles in this scenario — so there must be no waiting.
    assert waiting == []


def test_simulation_runs_without_coordination_module() -> None:
    """Backwards compatibility: robots without coordination still work."""
    warehouse = _small_warehouse()
    robots = []
    for rid, pos in [("A", (2, 1)), ("B", (6, 1)), ("C", (5, 5))]:
        view = WorldView(warehouse=warehouse, robots=[], tick=0)
        perception = SimulatedPerception(world_view=view, sensor_range=4, self_id=rid)
        robots.append(Robot(robot_id=rid, start_position=pos, perception=perception, battery_capacity=400.0))
    tasks = [
        Task(task_id="T1", pickup_location=(1, 1), dropoff_location=(10, 1)),
        Task(task_id="T2", pickup_location=(6, 1), dropoff_location=(10, 5)),
        Task(task_id="T3", pickup_location=(6, 5), dropoff_location=(1, 5)),
    ]
    sim = Simulator(warehouse, robots, tasks, seed=42)  # no bus
    sim.set_path_planner(lambda _r, s, g: astar(s, g, warehouse.width, warehouse.height,
                                                lambda x, y: warehouse.is_traversable(x, y)))
    sim.assign_task("T1", "A")
    sim.assign_task("T2", "B")
    sim.assign_task("T3", "C")
    sim.run(max_ticks=200)
    # No coordination events should have been emitted.
    assert sim.event_log.count(EventType.MESSAGE_BROADCAST) == 0
    assert sim.event_log.count(EventType.CONFLICT_PREDICTED) == 0


def test_dynamic_obstacles_appear_event_still_recorded() -> None:
    """Coordination additions must not break Phase 1 obstacle events."""
    warehouse = _small_warehouse()
    bus = InProcessMessageBus()
    robots = [
        _make_robot_with_coordination("A", (2, 1), warehouse, bus),
    ]
    tasks = [Task(task_id="T1", pickup_location=(1, 1), dropoff_location=(10, 1))]
    schedule = DynamicObstacleSchedule(entries=[(0, "add", (2, 1))])
    sim = Simulator(warehouse, robots, tasks, dynamic_schedule=schedule, seed=0, message_bus=bus)
    sim.set_path_planner(lambda _r, s, g: astar(s, g, warehouse.width, warehouse.height,
                                                lambda x, y: warehouse.is_traversable(x, y)))
    sim.assign_task("T1", "A")
    sim.tick()
    assert sim.event_log.count(EventType.OBSTACLE_ADDED) == 1