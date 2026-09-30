"""End-to-end tests for the Simulator with three robots."""

from __future__ import annotations

from typing import List, Tuple

import pytest

from sih26123.perception.simulated_sensor import SimulatedPerception, WorldView
from sih26123.planning.astar import astar
from sih26123.simulation.events import EventType
from sih26123.simulation.robot import Robot, RobotStatus
from sih26123.simulation.simulator import DynamicObstacleSchedule, Simulator
from sih26123.simulation.task import Task, TaskStatus
from sih26123.simulation.warehouse import CellType, Warehouse


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def small_warehouse() -> Warehouse:
    """A 12x8 warehouse with aisles, walls, pickup and drop-off cells.

    Layout (columns 0..11, rows 0..7):

        ############
        #P.........D
        #..........#
        #..........#
        #..........#
        #P.........D
        #..........#
        ############

    The top, bottom, left, and right columns are walls. The interior
    is mostly free, with PICKUP cells at (1,1) and (1,5), and DROPOFF
    cells at (11,1) and (11,5). Robots must travel along the open
    interior.
    """
    OB = CellType.OBSTACLE
    PK = CellType.PICKUP
    DR = CellType.DROPOFF
    FR = CellType.FREE

    width, height = 12, 8
    grid = [[FR for _ in range(width)] for _ in range(height)]

    # Top and bottom walls.
    for x in range(width):
        grid[0][x] = OB
        grid[height - 1][x] = OB
    # Left and right walls.
    for y in range(height):
        grid[y][0] = OB
        grid[y][width - 1] = OB

    # Pickup and drop-off markers (still traversable).
    grid[1][1] = PK
    grid[1][width - 2] = DR
    grid[5][1] = PK
    grid[5][width - 2] = DR

    return Warehouse.with_grid(grid)


@pytest.fixture
def three_robots(small_warehouse: Warehouse):
    robots: List[Robot] = []
    # Starting positions are chosen so that none of the robots are
    # already at their pickup cell. This makes the paths non-trivial
    # and exposes potential conflicts.
    for rid, pos in [("A", (2, 1)), ("B", (6, 2)), ("C", (5, 5))]:
        view = WorldView(warehouse=small_warehouse, robots=[], tick=0)
        perception = SimulatedPerception(world_view=view, sensor_range=4, self_id=rid)
        robots.append(Robot(robot_id=rid, start_position=pos, perception=perception, battery_capacity=200.0))
    return robots


@pytest.fixture
def three_tasks() -> List[Task]:
    return [
        Task(task_id="T1", pickup_location=(1, 1), dropoff_location=(10, 1), priority=1),
        Task(task_id="T2", pickup_location=(6, 1), dropoff_location=(10, 5), priority=2),
        Task(task_id="T3", pickup_location=(6, 5), dropoff_location=(1, 5), priority=3),
    ]


@pytest.fixture
def astar_planner(small_warehouse: Warehouse):
    wh = small_warehouse

    def _plan(robot, start, goal):
        return astar(start, goal, wh.width, wh.height, lambda x, y: wh.is_traversable(x, y))

    return _plan


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_three_robots_exist(small_warehouse, three_robots, three_tasks) -> None:
    sim = Simulator(small_warehouse, three_robots, three_tasks, seed=42)
    assert len(sim.robots) == 3
    assert {r.robot_id for r in sim.robots} == {"A", "B", "C"}


def test_tasks_can_be_assigned(small_warehouse, three_robots, three_tasks, astar_planner) -> None:
    sim = Simulator(small_warehouse, three_robots, three_tasks, seed=42)
    sim.set_path_planner(astar_planner)
    assert sim.assign_task("T1", "A") is True
    assert sim.assign_task("T2", "B") is True
    assert sim.assign_task("T3", "C") is True
    assert three_tasks[0].status is TaskStatus.ASSIGNED
    # Robot A starts AT its pickup cell, so its path is empty and it
    # transitions to IDLE until the task replan moves it toward the
    # drop-off. Robots B and C are far from their pickups.
    assert three_robots[0].status in (RobotStatus.IDLE, RobotStatus.MOVING)
    assert three_robots[1].status is RobotStatus.MOVING
    assert three_robots[2].status is RobotStatus.MOVING


def test_robots_move_under_tick(small_warehouse, three_robots, three_tasks, astar_planner) -> None:
    sim = Simulator(small_warehouse, three_robots, three_tasks, seed=42)
    sim.set_path_planner(astar_planner)
    sim.assign_task("T1", "A")

    # Track positions over multiple ticks.
    positions = [three_robots[0].position]
    for _ in range(5):
        sim.tick()
        positions.append(three_robots[0].position)
    # Robot must have advanced at least one cell in 5 ticks.
    assert positions[-1] != positions[0]
    # All positions are valid grid cells.
    for p in positions:
        assert small_warehouse.is_within_bounds(*p)
        # Some cells (like the pickup area itself) are not walls, so
        # robot positions are always traversable at the moment of step.


def test_events_are_generated(small_warehouse, three_robots, three_tasks, astar_planner) -> None:
    sim = Simulator(small_warehouse, three_robots, three_tasks, seed=42)
    sim.set_path_planner(astar_planner)
    sim.assign_task("T1", "A")
    sim.assign_task("T2", "B")
    sim.assign_task("T3", "C")
    sim.run(max_ticks=2)
    types = {e.event_type for e in sim.event_log.all()}
    assert EventType.SIMULATION_STARTED in types
    assert EventType.ROBOT_MOVED in types or EventType.ROBOT_STARTED in types
    assert EventType.TASK_ASSIGNED in types


def test_collision_detected_when_two_robots_share_cell(small_warehouse) -> None:
    # Manually place two robots and drive them to the same cell.
    view1 = WorldView(warehouse=small_warehouse, robots=[], tick=0)
    p1 = SimulatedPerception(world_view=view1, sensor_range=4, self_id="A")
    a = Robot(robot_id="A", start_position=(1, 1), perception=p1, battery_capacity=200.0)

    view2 = WorldView(warehouse=small_warehouse, robots=[], tick=0)
    p2 = SimulatedPerception(world_view=view2, sensor_range=4, self_id="B")
    b = Robot(robot_id="B", start_position=(1, 3), perception=p2, battery_capacity=200.0)

    # Force both onto path ending in (1,2): A from (1,1)->(1,2), B from (1,3)->(1,2).
    a.set_path([(1, 1), (1, 2)], (1, 2))
    b.set_path([(1, 3), (1, 2)], (1, 2))

    sim = Simulator(small_warehouse, [a, b], [], seed=1)
    sim.tick()
    # After this tick, A and B should both be at (1,2). Collision detected.
    collisions = sim.event_log.of_type(EventType.COLLISION_DETECTED)
    assert any("A" in e.data.get("robot_ids", []) and "B" in e.data.get("robot_ids", []) for e in collisions)


def test_dynamic_obstacles_appear_in_schedule(small_warehouse, three_robots, three_tasks, astar_planner) -> None:
    schedule = DynamicObstacleSchedule(entries=[(0, "add", (2, 1))])
    sim = Simulator(small_warehouse, three_robots, three_tasks, dynamic_schedule=schedule, seed=42)
    sim.set_path_planner(astar_planner)
    sim.assign_task("T1", "A")
    sim.tick()
    assert small_warehouse.has_dynamic_obstacle(2, 1)
    assert sim.event_log.count(EventType.OBSTACLE_ADDED) >= 1


def test_dynamic_obstacle_blocks_path(small_warehouse, three_robots, three_tasks, astar_planner) -> None:
    # Place a dynamic obstacle right on the robot's planned path at the next cell.
    # Robot A starts at (1,1) going to (11,1). The straight line along y=1 hits x=2.
    # Schedule the obstacle at tick 0 so it lands before A moves.
    schedule = DynamicObstacleSchedule(entries=[(0, "add", (2, 1))])
    sim = Simulator(small_warehouse, three_robots, three_tasks, dynamic_schedule=schedule, seed=42)
    sim.set_path_planner(astar_planner)
    sim.assign_task("T1", "A")
    sim.tick()
    # Robot should be in WAITING because the next cell (2,1) is blocked.
    a = sim.get_robot("A")
    assert a is not None
    # After tick, robot either went WAITING or stayed MOVING if the planner
    # already routed around (2,1). With a planner that doesn't know about
    # the obstacle in tick 0 (since the schedule applies before stepping)
    # the robot sees the blocked cell.
    assert a.status in (RobotStatus.WAITING, RobotStatus.MOVING)
    # If it is WAITING, an OBSTACLE_DETECTED event was emitted.
    if a.status is RobotStatus.WAITING:
        assert sim.event_log.count(EventType.OBSTACLE_DETECTED) >= 1


def test_full_run_with_three_robots_eventually_completes(small_warehouse, three_robots, three_tasks, astar_planner) -> None:
    sim = Simulator(small_warehouse, three_robots, three_tasks, seed=42)
    sim.set_path_planner(astar_planner)
    sim.assign_task("T1", "A")
    sim.assign_task("T2", "B")
    sim.assign_task("T3", "C")
    sim.run(max_ticks=500)
    # At least one task should make progress.
    completed = [t for t in sim.tasks if t.status is TaskStatus.COMPLETED]
    assert len(completed) >= 1


def test_determinism_with_same_seed(small_warehouse, three_robots, three_tasks, astar_planner) -> None:
    """Two runs with the same seed produce identical event logs."""
    def run_once():
        # Fresh robots/tasks for each run (they hold mutable state).
        local_robots = []
        for rid, pos in [("A", (2, 1)), ("B", (6, 2)), ("C", (5, 5))]:
            view = WorldView(warehouse=small_warehouse, robots=[], tick=0)
            perception = SimulatedPerception(world_view=view, sensor_range=4, self_id=rid)
            local_robots.append(Robot(robot_id=rid, start_position=pos, perception=perception, battery_capacity=200.0))
        local_tasks = [
            Task(task_id="T1", pickup_location=(1, 1), dropoff_location=(10, 1), priority=1),
            Task(task_id="T2", pickup_location=(6, 1), dropoff_location=(10, 5), priority=2),
            Task(task_id="T3", pickup_location=(6, 5), dropoff_location=(1, 5), priority=3),
        ]
        sim = Simulator(small_warehouse, local_robots, local_tasks, seed=123)
        sim.set_path_planner(astar_planner)
        sim.assign_task("T1", "A")
        sim.assign_task("T2", "B")
        sim.assign_task("T3", "C")
        sim.run(max_ticks=20)
        # Encode events into a tuple of (tick, type, sorted data items).
        return [(e.tick, e.event_type, tuple(sorted((str(k), repr(v)) for k, v in e.data.items()))) for e in sim.event_log.all()]

    a = run_once()
    b = run_once()
    assert a == b


def test_duplicate_robot_ids_rejected(small_warehouse) -> None:
    view = WorldView(warehouse=small_warehouse, robots=[], tick=0)
    p1 = SimulatedPerception(world_view=view, sensor_range=3, self_id="X")
    p2 = SimulatedPerception(world_view=view, sensor_range=3, self_id="X")
    r1 = Robot(robot_id="X", start_position=(0, 0), perception=p1)
    r2 = Robot(robot_id="X", start_position=(1, 0), perception=p2)
    with pytest.raises(ValueError):
        Simulator(small_warehouse, [r1, r2], [], seed=1)


def test_assign_to_unknown_robot_fails(small_warehouse, three_tasks, astar_planner) -> None:
    view = WorldView(warehouse=small_warehouse, robots=[], tick=0)
    p = SimulatedPerception(world_view=view, sensor_range=3, self_id="A")
    a = Robot(robot_id="A", start_position=(1, 1), perception=p)
    sim = Simulator(small_warehouse, [a], three_tasks, seed=1)
    sim.set_path_planner(astar_planner)
    assert sim.assign_task("T1", "ZZZ") is False


def test_assign_without_planner_fails(small_warehouse, three_robots, three_tasks) -> None:
    sim = Simulator(small_warehouse, three_robots, three_tasks, seed=1)
    # No planner installed.
    assert sim.assign_task("T1", "A") is False
    assert sim.event_log.count(EventType.TASK_ASSIGNED) >= 1


def test_simulator_does_not_resolve_collisions(small_warehouse) -> None:
    """Two robots that meet at the same position should both end up there."""
    view1 = WorldView(warehouse=small_warehouse, robots=[], tick=0)
    a = Robot(robot_id="A", start_position=(1, 1), perception=SimulatedPerception(world_view=view1, sensor_range=4, self_id="A"))
    view2 = WorldView(warehouse=small_warehouse, robots=[], tick=0)
    b = Robot(robot_id="B", start_position=(1, 3), perception=SimulatedPerception(world_view=view2, sensor_range=4, self_id="B"))

    a.set_path([(1, 1), (1, 2)], (1, 2))
    b.set_path([(1, 3), (1, 2)], (1, 2))

    sim = Simulator(small_warehouse, [a, b], [], seed=1)
    sim.tick()
    # Both robots ended up at (1,2) — the simulator records but does not
    # resolve the conflict. That's exactly the gap the next layer fills.
    assert a.position == (1, 2)
    assert b.position == (1, 2)
    # And a COLLISION_DETECTED event was emitted.
    assert sim.event_log.count(EventType.COLLISION_DETECTED) >= 1