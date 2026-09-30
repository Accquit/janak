"""Tests for the Robot agent."""

from __future__ import annotations

from typing import List

import pytest

from sih26123.perception.simulated_sensor import SimulatedPerception, WorldView
from sih26123.simulation.events import EventType
from sih26123.simulation.robot import Robot, RobotStatus
from sih26123.simulation.warehouse import Warehouse


@pytest.fixture
def robot_on_open_grid() -> Robot:
    warehouse = Warehouse(width=5, height=5)
    view = WorldView(warehouse=warehouse, robots=[], tick=0)
    perception = SimulatedPerception(world_view=view, sensor_range=3, self_id="R1")
    return Robot(robot_id="R1", start_position=(0, 0), perception=perception, battery_capacity=100.0)


def test_initial_state(robot_on_open_grid: Robot) -> None:
    r = robot_on_open_grid
    assert r.position == (0, 0)
    assert r.battery == 100.0
    assert r.status is RobotStatus.IDLE
    assert r.current_task is None
    assert r.path == []


def test_set_path_moves_robot_to_moving(robot_on_open_grid: Robot) -> None:
    r = robot_on_open_grid
    r.set_path([(0, 0), (1, 0), (2, 0)], (2, 0))
    assert r.status is RobotStatus.MOVING
    assert r.destination == (2, 0)
    assert r.has_path


def test_step_advances_along_path(robot_on_open_grid: Robot) -> None:
    r = robot_on_open_grid
    r.set_path([(0, 0), (1, 0), (2, 0)], (2, 0))
    events = r.step(tick=0, world=None)
    assert r.position == (1, 0)
    assert any(e.event_type is EventType.ROBOT_MOVED for e in events)


def test_battery_decreases_with_movement(robot_on_open_grid: Robot) -> None:
    r = robot_on_open_grid
    r.set_path([(0, 0), (1, 0), (2, 0)], (2, 0))
    before = r.battery
    r.step(tick=0, world=None)
    after_one = r.battery
    assert after_one < before
    # Move cost should be applied.
    r.step(tick=1, world=None)
    assert r.battery < after_one


def test_arrival_at_destination(robot_on_open_grid: Robot) -> None:
    r = robot_on_open_grid
    r.set_path([(0, 0), (1, 0)], (1, 0))
    r.step(tick=0, world=None)
    assert r.position == (1, 0)
    assert r.status is RobotStatus.IDLE
    assert not r.has_path


def test_battery_exhaustion_marks_failed() -> None:
    warehouse = Warehouse(width=10, height=10)
    view = WorldView(warehouse=warehouse, robots=[], tick=0)
    perception = SimulatedPerception(world_view=view, sensor_range=3, self_id="R2")
    r = Robot(robot_id="R2", start_position=(0, 0), perception=perception, battery_capacity=2.5, move_cost=1.0)
    r.set_path([(0, 0), (1, 0), (2, 0), (3, 0)], (3, 0))
    # Tick until exhausted.
    exhausted = False
    for i in range(10):
        events = r.step(tick=i, world=None)
        if r.status is RobotStatus.FAILED:
            exhausted = True
            assert any(e.event_type is EventType.ROBOT_FAILED for e in events)
            break
    assert exhausted
    assert r.battery == 0.0


def test_robot_waits_when_next_cell_has_dynamic_obstacle() -> None:
    warehouse = Warehouse(width=5, height=5)
    warehouse.add_dynamic_obstacle(1, 0)
    view = WorldView(warehouse=warehouse, robots=[], tick=0)
    perception = SimulatedPerception(world_view=view, sensor_range=3, self_id="R3")
    r = Robot(robot_id="R3", start_position=(0, 0), perception=perception, battery_capacity=100.0)
    r.set_path([(0, 0), (1, 0), (2, 0)], (2, 0))

    # First step: should detect the obstacle and transition to WAITING.
    events = r.step(tick=0, world=warehouse)
    assert r.status is RobotStatus.WAITING
    assert r.position == (0, 0)
    types = [e.event_type for e in events]
    assert EventType.ROBOT_WAITING in types
    assert EventType.OBSTACLE_DETECTED in types

    # Remove the obstacle: next step the robot should resume MOVING.
    warehouse.remove_dynamic_obstacle(1, 0)
    events = r.step(tick=1, world=warehouse)
    assert r.status is RobotStatus.MOVING
    assert r.position == (1, 0)


def test_clear_path_idles_robot(robot_on_open_grid: Robot) -> None:
    r = robot_on_open_grid
    r.set_path([(0, 0), (1, 0)], (1, 0))
    r.clear_path()
    assert r.status is RobotStatus.IDLE
    assert r.path == []


def test_set_path_with_empty_path() -> None:
    warehouse = Warehouse(width=3, height=3)
    view = WorldView(warehouse=warehouse, robots=[], tick=0)
    perception = SimulatedPerception(world_view=view, sensor_range=3, self_id="R4")
    r = Robot(robot_id="R4", start_position=(0, 0), perception=perception)
    r.set_path([], (0, 0))
    assert r.status is RobotStatus.IDLE


def test_idle_drain(robot_on_open_grid: Robot) -> None:
    r = robot_on_open_grid
    before = r.battery
    r.step(tick=0, world=None)  # IDLE
    assert r.battery < before


def test_set_path_strips_current_position() -> None:
    warehouse = Warehouse(width=3, height=3)
    view = WorldView(warehouse=warehouse, robots=[], tick=0)
    perception = SimulatedPerception(world_view=view, sensor_range=3, self_id="R5")
    r = Robot(robot_id="R5", start_position=(0, 0), perception=perception)
    # A planner that returns a path including the start cell is accepted;
    # the robot strips the current position so path[0] is the next cell.
    r.set_path([(0, 0), (1, 0), (2, 0)], (2, 0))
    assert r.path[0] == (1, 0)


def test_set_path_with_already_at_destination_idles() -> None:
    warehouse = Warehouse(width=3, height=3)
    view = WorldView(warehouse=warehouse, robots=[], tick=0)
    perception = SimulatedPerception(world_view=view, sensor_range=3, self_id="R5b")
    r = Robot(robot_id="R5b", start_position=(1, 1), perception=perception)
    r.set_path([(1, 1)], (1, 1))
    assert r.status is RobotStatus.IDLE
    assert r.path == []


def test_robot_motion_is_step_by_step() -> None:
    warehouse = Warehouse(width=10, height=10)
    view = WorldView(warehouse=warehouse, robots=[], tick=0)
    perception = SimulatedPerception(world_view=view, sensor_range=3, self_id="R6")
    r = Robot(robot_id="R6", start_position=(0, 0), perception=perception, battery_capacity=100.0)
    r.set_path([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], (4, 0))
    # It must never teleport. After each step the robot moves by exactly
    # one axis (4-connected).
    last = r.position
    for i in range(5):
        r.step(tick=i, world=warehouse)
        if r.status is RobotStatus.MOVING:
            dx = abs(r.position[0] - last[0])
            dy = abs(r.position[1] - last[1])
            assert dx + dy == 1
        last = r.position
    assert r.position == (4, 0)