"""Shared pytest fixtures for the foundation tests."""

from __future__ import annotations

import pytest

from sih26123.perception.simulated_sensor import SimulatedPerception, WorldView
from sih26123.planning.astar import astar
from sih26123.simulation.robot import Robot
from sih26123.simulation.task import Task
from sih26123.simulation.warehouse import CellType, Warehouse


@pytest.fixture
def empty_warehouse() -> Warehouse:
    """A small all-free warehouse for unit tests."""
    return Warehouse(width=5, height=5)


@pytest.fixture
def wall_warehouse() -> Warehouse:
    """A warehouse with a vertical wall in the middle.

    Layout (5x5):
        .....
        .XXX.
        .....
        .XXX.
        .....
    The wall blocks the middle column at rows 1 and 3.
    """
    grid = [
        [CellType.FREE] * 5,
        [CellType.FREE, CellType.FREE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.FREE],
        [CellType.FREE] * 5,
        [CellType.FREE, CellType.FREE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.FREE],
        [CellType.FREE] * 5,
    ]
    return Warehouse.with_grid(grid)


@pytest.fixture
def planner_for_warehouse():
    """Return a path-planner function bound to a warehouse.

    Usage in tests::

        def test_x(empty_warehouse, planner_for_warehouse):
            plan = planner_for_warehouse(empty_warehouse)
            path = plan((0, 0), (4, 4))
    """

    def _make(warehouse: Warehouse):
        wh = warehouse

        def _plan(start, goal):
            return astar(start, goal, wh.width, wh.height, lambda x, y: wh.is_traversable(x, y))

        return _plan

    return _make


@pytest.fixture
def make_robot():
    """Factory for robots with a stub perception that ignores the world."""

    def _make(robot_id: str, start_position, warehouse: Warehouse, sensor_range: int = 3):
        view = WorldView(warehouse=warehouse, robots=[], tick=0)
        perception = SimulatedPerception(world_view=view, sensor_range=sensor_range, self_id=robot_id)
        return Robot(robot_id=robot_id, start_position=start_position, perception=perception)

    return _make


@pytest.fixture
def make_task():
    def _make(task_id: str, pickup, dropoff, priority: int = 0, deadline=None):
        return Task(task_id=task_id, pickup_location=pickup, dropoff_location=dropoff, priority=priority, deadline=deadline)

    return _make