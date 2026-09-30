"""Tests for the A* path planner."""

from __future__ import annotations

from typing import List, Tuple

import pytest

from sih26123.planning.astar import astar, manhattan
from sih26123.simulation.warehouse import CellType, Warehouse

CellCoord = Tuple[int, int]


def test_manhattan_distance_is_correct() -> None:
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((2, 5), (2, 5)) == 0
    assert manhattan((1, 1), (5, 1)) == 4


def test_straight_path() -> None:
    """A* finds a straight line on an empty grid."""
    warehouse = Warehouse(width=10, height=10)
    is_free = lambda x, y: warehouse.is_traversable(x, y)
    path = astar((0, 0), (4, 0), warehouse.width, warehouse.height, is_free)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 0)
    # Manhattan distance is the optimal length.
    assert len(path) - 1 == 4
    # Every consecutive cell differs by exactly one axis (4-connected).
    for a, b in zip(path, path[1:]):
        assert (abs(a[0] - b[0]) + abs(a[1] - b[1])) == 1


def test_path_around_a_wall() -> None:
    """A* routes around an obstacle."""
    # Wall is at y=1 but x=0 and x=5 are open; creates a U-shaped barrier.
    grid = [
        [CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE],
        [CellType.FREE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.FREE],
        [CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE],
    ]
    warehouse = Warehouse.with_grid(grid)
    is_free = lambda x, y: warehouse.is_traversable(x, y)

    # Start and goal separated by a partial wall.
    path = astar((0, 0), (5, 2), warehouse.width, warehouse.height, is_free)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (5, 2)
    # No cell in the path is a wall.
    for (x, y) in path:
        assert warehouse.is_traversable(x, y)
    # Length should be > 5 (Manhattan) because of the detour through (0,1) or (5,1).
    assert len(path) - 1 >= 5 + 1


def test_maze_like_path() -> None:
    """A* solves a simple corridor maze."""
    # 7x5 grid, walls form an "S" curve.
    grid = [
        [CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE],
        [CellType.FREE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.FREE],
        [CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE],
        [CellType.FREE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.OBSTACLE, CellType.FREE],
        [CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE, CellType.FREE],
    ]
    warehouse = Warehouse.with_grid(grid)
    is_free = lambda x, y: warehouse.is_traversable(x, y)

    path = astar((0, 0), (6, 4), warehouse.width, warehouse.height, is_free)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (6, 4)
    for (x, y) in path:
        assert warehouse.is_traversable(x, y)


def test_unreachable_goal_returns_none() -> None:
    """Goal surrounded by walls yields None, not a hang."""
    grid = [
        [CellType.FREE, CellType.FREE, CellType.FREE],
        [CellType.FREE, CellType.OBSTACLE, CellType.FREE],
        [CellType.FREE, CellType.FREE, CellType.OBSTACLE],
        [CellType.FREE, CellType.FREE, CellType.FREE],
    ]
    warehouse = Warehouse.with_grid(grid)
    is_free = lambda x, y: warehouse.is_traversable(x, y)

    # Goal is isolated inside a 1x1 box of obstacles.
    # Build a tiny island by carving a 1x1 pit with walls around it.
    grid2 = [
        [CellType.FREE, CellType.OBSTACLE, CellType.FREE],
        [CellType.OBSTACLE, CellType.FREE, CellType.OBSTACLE],
        [CellType.FREE, CellType.OBSTACLE, CellType.FREE],
    ]
    warehouse2 = Warehouse.with_grid(grid2)
    is_free2 = lambda x, y: warehouse2.is_traversable(x, y)

    # From (0,0) we cannot reach (1,1).
    path = astar((0, 0), (1, 1), warehouse2.width, warehouse2.height, is_free2)
    assert path is None


def test_unreachable_goal_in_real_grid() -> None:
    """A blocked cell cannot be entered."""
    grid = [
        [CellType.FREE, CellType.OBSTACLE, CellType.FREE],
        [CellType.OBSTACLE, CellType.FREE, CellType.OBSTACLE],
        [CellType.FREE, CellType.OBSTACLE, CellType.FREE],
    ]
    warehouse = Warehouse.with_grid(grid)
    is_free = lambda x, y: warehouse.is_traversable(x, y)

    path = astar((0, 0), (1, 1), warehouse.width, warehouse.height, is_free)
    assert path is None


def test_start_equals_goal_returns_single_cell() -> None:
    warehouse = Warehouse(width=3, height=3)
    path = astar((1, 1), (1, 1), warehouse.width, warehouse.height, warehouse.is_traversable)
    assert path == [(1, 1)]


def test_astar_terminates_on_empty_grid() -> None:
    """Large empty grid should still finish quickly."""
    warehouse = Warehouse(width=200, height=200)
    path = astar((0, 0), (199, 199), warehouse.width, warehouse.height, warehouse.is_traversable)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (199, 199)
    # Optimal length is 2*(199) = 398 edges.
    assert len(path) - 1 == 398


def test_start_out_of_bounds_raises() -> None:
    warehouse = Warehouse(width=3, height=3)
    with pytest.raises(ValueError):
        astar((-1, 0), (1, 1), warehouse.width, warehouse.height, warehouse.is_traversable)


def test_unwalkable_start_returns_none() -> None:
    grid = [
        [CellType.OBSTACLE, CellType.FREE],
        [CellType.FREE, CellType.FREE],
    ]
    warehouse = Warehouse.with_grid(grid)
    path = astar((0, 0), (1, 1), warehouse.width, warehouse.height, warehouse.is_traversable)
    assert path is None