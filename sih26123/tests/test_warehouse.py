"""Tests for the warehouse grid model."""

from __future__ import annotations

import pytest

from sih26123.simulation.warehouse import CellType, Warehouse


def test_dimensions_and_default_grid(empty_warehouse: Warehouse) -> None:
    assert empty_warehouse.width == 5
    assert empty_warehouse.height == 5
    assert all(empty_warehouse.get_cell_type(x, y) is CellType.FREE for x in range(5) for y in range(5))


def test_invalid_dimensions_rejected() -> None:
    with pytest.raises(ValueError):
        Warehouse(width=0, height=5)
    with pytest.raises(ValueError):
        Warehouse(width=5, height=0)


def test_is_within_bounds() -> None:
    warehouse = Warehouse(width=3, height=3)
    assert warehouse.is_within_bounds(0, 0)
    assert warehouse.is_within_bounds(2, 2)
    assert not warehouse.is_within_bounds(-1, 0)
    assert not warehouse.is_within_bounds(3, 0)
    assert not warehouse.is_within_bounds(0, 3)


def test_is_traversable_treats_special_cells_as_open() -> None:
    grid = [
        [CellType.FREE, CellType.PICKUP, CellType.DROPOFF, CellType.CHARGING, CellType.OBSTACLE],
    ]
    warehouse = Warehouse.with_grid(grid)
    assert warehouse.is_traversable(0, 0)
    assert warehouse.is_traversable(1, 0)
    assert warehouse.is_traversable(2, 0)
    assert warehouse.is_traversable(3, 0)
    assert not warehouse.is_traversable(4, 0)


def test_static_obstacles_detected() -> None:
    grid = [
        [CellType.FREE, CellType.OBSTACLE],
        [CellType.FREE, CellType.FREE],
    ]
    warehouse = Warehouse.with_grid(grid)
    assert warehouse.has_static_obstacle(1, 0)
    assert not warehouse.has_static_obstacle(0, 0)


def test_dynamic_obstacles_added_and_removed() -> None:
    warehouse = Warehouse(width=3, height=3)
    assert not warehouse.has_dynamic_obstacle(1, 1)
    obs = warehouse.add_dynamic_obstacle(1, 1)
    assert warehouse.has_dynamic_obstacle(1, 1)
    assert warehouse.has_dynamic_obstacle(1, 1)
    # Adding a dynamic obstacle makes the cell non-traversable.
    assert not warehouse.is_traversable(1, 1)
    # Removing it restores traversability.
    assert warehouse.remove_dynamic_obstacle(1, 1)
    assert not warehouse.has_dynamic_obstacle(1, 1)
    assert warehouse.is_traversable(1, 1)


def test_dynamic_obstacle_cannot_be_placed_on_wall() -> None:
    grid = [
        [CellType.FREE, CellType.OBSTACLE],
        [CellType.FREE, CellType.FREE],
    ]
    warehouse = Warehouse.with_grid(grid)
    with pytest.raises(ValueError):
        warehouse.add_dynamic_obstacle(1, 0)


def test_dynamic_obstacle_out_of_bounds_rejected() -> None:
    warehouse = Warehouse(width=3, height=3)
    with pytest.raises(ValueError):
        warehouse.add_dynamic_obstacle(-1, 0)


def test_get_neighbors_returns_traversable_cells() -> None:
    grid = [
        [CellType.FREE, CellType.FREE, CellType.FREE],
        [CellType.FREE, CellType.OBSTACLE, CellType.FREE],
        [CellType.FREE, CellType.FREE, CellType.FREE],
    ]
    warehouse = Warehouse.with_grid(grid)
    nbrs = warehouse.get_neighbors(1, 1)
    # (1,1) is OBSTACLE so it is not traversable; choose a traversable one.
    nbrs = warehouse.get_neighbors(0, 1)
    assert (0, 0) in nbrs
    assert (1, 1) not in nbrs  # wall
    assert (0, 2) in nbrs
    assert (-1, 1) not in nbrs  # off-grid


def test_dynamic_obstacle_blocks_neighbors() -> None:
    warehouse = Warehouse(width=3, height=3)
    warehouse.add_dynamic_obstacle(2, 1)
    nbrs = warehouse.get_neighbors(1, 1)
    assert (2, 1) not in nbrs


def test_grid_shape_validation() -> None:
    with pytest.raises(ValueError):
        Warehouse(width=3, height=3, grid=[[CellType.FREE] * 2])


def test_get_dynamic_obstacles_returns_ids() -> None:
    warehouse = Warehouse(width=3, height=3)
    warehouse.add_dynamic_obstacle(0, 0, obstacle_id="obs-A")
    warehouse.add_dynamic_obstacle(1, 0, obstacle_id="obs-B")
    ids = {o.obstacle_id for o in warehouse.get_dynamic_obstacles()}
    assert ids == {"obs-A", "obs-B"}