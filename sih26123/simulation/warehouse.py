"""Grid-based warehouse world.

The warehouse is the shared environment that robots operate in. It is a
pure data structure: it does not run any logic or make decisions. It only
answers questions about the grid.

Coordinates
-----------
``(x, y)`` with ``x`` increasing to the right and ``y`` increasing
downward. ``grid[y][x]`` is the cell type. This matches the common
raster convention and makes rendering straightforward.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from .obstacle import Obstacle, ObstacleKind

CellCoord = Tuple[int, int]


class CellType(Enum):
    """Logical type of a warehouse grid cell."""

    FREE = 0
    OBSTACLE = 1
    PICKUP = 2
    DROPOFF = 3
    CHARGING = 4


# Cells that a robot may drive through.
_TRAVERSABLE_TYPES = frozenset({CellType.FREE, CellType.PICKUP, CellType.DROPOFF, CellType.CHARGING})


class Warehouse:
    """A grid-based warehouse world.

    Parameters
        ``grid``:
            Optional 2D sequence of cell rows (row-major: ``grid[y][x]``).
            If omitted an all-free grid of the requested size is created.
    """

    def __init__(
        self,
        width: int,
        height: int,
        grid: Optional[Sequence[Sequence[CellType]]] = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Warehouse dimensions must be positive")

        self.width: int = width
        self.height: int = height

        if grid is None:
            self._grid: List[List[CellType]] = [
                [CellType.FREE for _ in range(width)] for _ in range(height)
            ]
        else:
            if len(grid) != height or any(len(row) != width for row in grid):
                raise ValueError("Provided grid does not match requested width/height")
            self._grid = [[CellType(cell) for cell in row] for row in grid]

        # Dynamic obstacles are stored as a set of coordinates and a
        # separate id counter so each obstacle is identifiable in event
        # logs. ``_dyn_id_seq`` is monotonically increasing.
        self._dynamic_obstacles: Set[CellCoord] = set()
        self._dynamic_obstacle_meta: dict[CellCoord, str] = {}
        self._dyn_id_seq: int = 0

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------
    def is_within_bounds(self, x: int, y: int) -> bool:
        """Return True if ``(x, y)`` is a valid grid coordinate."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell_type(self, x: int, y: int) -> CellType:
        if not self.is_within_bounds(x, y):
            raise IndexError(f"Cell ({x}, {y}) is outside the warehouse")
        return self._grid[y][x]

    def has_dynamic_obstacle(self, x: int, y: int) -> bool:
        """Return True if a dynamic obstacle currently sits on ``(x, y)``."""
        return (x, y) in self._dynamic_obstacles

    def has_static_obstacle(self, x: int, y: int) -> bool:
        """Return True if a static obstacle (wall) sits on ``(x, y)``."""
        return self.get_cell_type(x, y) is CellType.OBSTACLE

    def is_traversable(self, x: int, y: int) -> bool:
        """Return True if a robot may occupy ``(x, y)`` at this moment.

        Traversable cells are any non-obstacle cell (free, pickup,
        drop-off, charging) that does not currently have a dynamic
        obstacle on top of it.
        """
        if not self.is_within_bounds(x, y):
            return False
        if self._grid[y][x] not in _TRAVERSABLE_TYPES:
            return False
        if (x, y) in self._dynamic_obstacles:
            return False
        return True

    def get_neighbors(self, x: int, y: int) -> List[CellCoord]:
        """Return traversable 4-connected neighbors of ``(x, y)``."""
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if self.is_traversable(*c)]

    # ------------------------------------------------------------------
    # Dynamic obstacles
    # ------------------------------------------------------------------
    def add_dynamic_obstacle(self, x: int, y: int, obstacle_id: Optional[str] = None) -> Obstacle:
        """Add a dynamic obstacle at ``(x, y)``.

        Raises ``ValueError`` if the cell is a static obstacle (wall).
        Returns the obstacle that was added.
        """
        if not self.is_within_bounds(x, y):
            raise ValueError(f"({x}, {y}) is outside the warehouse")
        if self._grid[y][x] is CellType.OBSTACLE:
            raise ValueError(f"({x}, {y}) is a wall; cannot place a dynamic obstacle here")

        self._dyn_id_seq += 1
        oid = obstacle_id or f"dyn-{self._dyn_id_seq}"
        self._dynamic_obstacles.add((x, y))
        self._dynamic_obstacle_meta[(x, y)] = oid
        return Obstacle(position=(x, y), kind=ObstacleKind.DYNAMIC, obstacle_id=oid)

    def remove_dynamic_obstacle(self, x: int, y: int) -> bool:
        """Remove the dynamic obstacle at ``(x, y)``.

        Returns ``True`` if an obstacle was removed, ``False`` otherwise.
        """
        if (x, y) in self._dynamic_obstacles:
            self._dynamic_obstacles.discard((x, y))
            self._dynamic_obstacle_meta.pop((x, y), None)
            return True
        return False

    def get_dynamic_obstacles(self) -> List[Obstacle]:
        """Return a list of all currently active dynamic obstacles."""
        return [
            Obstacle(position=pos, kind=ObstacleKind.DYNAMIC, obstacle_id=oid)
            for pos, oid in self._dynamic_obstacle_meta.items()
        ]

    def dynamic_obstacle_positions(self) -> Set[CellCoord]:
        """Return the set of coordinates with dynamic obstacles."""
        return set(self._dynamic_obstacles)

    # ------------------------------------------------------------------
    # Convenience for tests / introspection
    # ------------------------------------------------------------------
    @classmethod
    def with_grid(cls, grid: Sequence[Sequence[CellType]]) -> "Warehouse":
        """Build a warehouse directly from a 2D cell grid.

        Width / height are inferred from the grid. Provided as a helper
        for tests and demo layouts.
        """
        height = len(grid)
        width = len(grid[0]) if height else 0
        return cls(width=width, height=height, grid=grid)

    def all_traversable_cells(self) -> Iterable[CellCoord]:
        """Iterate over every traversable cell (read-only snapshot)."""
        for y in range(self.height):
            for x in range(self.width):
                if self.is_traversable(x, y):
                    yield (x, y)