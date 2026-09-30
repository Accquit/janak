"""A* path planner for a 4-connected grid.

Public API:
    - :func:`manhattan`  - admissible heuristic for 4-connected movement.
    - :func:`astar`      - find a shortest path that avoids obstacles.

Design notes
------------
* ``astar`` only knows about grid bounds and a ``is_traversable`` callable.
  It does not import :class:`Warehouse`, which keeps it testable in
  isolation and lets us reuse it for any grid (e.g. for rerouting in the
  next layer).
* The path returned always includes both ``start`` and ``goal`` cells
  (when reachable). When the goal is unreachable the function returns
  ``None``; it never raises and never loops forever. The closed set is
  explicit, so even degenerate inputs terminate.
"""

from __future__ import annotations

import heapq
from typing import Callable, List, Optional, Sequence, Tuple

CellCoord = Tuple[int, int]


def manhattan(a: Sequence[int], b: Sequence[int]) -> int:
    """Manhattan distance for a 4-connected grid.

    Both arguments must be 2-element sequences (e.g. tuples or lists).
    """
    if len(a) < 2 or len(b) < 2:
        raise ValueError("manhattan requires 2D coordinates")
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def astar(
    start: CellCoord,
    goal: CellCoord,
    width: int,
    height: int,
    is_traversable: Callable[[int, int], bool],
) -> Optional[List[CellCoord]]:
    """A* over a 4-connected grid.

    Parameters
        ``start``:
            ``(x, y)`` starting cell.
        ``goal``:
            ``(x, y)`` goal cell.
        ``width, height``:
            Grid dimensions.
        ``is_traversable``:
            Callable returning True if ``(x, y)`` may be entered. Used
            for both static and dynamic obstacles. The caller decides
            what counts as traversable; the planner stays oblivious.

    Returns
        A list of ``(x, y)`` cells from ``start`` to ``goal`` (inclusive),
        or ``None`` if no valid path exists.

    Notes
        Movement is 4-connected (N, S, E, W). All step costs are 1.
        The heuristic is Manhattan distance and is admissible.
    """
    if width <= 0 or height <= 0:
        return None
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])

    if not (0 <= sx < width and 0 <= sy < height):
        raise ValueError(f"start {start} is outside the grid")
    if not (0 <= gx < width and 0 <= gy < height):
        raise ValueError(f"goal {goal} is outside the grid")
    if not is_traversable(sx, sy):
        return None
    if not is_traversable(gx, gy):
        return None
    if (sx, sy) == (gx, gy):
        return [(sx, sy)]

    # Direction offsets: East, West, South, North.
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    # Open set as a min-heap of (f, g, x, y). We also keep a tentative-g
    # map so we can detect stale heap entries (lazy deletion).
    open_heap: List[Tuple[int, int, int, int]] = []
    g_score: dict[CellCoord, int] = {(sx, sy): 0}
    came_from: dict[CellCoord, CellCoord] = {}
    counter = 0  # tie-breaker to keep heap stable
    heapq.heappush(open_heap, (manhattan((sx, sy), (gx, gy)), 0, counter, (sx, sy)))
    counter += 1

    closed: set[CellCoord] = set()

    while open_heap:
        f, g, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        if g != g_score.get(current, None):
            # Stale heap entry.
            continue

        cx, cy = current
        if current == (gx, gy):
            # Reconstruct path.
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)

        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in closed:
                continue
            if not is_traversable(nx, ny):
                continue

            tentative_g = g + 1
            if tentative_g < g_score.get((nx, ny), float("inf")):
                g_score[(nx, ny)] = tentative_g
                came_from[(nx, ny)] = current
                f_score = tentative_g + manhattan((nx, ny), (gx, gy))
                heapq.heappush(open_heap, (f_score, tentative_g, counter, (nx, ny)))
                counter += 1

    # Open set exhausted without reaching goal.
    return None