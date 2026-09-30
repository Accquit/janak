"""Space-time A* planner for rerouting (Phase 2B).

The standard ``planning/astar.py`` reasons about space only. Here we
need to reason about space AND time: a robot that wants to avoid a
specific ``(cell, timestep)`` pair (because a peer has reserved it).

State
-----
A state is the triple ``(x, y, t)`` where ``(x, y)`` is the grid cell
and ``t`` is the absolute tick at which the robot will occupy that
cell. Time advances by one per step.

Actions
--------
From ``(x, y, t)`` we can transition to:

* ``(x+1, y, t+1)``   move east
* ``(x-1, y, t+1)``   move west
* ``(x, y+1, t+1)``   move south
* ``(x, y-1, t+1)``   move north
* ``(x, y, t+1)``     **wait** in place

All actions have cost ``1``.

Blocking
--------
A state ``(x, y, t)`` is **blocked** iff any of:

* ``(x, y)`` is outside the warehouse;
* the warehouse declares the cell non-traversable
  (``is_traversable`` False — covers walls and dynamic obstacles);
* a *peer* has reserved ``(x, y, t)`` in the reservation table.

Own reservations are NOT considered blocking (we are rebuilding them
in the same step). This makes the planner conservative towards peers
without prohibiting the planner from returning to cells it had
reserved earlier in its own trajectory.

Heuristic
---------
Manhattan distance. Admissible because each step costs ``1``.

Determinism
-----------
Tie-breaking in the heap uses a monotonically increasing counter so
two states with equal ``f`` are popped in FIFO order. Together with
the heuristic and tie-break this makes the returned path unique.
"""

from __future__ import annotations

import heapq
from typing import Callable, List, Optional, Sequence, Tuple

CellCoord = Tuple[int, int]
SpatioTemporalState = Tuple[int, int, int]  # (x, y, t)


def space_time_astar(
    start: CellCoord,
    goal: CellCoord,
    start_tick: int,
    width: int,
    height: int,
    is_cell_traversable: Callable[[int, int], bool],
    is_blocked_at: Callable[[int, int, int], bool],
    max_timestep: Optional[int] = None,
    is_transition_blocked: Optional[
        Callable[[CellCoord, CellCoord, int, int], bool]
    ] = None,
) -> Optional[List[SpatioTemporalState]]:
    """A* over a space-time grid.

    Parameters
        ``start``:
            ``(x, y)`` cell where the robot currently is.
        ``goal``:
            ``(x, y)`` destination cell.
        ``start_tick``:
            The current tick (the robot will be at ``start`` at this
            tick, so the first transition advances time to
            ``start_tick + 1``).
        ``width``, ``height``:
            Grid dimensions.
        ``is_cell_traversable``:
            Callable returning ``True`` iff the spatial cell may be
            entered (static + dynamic obstacles).
        ``is_blocked_at``:
            Callable returning ``True`` iff ``(x, y, t)`` is reserved
            by a peer (or otherwise unavailable).
        ``max_timestep``:
            Hard upper bound on ``t`` for the search. ``None`` means
            ``start_tick + (width * height)`` (a generous fallback
            that prevents infinite search on disconnected graphs).
        ``is_transition_blocked``:
            Optional predicate for transition conflicts such as robots
            traversing the same edge in opposite directions.

    Returns
        A list of space-time states from the planner's perspective,
        *including* the start state, ordered by increasing time. The
        spatial coordinates of the returned list (dropping ``t``)
        form the spatial path the robot should follow.

        ``None`` if no path is found within the horizon.
    """
    if width <= 0 or height <= 0:
        return None
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    t0 = int(start_tick)

    if not (0 <= sx < width and 0 <= sy < height):
        raise ValueError(f"start {start} is outside the grid")
    if not (0 <= gx < width and 0 <= gy < height):
        raise ValueError(f"goal {goal} is outside the grid")
    if not is_cell_traversable(sx, sy):
        return None
    if not is_cell_traversable(gx, gy):
        return None
    if is_blocked_at(sx, sy, t0):
        # Even the starting (x, y, t0) is forbidden by peers. Cannot plan.
        return None

    if max_timestep is None:
        max_timestep = t0 + width * height + 1

    # If start == goal we still return a single state (the robot is
    # already there).
    if (sx, sy) == (gx, gy):
        return [(sx, sy, t0)]

    neighbours: Sequence[Tuple[int, int, int]] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 0),  # wait
    )

    start_state: SpatioTemporalState = (sx, sy, t0)
    g_score: dict[SpatioTemporalState, int] = {start_state: 0}
    came_from: dict[SpatioTemporalState, SpatioTemporalState] = {}

    def h(state: SpatioTemporalState) -> int:
        x, y, _ = state
        return abs(x - gx) + abs(y - gy)

    counter = 0
    open_heap: List[Tuple[int, int, int, SpatioTemporalState]] = []
    heapq.heappush(open_heap, (h(start_state), 0, counter, start_state))
    counter += 1
    closed: set[SpatioTemporalState] = set()

    while open_heap:
        f, g, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if g != g_score.get(current, None):
            # Stale heap entry.
            continue

        cx, cy, ct = current
        if (cx, cy) == (gx, gy):
            # Reconstruct.
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)

        nt = ct + 1
        if nt > max_timestep:
            continue

        for dx, dy, _ in neighbours:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if not is_cell_traversable(nx, ny):
                continue
            if is_blocked_at(nx, ny, nt):
                continue
            if is_transition_blocked is not None and is_transition_blocked(
                (cx, cy), (nx, ny), ct, nt,
            ):
                continue
            new_state: SpatioTemporalState = (nx, ny, nt)
            if new_state in closed:
                continue
            tentative_g = g + 1
            if tentative_g < g_score.get(new_state, 10**18):
                g_score[new_state] = tentative_g
                came_from[new_state] = current
                f_score = tentative_g + h(new_state)
                heapq.heappush(open_heap, (f_score, tentative_g, counter, new_state))
                counter += 1

    return None


def spatial_path(space_time_path: List[SpatioTemporalState]) -> List[CellCoord]:
    """Strip the time component from a space-time path."""
    return [(x, y) for (x, y, _t) in space_time_path]


def remaining_ticks_at(path: List[SpatioTemporalState], tick: int) -> int:
    """Return the index in ``path`` of the first state with ``t > tick``.

    Used to compute "how many ticks does the current spatial path
    still span from this tick onwards".
    """
    for i, (_x, _y, t) in enumerate(path):
        if t > tick:
            return len(path) - i
    return 0
