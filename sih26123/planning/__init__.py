"""Single-robot A* path planner over a grid.

The planner is decoupled from the warehouse: it only requires a width,
height and a callable ``is_traversable(x, y) -> bool``. This keeps the
planner reusable for both static and dynamic environments, and makes it
trivial to plug in a different world model later.
"""

from .astar import astar, manhattan

__all__ = ["astar", "manhattan"]