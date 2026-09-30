"""Obstacle abstraction.

Both static obstacles (encoded directly in the warehouse grid) and
dynamic obstacles (added by the simulator at runtime) share the same
logical representation. The warehouse tracks dynamic obstacles as a
set of ``(x, y)`` coordinates, but tests and perception code can refer
to the uniform :class:`Obstacle` type.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ObstacleKind(Enum):
    """Kind of obstacle. Used by perception / planning layers."""

    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class Obstacle:
    """A logical obstacle located at a single grid cell.

    Attributes:
        position: ``(x, y)`` grid coordinate.
        kind: static or dynamic.
        obstacle_id: stable identifier (helpful in event logs).
    """

    position: Tuple[int, int]
    kind: ObstacleKind
    obstacle_id: str

    @property
    def x(self) -> int:
        return self.position[0]

    @property
    def y(self) -> int:
        return self.position[1]