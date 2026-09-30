"""Perception abstraction and a deterministic simulator implementation.

The :class:`Perception` base class is the contract that any robot
perception module must satisfy. The current
:class:`SimulatedPerception` implementation reads from a *read-only*
``WorldView`` snapshot that the simulator hands to it. In the next
stage, this is the seam where we plug in either a real sensor model
or a P2P-received local view.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from ..simulation.obstacle import Obstacle, ObstacleKind
from ..simulation.robot import Robot


@dataclass(frozen=True)
class RobotObservation:
    """A summary of what another robot looks like to the perceiver.

    Foundation only uses position and id; the next layer will add
    velocity, intent, planned path, etc.
    """

    robot_id: str
    position: Tuple[int, int]


class Perception(ABC):
    """Abstract base class for any perception module a robot owns."""

    @abstractmethod
    def get_static_obstacles(self, robot: Robot) -> List[Obstacle]:
        """Return static obstacles (walls) within the robot's sensor range."""

    @abstractmethod
    def get_dynamic_obstacles(self, robot: Robot) -> List[Obstacle]:
        """Return dynamic obstacles within the robot's sensor range."""

    @abstractmethod
    def get_nearby_robots(self, robot: Robot) -> List[RobotObservation]:
        """Return other robots within the robot's sensor range."""


class SimulatedPerception(Perception):
    """A deterministic, ideal-noise-free sensor.

    Parameters
        ``world_view``:
            Read-only object exposing ``warehouse``, ``robots`` and
            ``tick``. The simulator hands each robot its own copy of
            this view, so the robot never reaches into simulator
            internals.
        ``sensor_range``:
            Manhattan radius in cells. Cells whose Manhattan distance
            from the robot exceeds this value are invisible.
        ``self_id``:
            Id of the robot that owns this perception. Used to filter
            the perceiver out of its own ``get_nearby_robots`` results.
    """

    def __init__(self, world_view: "WorldView", sensor_range: int = 3, self_id: str = "") -> None:
        self.world_view = world_view
        self.sensor_range = max(0, int(sensor_range))
        self.self_id = self_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _in_range(self, robot: Robot, x: int, y: int) -> bool:
        rx, ry = robot.position
        if self.sensor_range == 0:
            return x == rx and y == ry
        return abs(x - rx) + abs(y - ry) <= self.sensor_range

    def _iter_cells(self, robot: Robot) -> Iterable[Tuple[int, int]]:
        rx, ry = robot.position
        rng = self.sensor_range
        height = self.world_view.warehouse.height
        width = self.world_view.warehouse.width
        for y in range(max(0, ry - rng), min(height, ry + rng + 1)):
            for x in range(max(0, rx - rng), min(width, rx + rng + 1)):
                yield x, y

    # ------------------------------------------------------------------
    # Public Perception API
    # ------------------------------------------------------------------
    def get_static_obstacles(self, robot: Robot) -> List[Obstacle]:
        wh = self.world_view.warehouse
        out: List[Obstacle] = []
        for x, y in self._iter_cells(robot):
            if self._in_range(robot, x, y) and wh.has_static_obstacle(x, y):
                out.append(Obstacle(position=(x, y), kind=ObstacleKind.STATIC, obstacle_id=f"static-{x}-{y}"))
        return out

    def get_dynamic_obstacles(self, robot: Robot) -> List[Obstacle]:
        wh = self.world_view.warehouse
        out: List[Obstacle] = []
        for x, y in self._iter_cells(robot):
            if not self._in_range(robot, x, y):
                continue
            if wh.has_dynamic_obstacle(x, y):
                out.append(Obstacle(position=(x, y), kind=ObstacleKind.DYNAMIC, obstacle_id=f"dyn-{x}-{y}"))
        return out

    def get_nearby_robots(self, robot: Robot) -> List[RobotObservation]:
        out: List[RobotObservation] = []
        for other in self.world_view.robots:
            if other.robot_id == self.self_id:
                continue
            if self._in_range(robot, other.position[0], other.position[1]):
                out.append(RobotObservation(robot_id=other.robot_id, position=other.position))
        return out


@dataclass
class WorldView:
    """A read-only snapshot of the world a robot can perceive against.

    The simulator hands each robot its own :class:`WorldView` instance.
    Robots do not (and should not) mutate anything through this view.

    In the next layer this object will be replaced by per-robot views
    derived from P2P messages.
    """

    warehouse: object
    robots: List[Robot]
    tick: int