"""Robot agent — foundation version (extended with optional coordination).

A :class:`Robot` is the unit of agency in this prototype. The robot owns
its own state, perception and battery, and executes its own one-tick
decision making via :meth:`Robot.step`. The simulator does not drive
the robot; it just calls ``step`` once per tick.

Foundation responsibilities of :meth:`step`
-------------------------------------------
1. Read its perception to detect static and dynamic obstacles around
   the planned path.
2. If the next planned cell is blocked by a dynamic obstacle, transition
   to :attr:`RobotStatus.WAITING`. (Re-routing is intentionally left
   for the next layer.)
3. Otherwise consume battery, advance position along the path, and
   transition state appropriately when destinations are reached.
4. Return the list of events the robot produced this tick; the
   simulator records them.

This keeps the architecture honest about the eventual decentralization:
each robot, given only its own state and a local view, can decide what
to do.

Phase 2A extension
------------------
The robot may optionally be constructed with a ``coordination`` field
of type :class:`coordination.agent.RobotCoordination`. The simulator
calls the coordination hooks before and after ``step``. ``step`` itself
is unchanged: motion remains local to the robot.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

from .events import Event, EventType

if TYPE_CHECKING:
    from ..coordination.agent import RobotCoordination
    from ..perception.simulated_sensor import Perception

CellCoord = Tuple[int, int]


class RobotStatus(Enum):
    """Lifecycle status of a robot."""

    IDLE = "idle"            # no task; standing by
    MOVING = "moving"        # following a path
    WAITING = "waiting"      # path is obstructed; needs re-plan (foundation: just waits)
    COMPLETED = "completed"  # finished current task
    FAILED = "failed"        # unrecoverable (e.g. battery exhausted)


# Battery consumption model: per-cell movement + small idle drain.
DEFAULT_MOVE_COST = 1.0
DEFAULT_IDLE_COST = 0.05


class RobotStatus(Enum):
    """Lifecycle status of a robot."""

    IDLE = "idle"            # no task; standing by
    MOVING = "moving"        # following a path
    WAITING = "waiting"      # path is obstructed; needs re-plan (foundation: just waits)
    COMPLETED = "completed"  # finished current task
    FAILED = "failed"        # unrecoverable (e.g. battery exhausted)


# Battery consumption model: per-cell movement + small idle drain.
DEFAULT_MOVE_COST = 1.0
DEFAULT_IDLE_COST = 0.05


class Robot:
    """A single autonomous mobile robot (AMR).

    Parameters
        ``robot_id``:
            Unique identifier.
        ``start_position``:
            Initial ``(x, y)`` grid coordinate.
        ``perception``:
            A :class:`Perception` instance local to this robot.
        ``battery_capacity``:
            Maximum battery level (used for normalization). Defaults
            to ``100.0``.
        ``move_cost``:
            Battery consumed per cell of movement.
        ``idle_cost``:
            Battery consumed per tick when not moving.
        ``coordination``:
            Optional :class:`RobotCoordination`. When ``None`` (the
            default) the robot behaves exactly as in Phase 1: the
            simulator never touches any coordination layer.
    """

    def __init__(
        self,
        robot_id: str,
        start_position: CellCoord,
        perception: "Perception",
        battery_capacity: float = 100.0,
        move_cost: float = DEFAULT_MOVE_COST,
        idle_cost: float = DEFAULT_IDLE_COST,
        coordination: Optional["RobotCoordination"] = None,
    ) -> None:
        self.robot_id = robot_id
        self.position: CellCoord = tuple(start_position)
        self.velocity: Tuple[int, int] = (0, 0)
        self.battery: float = float(battery_capacity)
        self.battery_capacity: float = float(battery_capacity)

        self.current_task = None  # type: Optional[object]
        self.path: List[CellCoord] = []
        self.path_index: int = 0
        self.destination: Optional[CellCoord] = None

        self.status: RobotStatus = RobotStatus.IDLE

        self.perception = perception
        self.move_cost = float(move_cost)
        self.idle_cost = float(idle_cost)

        # Optional Phase 2A coordination module. If None, the robot
        # has no peer awareness, no reservations and no conflict
        # prediction. The simulator skips all coordination hooks.
        self.coordination: Optional["RobotCoordination"] = coordination

        # Phase 2B additions: negotiation-driven waiting. ``wait``
        # actions set ``status = WAITING`` *and* ``waiting_until_tick``;
        # ``step`` resumes ``MOVING`` when the tick is reached.
        self.waiting_until_tick: Optional[int] = None
        self.waiting_time: int = 0

        # Phase 2C test hook: if set, the robot emits no broadcasts
        # and emits no heartbeats. Use to simulate a robot that has
        # gone completely silent (its broadcasts are no longer reaching
        # the bus at all). Production code never sets this.
        self.silenced: bool = False

    # ------------------------------------------------------------------
    # Coordination intent helper (used by the simulator when broadcasting)
    # ------------------------------------------------------------------
    def describe_intent(self) -> str:
        """Return a short human-readable intent string for broadcasts.

        This is intentionally a string (not an enum) so future
        layers can extend the vocabulary without coordination-layer
        changes.
        """
        if self.status is RobotStatus.MOVING:
            if self.current_task is not None:
                # Distinguish pickup vs. dropoff without importing Task
                # to avoid circular imports.
                phase = getattr(self.current_task, "phase", None)
                if phase is not None and getattr(phase, "value", None) == "to_pickup":
                    return "moving_to_pickup"
                if phase is not None and getattr(phase, "value", None) == "to_dropoff":
                    return "moving_to_dropoff"
            return "moving"
        if self.status is RobotStatus.WAITING:
            return "waiting"
        if self.status is RobotStatus.IDLE:
            return "idle"
        if self.status is RobotStatus.COMPLETED:
            return "completed"
        return "failed"

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_alive(self) -> bool:
        return self.status is not RobotStatus.FAILED

    @property
    def has_path(self) -> bool:
        return self.path_index < len(self.path)

    def next_path_cell(self) -> Optional[CellCoord]:
        """Return the cell the robot is about to move into next, or None."""
        if not self.has_path:
            return None
        return self.path[self.path_index]

    def remaining_path(self) -> List[CellCoord]:
        """Return the path cells the robot has not yet entered."""
        if not self.has_path:
            return []
        return list(self.path[self.path_index:])

    # ------------------------------------------------------------------
    # Planning / assignment (foundation: caller-driven)
    # ------------------------------------------------------------------
    def set_path(self, path: List[CellCoord], destination: CellCoord) -> None:
        """Set a planned path and destination.

        ``path`` is a list of grid cells the robot will move into, in
        order. The robot's current position is **not** included in the
        path. If A* (or any other planner) returns a path that does
        include the current cell as the first step, it is stripped
        here. If the resulting path is empty (robot already at
        destination), the robot transitions to :attr:`RobotStatus.IDLE`.
        """
        cleaned = [tuple(c) for c in path]
        # Strip a leading cell equal to the robot's current position.
        while cleaned and cleaned[0] == self.position:
            cleaned.pop(0)

        self.path = cleaned
        self.path_index = 0
        self.destination = tuple(destination)

        if not self.path or self.position == self.destination:
            self.status = RobotStatus.IDLE
            return

        self.status = RobotStatus.MOVING

    def clear_path(self) -> None:
        self.path = []
        self.path_index = 0
        self.destination = None
        if self.status in (RobotStatus.MOVING, RobotStatus.WAITING):
            self.status = RobotStatus.IDLE

    # ------------------------------------------------------------------
    # One tick of local decision-making
    # ------------------------------------------------------------------
    def step(self, tick: int, world) -> List[Event]:
        """Advance the robot by one timestep.

        ``world`` is the :class:`simulation.warehouse.Warehouse`
        instance. The robot only reads from it (via its perception, if
        needed). The simulator does the actual mutation of obstacles
        and tasks. This method *only* mutates the robot itself.
        """
        events: List[Event] = []

        if self.status is RobotStatus.FAILED:
            return events

        # Phase 2B: resume movement after a negotiation-driven wait.
        if (
            self.status is RobotStatus.WAITING
            and self.waiting_until_tick is not None
            and tick >= self.waiting_until_tick
        ):
            self.status = RobotStatus.MOVING
            self.waiting_until_tick = None

        # Track cumulative waiting time for priority computation.
        if self.status is RobotStatus.WAITING:
            self.waiting_time += 1
        else:
            self.waiting_time = 0

        # Detect failure conditions first.
        if self.battery <= 0.0:
            self.status = RobotStatus.FAILED
            if self.current_task is not None:
                self.current_task.mark_failed()
            events.append(Event(tick, EventType.ROBOT_FAILED, robot_id=self.robot_id, reason="battery_exhausted"))
            return events

        # WAITING robots re-evaluate the path each tick. If the obstacle
        # cleared, they become MOVING and continue into the movement
        # branch below.
        if self.status is RobotStatus.WAITING:
            # Phase 2B: a negotiation-driven wait takes precedence
            # over the dynamic-obstacle check. The robot stays put
            # until the resume check at the top of ``step`` lifts
            # ``status`` back to MOVING at ``waiting_until_tick``.
            if self.waiting_until_tick is not None:
                self.battery = max(0.0, self.battery - self.idle_cost)
                return events
            if not self.has_path:
                self.status = RobotStatus.IDLE
                self.battery = max(0.0, self.battery - self.idle_cost)
                return events
            next_cell = self.next_path_cell()
            assert next_cell is not None
            dynamic_blockers = [
                o for o in self.perception.get_dynamic_obstacles(self)
                if o.position == next_cell
            ]
            if dynamic_blockers:
                self.battery = max(0.0, self.battery - self.idle_cost)
                return events
            # Path is clear again: become MOVING this tick.
            self.status = RobotStatus.MOVING

        # MOVING: try to step into the next cell.
        if self.status is RobotStatus.MOVING and self.has_path:
            next_cell = self.next_path_cell()
            assert next_cell is not None

            dynamic_blockers = [
                o for o in self.perception.get_dynamic_obstacles(self)
                if o.position == next_cell
            ]
            if dynamic_blockers:
                # Blocked by a dynamic obstacle; robot waits. The
                # next layer will plug in re-routing / negotiation.
                self.status = RobotStatus.WAITING
                self.velocity = (0, 0)
                events.append(Event(
                    tick,
                    EventType.ROBOT_WAITING,
                    robot_id=self.robot_id,
                    blocked_cell=list(next_cell),
                    obstacle_id=dynamic_blockers[0].obstacle_id,
                ))
                events.append(Event(
                    tick,
                    EventType.OBSTACLE_DETECTED,
                    robot_id=self.robot_id,
                    obstacle_id=dynamic_blockers[0].obstacle_id,
                    position=list(next_cell),
                ))
                return events

            # Move into next cell.
            old = self.position
            self.position = next_cell
            self.path_index += 1
            self.velocity = (self.position[0] - old[0], self.position[1] - old[1])
            self.battery = max(0.0, self.battery - self.move_cost)
            events.append(Event(tick, EventType.ROBOT_MOVED, robot_id=self.robot_id, from_=list(old), to=list(self.position)))

            if not self.has_path:
                self.velocity = (0, 0)
                self.status = RobotStatus.IDLE
                events.append(Event(tick, EventType.ROBOT_ARRIVED, robot_id=self.robot_id, position=list(self.position), destination=list(self.destination) if self.destination else None))
            return events

        # IDLE / COMPLETED: small idle drain and that's it.
        self.battery = max(0.0, self.battery - self.idle_cost)
        return events

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Robot(id={self.robot_id!r}, pos={self.position}, "
            f"status={self.status.value}, battery={self.battery:.1f})"
        )