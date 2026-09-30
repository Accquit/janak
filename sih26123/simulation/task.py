"""Task abstraction: a single pickup-and-delivery job.

A task carries information about the pickup location, drop-off
location, priority and current status. For the foundation, tasks are
assigned manually; intelligent task allocation will be added in the
next layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

CellCoord = Tuple[int, int]


class TaskStatus(Enum):
    """Lifecycle status of a task."""

    PENDING = "pending"          # created but no robot assigned
    ASSIGNED = "assigned"        # a robot is assigned but not yet moving
    IN_PROGRESS = "in_progress"  # robot is moving toward pickup or dropoff
    COMPLETED = "completed"      # drop-off reached
    FAILED = "failed"            # failed (e.g., battery died)


class TaskPhase(Enum):
    """Within ``IN_PROGRESS`` we track whether we are heading to pickup
    or drop-off. ``DONE`` is used after the task completes.
    """

    TO_PICKUP = "to_pickup"
    TO_DROPOFF = "to_dropoff"
    DONE = "done"


@dataclass
class Task:
    """A single pickup-and-delivery task.

    Attributes
        ``task_id``:
            Unique identifier. Used in event logs.
        ``pickup_location``:
            Grid coordinate where the robot picks up.
        ``dropoff_location``:
            Grid coordinate where the robot drops off.
        ``priority``:
            Higher number = higher priority. Ties broken by creation
            order. Foundation only uses priority for descriptive
            purposes; the next layer will use it for negotiation.
        ``deadline``:
            Optional maximum simulation time (tick) by which the task
            must complete. ``None`` means no deadline.
        ``status``:
            Lifecycle status.
        ``phase``:
            Current sub-phase when ``status`` is ``IN_PROGRESS``.
        ``assigned_robot``:
            ``robot_id`` of the assigned robot, or ``None``.
    """

    task_id: str
    pickup_location: CellCoord
    dropoff_location: CellCoord
    priority: int = 0
    deadline: Optional[int] = None
    status: TaskStatus = TaskStatus.PENDING
    phase: TaskPhase = TaskPhase.TO_PICKUP
    assigned_robot: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def current_target(self) -> Optional[CellCoord]:
        """Return the grid coordinate the assigned robot is heading to."""
        if self.status in (TaskStatus.PENDING, TaskStatus.FAILED):
            return None
        if self.status is TaskStatus.COMPLETED:
            return None
        if self.phase is TaskPhase.TO_PICKUP:
            return self.pickup_location
        if self.phase is TaskPhase.TO_DROPOFF:
            return self.dropoff_location
        return None

    def mark_assigned(self, robot_id: str) -> None:
        """Transition the task to ``ASSIGNED`` and record the robot."""
        if self.status is not TaskStatus.PENDING:
            raise RuntimeError(
                f"Task {self.task_id} cannot be assigned; status is {self.status.value}"
            )
        self.assigned_robot = robot_id
        self.status = TaskStatus.ASSIGNED
        self.phase = TaskPhase.TO_PICKUP

    def mark_in_progress(self) -> None:
        if self.status not in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            raise RuntimeError(
                f"Task {self.task_id} cannot enter progress from {self.status.value}"
            )
        self.status = TaskStatus.IN_PROGRESS

    def mark_pickup_done(self) -> None:
        """Called locally by the robot once it reaches the pickup cell."""
        if self.phase is not TaskPhase.TO_PICKUP:
            raise RuntimeError(
                f"Task {self.task_id} is not in TO_PICKUP phase (was {self.phase.value})"
            )
        self.phase = TaskPhase.TO_DROPOFF

    def mark_completed(self) -> None:
        if self.status is not TaskStatus.IN_PROGRESS:
            raise RuntimeError(
                f"Task {self.task_id} cannot complete from {self.status.value}"
            )
        self.phase = TaskPhase.DONE
        self.status = TaskStatus.COMPLETED

    def mark_failed(self) -> None:
        self.status = TaskStatus.FAILED

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Task(id={self.task_id!r}, status={self.status.value}, "
            f"phase={self.phase.value}, robot={self.assigned_robot!r})"
        )