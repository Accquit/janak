"""Deterministic priority score for decentralized conflict resolution (Phase 2B).

Each robot computes its own priority score from local state. The score is a
pure function: identical inputs always produce identical outputs. Ties are
broken lexicographically by ``robot_id`` so the comparison is fully
deterministic.

The score is in ``[0, 1]``. The formula is intentionally simple and
explainable — no ML, no hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PriorityInputs:
    """All inputs required to compute a robot's priority score.

    Attributes
        ``robot_id``:
            Owning robot's id. Used for tie-breaking only.
        ``task_priority``:
            Raw integer priority of the currently assigned task. Higher
            means more urgent. ``0`` if no task.
        ``max_task_priority``:
            Upper bound used to normalise ``task_priority`` to ``[0, 1]``.
            Must be ``>= 1``.
        ``deadline``:
            Absolute tick by which the task must complete. ``None`` if no
            deadline; in that case deadline pressure is zero.
        ``current_tick``:
            The simulator's current tick.
        ``battery``:
            Current battery level.
        ``battery_capacity``:
            Battery capacity (used to normalise pressure to ``[0, 1]``).
        ``waiting_time``:
            Number of ticks the robot has been waiting recently.
        ``remaining_path_length``:
            Cells still to be traversed on the robot's current plan.
        ``expected_path_length``:
            Original path length used for route-progress normalisation.
        ``max_horizon``:
            Reference horizon (in ticks) used to normalise deadline and
            waiting pressure. Must be ``>= 1``.
    """

    robot_id: str
    task_priority: int
    max_task_priority: int
    deadline: Optional[int]
    current_tick: int
    battery: float
    battery_capacity: float
    waiting_time: int
    remaining_path_length: int
    expected_path_length: int
    max_horizon: int = 20


# Weights. They sum to 1.0 and are easy to explain to reviewers.
W_TASK = 0.40
W_DEADLINE = 0.25
W_ROUTE = 0.15
W_WAITING = 0.10
W_BATTERY = 0.10


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_priority(inputs: PriorityInputs) -> float:
    """Return the deterministic priority score in ``[0, 1]``.

    Same inputs always yield the same result. No global state, no
    randomness, no time-of-day effects.
    """
    # Task urgency: linear normalisation into [0, 1].
    task_score = _clip01(inputs.task_priority / max(inputs.max_task_priority, 1))

    # Deadline pressure: 0 if no deadline, 1 if overdue or imminent.
    deadline_score = 0.0
    if inputs.deadline is not None:
        horizon = max(inputs.max_horizon, 1)
        remaining = int(inputs.deadline) - int(inputs.current_tick)
        if remaining <= 0:
            deadline_score = 1.0
        else:
            deadline_score = _clip01(1.0 - remaining / horizon)

    # Route progress: closer to goal → higher score.
    route_score = 0.5
    if inputs.expected_path_length > 0:
        route_score = _clip01(1.0 - inputs.remaining_path_length / inputs.expected_path_length)

    # Waiting pressure: bounded linear.
    horizon = max(inputs.max_horizon, 1)
    waiting_score = _clip01(inputs.waiting_time / horizon)

    # Battery pressure: low battery → higher score.
    battery_score = 0.0
    if inputs.battery_capacity > 0:
        battery_score = _clip01(1.0 - inputs.battery / inputs.battery_capacity)

    score = (
        W_TASK * task_score
        + W_DEADLINE * deadline_score
        + W_ROUTE * route_score
        + W_WAITING * waiting_score
        + W_BATTERY * battery_score
    )
    return _clip01(score)


def is_higher_priority(
    score_a: float,
    score_b: float,
    id_a: str,
    id_b: str,
) -> bool:
    """Decide which robot should proceed when both want to.

    Returns ``True`` if robot A has priority over robot B. On a tie
    (within a tiny epsilon), the lexicographically higher ``robot_id``
    wins — making the choice deterministic across runs.
    """
    EPS = 1e-9
    if score_a > score_b + EPS:
        return True
    if score_a + EPS < score_b:
        return False
    return id_a > id_b


def priority_inputs_from_robot(
    robot_id: str,
    task_priority: int,
    max_task_priority: int,
    deadline: Optional[int],
    current_tick: int,
    battery: float,
    battery_capacity: float,
    waiting_time: int,
    remaining_path_length: int,
    expected_path_length: int,
    max_horizon: int = 20,
) -> PriorityInputs:
    """Convenience builder that mirrors the dataclass signature."""
    return PriorityInputs(
        robot_id=robot_id,
        task_priority=task_priority,
        max_task_priority=max_task_priority,
        deadline=deadline,
        current_tick=current_tick,
        battery=battery,
        battery_capacity=battery_capacity,
        waiting_time=waiting_time,
        remaining_path_length=remaining_path_length,
        expected_path_length=expected_path_length,
        max_horizon=max_horizon,
    )