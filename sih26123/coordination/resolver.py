"""Wait-vs-reroute cost comparison (Phase 2B).

Decoupled from the simulator: a pure function on numbers. The robot
uses it to decide whether it is cheaper to stand still for a few ticks
or to take a detour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .negotiation import NegotiationAction


@dataclass(frozen=True)
class ResolutionChoice:
    """Result of :func:`choose_yield_action`."""

    action: NegotiationAction
    reason: str
    wait_ticks: int = 0
    reroute_cost: int = 0
    wait_cost: int = 0


def choose_yield_action(
    conflict_timestep: int,
    current_tick: int,
    reroute_path_length: Optional[int],
    old_remaining_length: int,
    reroute_threshold_multiplier: float = 2.0,
) -> ResolutionChoice:
    """Decide whether the yielding robot should WAIT or REROUTE.

    Parameters
        ``conflict_timestep``:
            The tick at which the conflict is predicted to occur.
        ``current_tick``:
            The simulator's current tick.
        ``reroute_path_length``:
            Length of the alternative spatial path returned by the
            space-time A* planner, or ``None`` if no path exists.
        ``old_remaining_length``:
            Number of cells still to traverse on the current spatial
            path.
        ``reroute_threshold_multiplier``:
            A reroute is preferred over waiting if
            ``reroute_cost <= reroute_threshold_multiplier * wait_cost``.
            Defaults to ``2.0`` (waits are cheap, reroutes are noisy).

    Returns
        A :class:`ResolutionChoice` describing the action and the
        explanatory reason. The reason is plain text suitable for the
        ``ROBOT_YIELDED`` event payload.
    """
    wait_ticks = max(1, int(conflict_timestep) - int(current_tick))
    wait_cost = wait_ticks

    if reroute_path_length is None:
        return ResolutionChoice(
            action=NegotiationAction.WAIT,
            reason="no_feasible_reroute",
            wait_ticks=wait_ticks,
            wait_cost=wait_cost,
        )

    reroute_cost = max(0, int(reroute_path_length) - int(old_remaining_length))

    if reroute_cost <= 0:
        return ResolutionChoice(
            action=NegotiationAction.REROUTE,
            reason="reroute_free_or_shorter",
            reroute_cost=reroute_cost,
            wait_cost=wait_cost,
        )

    if reroute_cost <= reroute_threshold_multiplier * wait_cost:
        return ResolutionChoice(
            action=NegotiationAction.REROUTE,
            reason="reroute_cheaper_than_wait",
            reroute_cost=reroute_cost,
            wait_cost=wait_cost,
        )

    return ResolutionChoice(
        action=NegotiationAction.WAIT,
        reason="wait_cheaper_than_reroute",
        wait_ticks=wait_ticks,
        reroute_cost=reroute_cost,
        wait_cost=wait_cost,
    )