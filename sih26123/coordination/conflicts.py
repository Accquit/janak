"""Conflict prediction (Phase 2A).

The :class:`ConflictDetector` answers **one** question, locally:

    "Given my own planned trajectory and the peer trajectories I have
     learned about, will I (or any pair of robots I know about) be in
     conflict within the next H ticks?"

It does **not** decide who should yield. That belongs to the future
negotiation layer.

Conflict types
--------------
* **Vertex conflict** — two robots occupy the same cell at the same
  tick.
* **Edge conflict** — two robots swap cells during the same tick,
  i.e. ``A`` moves ``X -> Y`` while ``B`` moves ``Y -> X``.

A vertex conflict and an edge conflict at the same tick are reported
as separate :class:`PredictedConflict` objects; the caller may
deduplicate if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .reservations import Reservation, ReservationTable

CellCoord = Tuple[int, int]


class ConflictType(Enum):
    VERTEX = "vertex"
    EDGE = "edge"


@dataclass(frozen=True)
class PredictedConflict:
    """A predicted conflict involving two robots at a future tick."""

    conflict_type: ConflictType
    timestep: int
    robot_a: str
    robot_b: str
    # For VERTEX conflicts, ``cell`` is the contested cell.
    # For EDGE conflicts, ``edge`` is the (from_a, to_a) edge A traverses;
    # the symmetric edge (to_a, from_a) is what B traverses.
    cell: Optional[CellCoord] = None
    edge: Optional[Tuple[CellCoord, CellCoord]] = None

    def involves(self, robot_id: str) -> bool:
        return robot_id in (self.robot_a, self.robot_b)

    def description(self) -> str:
        if self.conflict_type is ConflictType.VERTEX:
            return (
                f"VERTEX conflict at t={self.timestep}: "
                f"{self.robot_a} and {self.robot_b} both at {self.cell}"
            )
        a_from, a_to = self.edge or ((-1, -1), (-1, -1))
        return (
            f"EDGE conflict at t={self.timestep}: "
            f"{self.robot_a} swaps {a_from}->{a_to} with {self.robot_b}"
        )


class ConflictDetector:
    """Local conflict predictor.

    Parameters
        ``owner_id``:
            The id of the robot that owns this detector. Only
            conflicts that involve the owner are reported (so each
            robot only sees conflicts that affect it).
        ``reservations``:
            The owner's :class:`ReservationTable` (own + peer).
        ``lookahead_horizon``:
            How many future ticks to look at. ``current_tick`` itself
            is included; ``current_tick + horizon`` is included.
    """

    def __init__(
        self,
        owner_id: str,
        reservations: ReservationTable,
        lookahead_horizon: int = 10,
    ) -> None:
        self.owner_id = owner_id
        self.reservations = reservations
        self.lookahead_horizon = max(1, int(lookahead_horizon))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def predict(self, current_tick: int) -> List[PredictedConflict]:
        """Return all conflicts the owner can predict at ``current_tick``."""
        t_start = int(current_tick)
        t_end = int(current_tick) + self.lookahead_horizon

        reservations = self.reservations.reservations_in_range(t_start, t_end)
        if not reservations:
            return []

        vertex = self._detect_vertex(reservations, t_start, t_end)
        edge = self._detect_edge(reservations, t_start, t_end)

        return vertex + edge

    # ------------------------------------------------------------------
    # Vertex detection
    # ------------------------------------------------------------------
    def _detect_vertex(
        self,
        reservations: List[Reservation],
        t_start: int,
        t_end: int,
    ) -> List[PredictedConflict]:
        # Group reservations by (cell, timestep) -> set of robot ids.
        groups: Dict[Tuple[CellCoord, int], Set[str]] = {}
        for r in reservations:
            groups.setdefault((r.cell, r.timestep), set()).add(r.robot_id)

        conflicts: List[PredictedConflict] = []
        for (cell, t), robots in groups.items():
            if len(robots) < 2:
                continue
            if self.owner_id not in robots:
                # The owner is not affected; skip. This is what makes
                # the detector strictly local.
                continue
            others = sorted(robots - {self.owner_id})
            for other in others:
                conflicts.append(PredictedConflict(
                    conflict_type=ConflictType.VERTEX,
                    timestep=t,
                    robot_a=self.owner_id,
                    robot_b=other,
                    cell=cell,
                ))
        return conflicts

    # ------------------------------------------------------------------
    # Edge detection
    # ------------------------------------------------------------------
    def _detect_edge(
        self,
        reservations: List[Reservation],
        t_start: int,
        t_end: int,
    ) -> List[PredictedConflict]:
        # Build a position-at-tick map: tick -> {robot_id -> cell}.
        pos_at_t: Dict[int, Dict[str, CellCoord]] = {}
        for r in reservations:
            bucket = pos_at_t.setdefault(r.timestep, {})
            # First observation wins; subsequent entries for the same
            # (tick, robot) would only happen if the same robot
            # reserved two different cells at the same tick, which
            # should not occur with a consistent reservation model.
            bucket.setdefault(r.robot_id, r.cell)

        conflicts: List[PredictedConflict] = []
        seen: Set[Tuple[int, str, str]] = set()

        for t in range(t_start, t_end):
            t_next = t + 1
            if t_next > t_end:
                break
            at_t = pos_at_t.get(t, {})
            at_next = pos_at_t.get(t_next, {})
            if self.owner_id not in at_t or self.owner_id not in at_next:
                continue
            x = at_t[self.owner_id]
            y = at_next[self.owner_id]
            if x == y:
                # No actual move this tick; not a swap.
                continue
            for other, pos_other_t in at_t.items():
                if other == self.owner_id:
                    continue
                pos_other_next = at_next.get(other)
                if pos_other_next is None:
                    continue
                if pos_other_next == pos_other_t:
                    continue  # peer did not move this tick
                # Owner does (x -> y). Peer must do (y -> x).
                if pos_other_t == y and pos_other_next == x:
                    key = (t_next, self.owner_id, other)
                    if key in seen:
                        continue
                    seen.add(key)
                    conflicts.append(PredictedConflict(
                        conflict_type=ConflictType.EDGE,
                        timestep=t_next,
                        robot_a=self.owner_id,
                        robot_b=other,
                        edge=(x, y),
                    ))
        return conflicts