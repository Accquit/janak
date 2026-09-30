"""Space-time reservations (Phase 2A).

Each robot maintains its own reservation table that contains:

* its **own** future reservations, derived from its current position
  and remaining path;
* **peer** reservations it has learned about through the message bus.

The table stores a mapping ``(cell, timestep) -> set(robot_id)`` so
that multiple robots competing for the same space-time cell are all
recorded (this is essential for vertex conflict detection).

Conventions
-----------
``Reservation(cell, timestep, robot_id)`` means *robot_id arrives at
cell at the end of tick timestep*. So the cell is occupied by that
robot at that tick.

When the reservation horizon exceeds the robot's known path length,
the remaining cells are padded with the last known cell (the
destination), which models an idle / done state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

CellCoord = Tuple[int, int]


@dataclass(frozen=True)
class Reservation:
    """A single space-time reservation."""

    cell: CellCoord
    timestep: int
    robot_id: str


class ReservationTable:
    """A robot's local view of all space-time reservations it knows about.

    Parameters
        ``owner_id``:
            The id of the robot that owns this table.
        ``horizon``:
            Default look-ahead horizon (in ticks) used by the helper
            methods that build reservations from a path.
    """

    def __init__(self, owner_id: str, horizon: int = 10) -> None:
        self.owner_id = owner_id
        self.horizon = max(1, int(horizon))
        # (cell, timestep) -> set of robot ids competing for it
        self._reservations: Dict[Tuple[CellCoord, int], Set[str]] = {}

    # ------------------------------------------------------------------
    # Low-level mutation
    # ------------------------------------------------------------------
    def add(self, cell: CellCoord, timestep: int, robot_id: str) -> None:
        """Add a single reservation, retaining any competing robots."""
        key = (tuple(cell), int(timestep))
        self._reservations.setdefault(key, set()).add(robot_id)

    def remove(self, cell: CellCoord, timestep: int, robot_id: Optional[str] = None) -> bool:
        """Remove a reservation.

        If ``robot_id`` is ``None`` the whole entry is dropped. Otherwise
        only that robot id is removed (other competitors stay).
        """
        key = (tuple(cell), int(timestep))
        bucket = self._reservations.get(key)
        if bucket is None:
            return False
        if robot_id is None:
            del self._reservations[key]
            return True
        bucket.discard(robot_id)
        if not bucket:
            del self._reservations[key]
            return True
        return True

    def clear_robot(self, robot_id: str) -> int:
        """Remove all reservations belonging to ``robot_id``.

        Returns the number of entries affected.
        """
        affected = 0
        for key in list(self._reservations.keys()):
            bucket = self._reservations[key]
            if robot_id in bucket:
                bucket.discard(robot_id)
                affected += 1
                if not bucket:
                    del self._reservations[key]
        return affected

    def reset(self) -> None:
        self._reservations.clear()

    # ------------------------------------------------------------------
    # Bulk rebuild helpers
    # ------------------------------------------------------------------
    def update_own_for_path(
        self,
        current_pos: CellCoord,
        path: Iterable[CellCoord],
        start_tick: int,
        horizon: Optional[int] = None,
    ) -> None:
        """Clear own reservations and rebuild from current state.

        The reservations produced (relative to ``start_tick``) are:

        * ``(current_pos, start_tick)``
        * ``(path[0], start_tick + 1)``
        * ``(path[1], start_tick + 2)``
        * ... up to the horizon. If the path is shorter, the
          destination cell is padded forward.
        """
        self.clear_robot(self.owner_id)
        self._reserve_trajectory(
            robot_id=self.owner_id,
            current_pos=tuple(current_pos),
            path=[tuple(c) for c in path],
            start_tick=int(start_tick),
            horizon=horizon if horizon is not None else self.horizon,
        )

    def update_peer_from_state(
        self,
        peer_id: str,
        peer_position: CellCoord,
        peer_path: Iterable[CellCoord],
        peer_timestamp: int,
        horizon: Optional[int] = None,
    ) -> None:
        """Clear and rebuild a peer's reservations from a state message.

        The ``peer_timestamp`` is the broadcast tick. After a peer
        broadcasts at tick T, its reported position is the *post-step*
        position, which is reached at the end of tick T. We therefore
        reserve ``peer_position`` at tick ``T + 1`` and the planned
        path cells at ``T + 2, T + 3, …``.
        """
        self.clear_robot(peer_id)
        self._reserve_trajectory(
            robot_id=peer_id,
            current_pos=tuple(peer_position),
            path=[tuple(c) for c in peer_path],
            start_tick=int(peer_timestamp) + 1,
            horizon=horizon if horizon is not None else self.horizon,
        )

    def reserve_stationary(
        self,
        robot_id: str,
        position: CellCoord,
        start_tick: int,
        horizon: Optional[int] = None,
    ) -> None:
        """Hold a peer's last known cell across the local safety horizon."""
        span = self.horizon if horizon is None else max(0, int(horizon))
        self.clear_robot(robot_id)
        for offset in range(span + 1):
            self.add(tuple(position), int(start_tick) + offset, robot_id)

    def _reserve_trajectory(
        self,
        robot_id: str,
        current_pos: CellCoord,
        path: List[CellCoord],
        start_tick: int,
        horizon: int,
    ) -> None:
        if horizon <= 0:
            return
        # Reservation at the current tick.
        key = (current_pos, start_tick)
        self._reservations.setdefault(key, set()).add(robot_id)
        if not path:
            # Pad with the current cell up to the horizon.
            for k in range(1, horizon + 1):
                key = (current_pos, start_tick + k)
                self._reservations.setdefault(key, set()).add(robot_id)
            return

        # First, the path cells.
        for i, cell in enumerate(path[:horizon]):
            key = (cell, start_tick + i + 1)
            self._reservations.setdefault(key, set()).add(robot_id)
        # Then pad with the last cell.
        last = tuple(path[-1])
        for k in range(len(path), horizon):
            key = (last, start_tick + k + 1)
            self._reservations.setdefault(key, set()).add(robot_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def robots_at(self, cell: CellCoord, timestep: int) -> Set[str]:
        """Return the set of robot ids that have reserved (cell, timestep)."""
        return set(self._reservations.get((tuple(cell), int(timestep)), set()))

    def has_robot(self, cell: CellCoord, timestep: int, robot_id: str) -> bool:
        return robot_id in self.robots_at(cell, timestep)

    def get_robot_at(self, cell: CellCoord, timestep: int) -> Optional[str]:
        """Return any robot id reserved at (cell, timestep), or None.

        Provided for back-compat / convenience; prefer
        :meth:`robots_at` when multiple robots may compete.
        """
        bucket = self._reservations.get((tuple(cell), int(timestep)))
        if not bucket:
            return None
        return next(iter(bucket))

    def all_reservations(self) -> List[Reservation]:
        out: List[Reservation] = []
        for (cell, t), bucket in self._reservations.items():
            for rid in bucket:
                out.append(Reservation(cell=cell, timestep=t, robot_id=rid))
        return out

    def reservations_in_range(self, t_start: int, t_end: int) -> List[Reservation]:
        out: List[Reservation] = []
        for (cell, t), bucket in self._reservations.items():
            if t_start <= t <= t_end:
                for rid in bucket:
                    out.append(Reservation(cell=cell, timestep=t, robot_id=rid))
        return out

    def reservations_for(self, robot_id: str) -> List[Reservation]:
        out: List[Reservation] = []
        for (cell, t), bucket in self._reservations.items():
            if robot_id in bucket:
                out.append(Reservation(cell=cell, timestep=t, robot_id=robot_id))
        return out

    def __len__(self) -> int:
        return len(self._reservations)

    def __contains__(self, key: Tuple[CellCoord, int]) -> bool:
        return (tuple(key[0]), int(key[1])) in self._reservations
