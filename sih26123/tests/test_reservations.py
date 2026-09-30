"""Tests for the Phase 2A reservation system."""

from __future__ import annotations

import pytest

from sih26123.coordination.reservations import Reservation, ReservationTable


def test_add_and_query_reservation() -> None:
    table = ReservationTable(owner_id="A")
    table.add((3, 4), 10, "A")
    assert table.get_robot_at((3, 4), 10) == "A"
    assert table.get_robot_at((3, 4), 11) is None


def test_reservation_can_be_overwritten() -> None:
    """Two robots reserving the same (cell, timestep) co-exist.

    ``get_robot_at`` returns an arbitrary single id; use
    :meth:`ReservationTable.robots_at` for the full set.
    """
    table = ReservationTable(owner_id="A")
    table.add((1, 1), 5, "A")
    table.add((1, 1), 5, "B")
    assert table.robots_at((1, 1), 5) == {"A", "B"}


def test_remove_reservation() -> None:
    table = ReservationTable(owner_id="A")
    table.add((1, 1), 5, "A")
    assert table.remove((1, 1), 5) is True
    assert table.get_robot_at((1, 1), 5) is None
    assert table.remove((1, 1), 5) is False


def test_clear_robot_keeps_others() -> None:
    table = ReservationTable(owner_id="A")
    table.add((0, 0), 0, "A")
    table.add((1, 0), 1, "A")
    table.add((1, 0), 1, "B")
    table.add((2, 0), 2, "A")
    removed = table.clear_robot("A")
    assert removed == 3  # A is present in three entries (one of them shared with B)
    assert table.get_robot_at((0, 0), 0) is None
    assert table.get_robot_at((1, 0), 1) == "B"
    assert table.get_robot_at((2, 0), 2) is None


def test_reset_clears_everything() -> None:
    table = ReservationTable(owner_id="A")
    table.add((0, 0), 0, "A")
    table.add((1, 0), 1, "B")
    table.reset()
    assert len(table) == 0


def test_update_own_for_path_builds_reservations() -> None:
    table = ReservationTable(owner_id="A", horizon=10)
    table.update_own_for_path(
        current_pos=(0, 0),
        path=[(1, 0), (2, 0), (3, 0)],
        start_tick=5,
        horizon=5,
    )
    assert table.get_robot_at((0, 0), 5) == "A"
    assert table.get_robot_at((1, 0), 6) == "A"
    assert table.get_robot_at((2, 0), 7) == "A"
    assert table.get_robot_at((3, 0), 8) == "A"
    # Pad with destination up to horizon.
    assert table.get_robot_at((3, 0), 9) == "A"
    assert table.get_robot_at((3, 0), 10) == "A"


def test_update_own_for_path_pads_when_path_shorter_than_horizon() -> None:
    table = ReservationTable(owner_id="A", horizon=6)
    table.update_own_for_path(
        current_pos=(0, 0),
        path=[(1, 0)],
        start_tick=10,
        horizon=6,
    )
    # At tick 10: at (0,0); tick 11: at (1,0); tick 12+: stay at (1,0).
    assert table.get_robot_at((0, 0), 10) == "A"
    assert table.get_robot_at((1, 0), 11) == "A"
    for k in range(12, 17):
        assert table.get_robot_at((1, 0), k) == "A"


def test_update_own_for_path_with_empty_path_pads_with_current_pos() -> None:
    table = ReservationTable(owner_id="A", horizon=3)
    table.update_own_for_path(
        current_pos=(2, 2),
        path=[],
        start_tick=0,
        horizon=3,
    )
    for k in range(0, 4):
        assert table.get_robot_at((2, 2), k) == "A"


def test_update_own_for_path_clears_previous_own() -> None:
    table = ReservationTable(owner_id="A", horizon=5)
    table.update_own_for_path(current_pos=(0, 0), path=[(1, 0)], start_tick=0, horizon=5)
    table.update_own_for_path(current_pos=(5, 5), path=[(6, 5), (7, 5)], start_tick=10, horizon=5)
    # Old reservations are gone.
    assert table.get_robot_at((0, 0), 0) is None
    assert table.get_robot_at((1, 0), 1) is None
    # New reservations present.
    assert table.get_robot_at((5, 5), 10) == "A"
    assert table.get_robot_at((6, 5), 11) == "A"
    assert table.get_robot_at((7, 5), 12) == "A"


def test_update_peer_from_state_uses_peer_timestamp() -> None:
    table = ReservationTable(owner_id="A", horizon=5)
    table.update_peer_from_state(
        peer_id="B",
        peer_position=(0, 0),
        peer_path=[(1, 0), (2, 0)],
        peer_timestamp=3,
        horizon=5,
    )
    # Broadcast at tick T means the peer is at peer_position starting
    # at tick T+1. The first path cell is reached at T+2.
    assert table.has_robot((0, 0), 4, "B")
    assert table.has_robot((1, 0), 5, "B")
    assert table.has_robot((2, 0), 6, "B")


def test_update_peer_from_state_clears_previous_peer() -> None:
    table = ReservationTable(owner_id="A", horizon=5)
    table.update_peer_from_state(peer_id="B", peer_position=(0, 0), peer_path=[(1, 0)],
                                  peer_timestamp=1, horizon=5)
    table.update_peer_from_state(peer_id="B", peer_position=(9, 9), peer_path=[],
                                  peer_timestamp=10, horizon=5)
    # Old peer reservations gone.
    assert table.get_robot_at((1, 0), 4) is None
    # New peer reservation present.
    assert table.has_robot((9, 9), 11, "B")


def test_same_cell_different_timestep_is_not_a_conflict() -> None:
    table = ReservationTable(owner_id="A", horizon=3)
    # Robot A reverses on itself (returns to (0,0) after reaching (1,0)).
    # At tick 0: at (0,0); tick 1: at (1,0); tick 2: at (0,0) again.
    table.update_own_for_path(current_pos=(0, 0), path=[(1, 0), (0, 0)], start_tick=0, horizon=3)
    assert table.has_robot((1, 0), 1, "A")
    assert table.has_robot((0, 0), 2, "A")
    # Tick 3 onwards is padded with the last cell (0, 0) up to horizon.
    matching = [r for r in table.all_reservations() if r.cell in {(0, 0), (1, 0)}]
    assert len(matching) == 4  # (0,0)@0, (1,0)@1, (0,0)@2, (0,0)@3


def test_two_robots_can_share_a_cell_at_same_tick() -> None:
    """Two robots both reserving the same cell at the same tick co-exist."""
    table = ReservationTable(owner_id="A", horizon=3)
    table.update_own_for_path(current_pos=(4, 5), path=[(5, 5)], start_tick=10, horizon=3)
    # B's broadcast at tick 9 reserves B's position at tick 10 and the
    # first path cell at tick 11 — colliding with A's planned (5, 5).
    table.update_peer_from_state(peer_id="B", peer_position=(5, 4), peer_path=[(5, 5)],
                                  peer_timestamp=9, horizon=3)
    bucket = table.robots_at((5, 5), 11)
    assert bucket == {"A", "B"}


def test_same_cell_same_timestep_is_a_conflict_in_storage() -> None:
    """Storage layer: both reservations co-exist and overlap."""
    table = ReservationTable(owner_id="A", horizon=5)
    # A: current (5,5) at tick 10, moves to (6,5) at tick 11.
    table.update_own_for_path(current_pos=(5, 5), path=[(6, 5)], start_tick=10, horizon=5)
    # B broadcast at tick 9 puts B at (5,5) at tick 10 and at (6,5) at tick 11.
    table.update_peer_from_state(peer_id="B", peer_position=(5, 5), peer_path=[(6, 5)],
                                  peer_timestamp=9, horizon=5)
    bucket_11 = table.robots_at((6, 5), 11)
    assert bucket_11 == {"A", "B"}


def test_reservations_in_range_filters_by_timestep() -> None:
    table = ReservationTable(owner_id="A", horizon=5)
    table.update_own_for_path(current_pos=(0, 0), path=[(1, 0), (2, 0)], start_tick=10, horizon=5)
    in_range = table.reservations_in_range(11, 12)
    cells = sorted(r.cell for r in in_range)
    assert cells == [(1, 0), (2, 0)]


def test_reservations_for_filters_by_robot() -> None:
    table = ReservationTable(owner_id="A", horizon=5)
    table.update_own_for_path(current_pos=(0, 0), path=[(1, 0)], start_tick=0, horizon=5)
    table.update_peer_from_state(peer_id="B", peer_position=(0, 0), peer_path=[(1, 0)],
                                  peer_timestamp=0, horizon=5)
    a_res = table.reservations_for("A")
    b_res = table.reservations_for("B")
    assert all(r.robot_id == "A" for r in a_res)
    assert all(r.robot_id == "B" for r in b_res)