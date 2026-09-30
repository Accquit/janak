"""Tests for the Phase 2A conflict detector."""

from __future__ import annotations

import pytest

from sih26123.coordination.conflicts import (
    ConflictDetector,
    ConflictType,
    PredictedConflict,
)
from sih26123.coordination.reservations import ReservationTable


def _make_tables():
    """Convenience: build a detector + table for owner A."""
    table = ReservationTable(owner_id="A", horizon=10)
    detector = ConflictDetector(owner_id="A", reservations=table, lookahead_horizon=10)
    return table, detector


def test_no_reservations_no_conflicts() -> None:
    _, detector = _make_tables()
    assert detector.predict(current_tick=0) == []


def test_no_conflict_when_paths_do_not_overlap() -> None:
    table, detector = _make_tables()
    # A goes (0,0)->(1,0); B goes (5,5)->(6,5). No overlap.
    table.update_own_for_path(current_pos=(0, 0), path=[(1, 0), (2, 0), (3, 0)], start_tick=0)
    table.update_peer_from_state(
        peer_id="B",
        peer_position=(5, 5),
        peer_path=[(6, 5), (7, 5), (8, 5)],
        peer_timestamp=0,
    )
    assert detector.predict(current_tick=0) == []


def test_vertex_conflict_detected() -> None:
    table, detector = _make_tables()
    # Both robots target (5, 5) at tick 11; A keeps moving afterwards
    # so there is only one vertex conflict.
    table.update_own_for_path(
        current_pos=(4, 5),
        path=[(5, 5), (6, 5), (7, 5), (8, 5), (9, 5)],
        start_tick=10,
    )
    # B's broadcast was at tick 9, so B's position is reserved at tick 10
    # and its first path cell at tick 11.
    table.update_peer_from_state(
        peer_id="B",
        peer_position=(5, 4),
        peer_path=[(5, 5)],
        peer_timestamp=9,
    )
    conflicts = detector.predict(current_tick=10)
    vertex = [c for c in conflicts if c.conflict_type is ConflictType.VERTEX]
    assert len(vertex) == 1
    c = vertex[0]
    assert c.cell == (5, 5)
    assert c.timestep == 11
    assert c.robot_a == "A"
    assert c.robot_b == "B"


def test_same_cell_different_timestep_is_not_a_conflict() -> None:
    table, detector = _make_tables()
    table.update_own_for_path(current_pos=(5, 5), path=[(6, 5), (7, 5), (8, 5)], start_tick=10)
    # B's broadcast at tick 12 means B is at (5,5) starting at tick 13
    # and at (6,5) starting at tick 14. A has already left (5,5) by
    # tick 11. No overlap.
    table.update_peer_from_state(
        peer_id="B",
        peer_position=(5, 5),
        peer_path=[(6, 5), (7, 5)],
        peer_timestamp=12,
    )
    assert detector.predict(current_tick=10) == []


def test_edge_conflict_swap_detected() -> None:
    """A swaps cells with B in the same tick → edge conflict."""
    table, detector = _make_tables()
    # A: (5,5) -> (6,5) -> ... so A does not stay at (6,5)
    table.update_own_for_path(
        current_pos=(5, 5), path=[(6, 5), (7, 5), (8, 5)], start_tick=10,
    )
    # B's broadcast at tick 9 puts B at (6,5) at tick 10 and at (5,5) at tick 11.
    # So during tick 11, A moves 5,5 -> 6,5 and B moves 6,5 -> 5,5 → swap.
    table.update_peer_from_state(
        peer_id="B",
        peer_position=(6, 5),
        peer_path=[(5, 5)],
        peer_timestamp=9,
    )
    conflicts = detector.predict(current_tick=10)
    edge = [c for c in conflicts if c.conflict_type is ConflictType.EDGE]
    assert len(edge) == 1
    c = edge[0]
    assert c.timestep == 11
    assert c.edge == ((5, 5), (6, 5))
    assert c.robot_a == "A"
    assert c.robot_b == "B"


def test_non_swapping_movement_is_not_an_edge_conflict() -> None:
    """Two robots moving in the same direction must not flag an edge conflict."""
    table, detector = _make_tables()
    # A: (5,5) -> (6,5) at tick 11
    table.update_own_for_path(current_pos=(5, 5), path=[(6, 5)], start_tick=10)
    # B broadcast at tick 9 so B is at (5,5) at tick 10 and at (6,5) at tick 11.
    # Both robots go 5,5 -> 6,5 in the same tick: same direction, vertex conflict.
    table.update_peer_from_state(
        peer_id="B",
        peer_position=(5, 5),
        peer_path=[(6, 5)],
        peer_timestamp=9,
    )
    conflicts = detector.predict(current_tick=10)
    edge = [c for c in conflicts if c.conflict_type is ConflictType.EDGE]
    assert edge == []
    # But there IS a vertex conflict at (6, 5) at tick 11.
    vertex = [c for c in conflicts if c.conflict_type is ConflictType.VERTEX]
    assert any(c.cell == (6, 5) and c.timestep == 11 for c in vertex)


def test_conflict_outside_horizon_ignored() -> None:
    table, detector = _make_tables()
    # Both robots arrive at (9, 9) at tick 25; the horizon is 10 so the
    # detector at tick 24 looks at [24, 34] — but the conflict is at
    # tick 25 which IS inside the horizon. To make the conflict fall
    # outside, push it to tick 35.
    table.update_own_for_path(current_pos=(8, 9), path=[(9, 9)], start_tick=34)
    table.update_peer_from_state(
        peer_id="B",
        peer_position=(9, 8),
        peer_path=[(9, 9)],
        peer_timestamp=34,
    )
    conflicts = detector.predict(current_tick=24)
    assert conflicts == []


def test_conflict_just_inside_horizon_detected() -> None:
    table = ReservationTable(owner_id="A", horizon=5)
    detector = ConflictDetector(owner_id="A", reservations=table, lookahead_horizon=5)
    table.update_own_for_path(current_pos=(8, 9), path=[(9, 9)], start_tick=24)
    table.update_peer_from_state(
        peer_id="B",
        peer_position=(9, 8),
        peer_path=[(9, 9)],
        peer_timestamp=24,
    )
    # Both arrive at (9, 9) at tick 25 (A from start_tick+1, B from
    # peer_timestamp+2). Inside horizon [24, 29].
    conflicts = detector.predict(current_tick=24)
    assert any(c.conflict_type is ConflictType.VERTEX for c in conflicts)


def test_no_false_positive_when_paths_cross_at_different_times() -> None:
    table, detector = _make_tables()
    # A: (5,5) -> (6,5) -> (7,5) -> (8,5); B arrives at (6,5) at tick 13.
    # A has moved past (6,5) by tick 13, so no conflict.
    table.update_own_for_path(current_pos=(5, 5), path=[(6, 5), (7, 5), (8, 5)], start_tick=10)
    table.update_peer_from_state(
        peer_id="B", peer_position=(5, 5), peer_path=[(6, 5)],
        peer_timestamp=12,
    )
    assert detector.predict(current_tick=10) == []


def test_detector_only_reports_conflicts_involving_owner() -> None:
    """Conflicts that do NOT involve the owner must not be reported."""
    table = ReservationTable(owner_id="X", horizon=5)
    detector = ConflictDetector(owner_id="X", reservations=table, lookahead_horizon=5)
    # A and B conflict (vertex at (3,3)) — owner X is idle.
    table.update_peer_from_state(
        peer_id="A", peer_position=(2, 3), peer_path=[(3, 3)], peer_timestamp=10,
    )
    table.update_peer_from_state(
        peer_id="B", peer_position=(3, 2), peer_path=[(3, 3)], peer_timestamp=10,
    )
    # The owner's reservations are empty — they are at X's own position only.
    table.update_own_for_path(current_pos=(0, 0), path=[], start_tick=10)
    assert detector.predict(current_tick=10) == []


def test_predicted_conflict_description_is_informative() -> None:
    c_vertex = PredictedConflict(
        conflict_type=ConflictType.VERTEX,
        timestep=11,
        robot_a="A",
        robot_b="B",
        cell=(5, 5),
    )
    c_edge = PredictedConflict(
        conflict_type=ConflictType.EDGE,
        timestep=12,
        robot_a="A",
        robot_b="B",
        edge=((5, 5), (6, 5)),
    )
    assert "VERTEX" in c_vertex.description()
    assert "(5, 5)" in c_vertex.description()
    assert "EDGE" in c_edge.description()
    assert "swap" in c_edge.description().lower()


def test_multiple_conflicts_detected() -> None:
    """Two robots with two overlapping timesteps should produce two flags."""
    table3 = ReservationTable(owner_id="A", horizon=5)
    det3 = ConflictDetector(owner_id="A", reservations=table3, lookahead_horizon=5)
    # A goes (0,0) -> (1,0) -> (2,0) -> (3,0) (so it doesn't pad at (2,0)).
    # B broadcast at tick 9 puts B at (0,1) at tick 10 and at (1,0) at tick 11,
    # then at (2,0) at tick 12. Both target (1,0) at tick 11 and (2,0) at tick 12.
    table3.update_own_for_path(
        current_pos=(0, 0), path=[(1, 0), (2, 0), (3, 0)], start_tick=10,
    )
    table3.update_peer_from_state(
        peer_id="B", peer_position=(0, 1), peer_path=[(1, 0), (2, 0)],
        peer_timestamp=9,
    )
    conflicts = det3.predict(current_tick=10)
    vertex = [c for c in conflicts if c.conflict_type is ConflictType.VERTEX]
    timesteps = sorted(c.timestep for c in vertex)
    assert timesteps == [11, 12]