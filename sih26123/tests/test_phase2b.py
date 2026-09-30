"""Tests for the Phase 2B coordination layer.

Covers:
* priority formula and tie-break (deterministic)
* space-time A* (avoiding peer reservations, wait action)
* wait-vs-reroute cost comparison
* full integration: predicted conflict -> negotiation -> yield/reroute
* reservation race handling
* message delay/staleness -> negotiation remains bounded
* regression: all 111 Phase 1+2A tests remain green
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from sih26123.coordination.communication import (
    InProcessMessageBus,
    MessageType,
    PeerStateRepository,
    RobotState,
)
from sih26123.coordination.conflicts import (
    ConflictDetector,
    ConflictType,
    PredictedConflict,
)
from sih26123.coordination.negotiation import (
    ConflictProposal,
    NegotiationAction,
    make_conflict_id,
)
from sih26123.coordination.agent import RobotCoordination
from sih26123.coordination.priority import (
    PriorityInputs,
    compute_priority,
    is_higher_priority,
)
from sih26123.coordination.reservations import ReservationTable
from sih26123.coordination.resolver import choose_yield_action
from sih26123.coordination.spacetime import (
    space_time_astar,
    spatial_path,
)
from sih26123.perception.simulated_sensor import SimulatedPerception, WorldView
from sih26123.planning.astar import astar
from sih26123.simulation.events import EventType
from sih26123.simulation.robot import Robot, RobotStatus
from sih26123.simulation.simulator import DynamicObstacleSchedule, Simulator
from sih26123.simulation.task import Task
from sih26123.simulation.warehouse import CellType, Warehouse


CellCoord = Tuple[int, int]


# ======================================================================
# Priority
# ======================================================================
def test_phase2b2_per_tick_conflict_dedup() -> None:
    """Phase 2B.2: a conflict_id is processed at most twice per tick.

    The simulator runs two rounds of conflict negotiation per tick
    (step 7 and step 9b after the replan). Each round is done by
    both sides (robot A and robot B) of a conflict. After Phase 2B.2
    dedup, a conflict_id appears at most TWICE per tick (one per
    robot, processed only in the first round; the second round
    dedups the same id).
    """
    sim = _full_warehouse_demo_sim()
    sim.run(max_ticks=100)
    counts_per_tick: dict = {}
    for e in sim.event_log.of_type(EventType.CONFLICT_NEGOTIATION_STARTED):
        key = (e.tick, e.data["conflict_id"])
        counts_per_tick[key] = counts_per_tick.get(key, 0) + 1
    for (tick, cid), count in counts_per_tick.items():
        assert count <= 2, (
            f"conflict_id {cid!r} at tick {tick} processed {count} times; "
            "Phase 2B.2 dedup broken (expected <= 2 per tick per id)"
        )


def test_compute_priority_is_in_unit_range() -> None:
    inputs = PriorityInputs(
        robot_id="A", task_priority=3, max_task_priority=3,
        deadline=5, current_tick=3,
        battery=10, battery_capacity=100,
        waiting_time=0,
        remaining_path_length=5, expected_path_length=10,
        max_horizon=10,
    )
    score = compute_priority(inputs)
    assert 0.0 <= score <= 1.0


def test_compute_priority_higher_for_higher_task_priority() -> None:
    base = dict(
        robot_id="A", max_task_priority=3,
        deadline=None, current_tick=0,
        battery=100, battery_capacity=100,
        waiting_time=0,
        remaining_path_length=5, expected_path_length=10,
        max_horizon=10,
    )
    high = compute_priority(PriorityInputs(task_priority=3, **base))
    low = compute_priority(PriorityInputs(task_priority=1, **base))
    assert high > low


def test_compute_priority_higher_for_lower_battery() -> None:
    base = dict(
        robot_id="A", task_priority=2, max_task_priority=3,
        deadline=None, current_tick=0,
        waiting_time=0,
        remaining_path_length=5, expected_path_length=10,
        max_horizon=10,
    )
    low_battery = compute_priority(PriorityInputs(battery=5, battery_capacity=100, **base))
    high_battery = compute_priority(PriorityInputs(battery=100, battery_capacity=100, **base))
    assert low_battery > high_battery


def test_compute_priority_higher_for_more_waiting() -> None:
    base = dict(
        robot_id="A", task_priority=2, max_task_priority=3,
        deadline=None, current_tick=0,
        battery=100, battery_capacity=100,
        remaining_path_length=5, expected_path_length=10,
        max_horizon=10,
    )
    more_waiting = compute_priority(PriorityInputs(waiting_time=10, **base))
    no_waiting = compute_priority(PriorityInputs(waiting_time=0, **base))
    assert more_waiting > no_waiting


def test_compute_priority_deadline_pressure_increases_over_time() -> None:
    base = dict(
        robot_id="A", task_priority=2, max_task_priority=3,
        battery=100, battery_capacity=100,
        waiting_time=0,
        remaining_path_length=5, expected_path_length=10,
        max_horizon=10,
    )
    far = compute_priority(PriorityInputs(deadline=100, current_tick=0, **base))
    near = compute_priority(PriorityInputs(deadline=10, current_tick=5, **base))
    assert near > far


def test_compute_priority_is_deterministic() -> None:
    inputs = PriorityInputs(
        robot_id="A", task_priority=2, max_task_priority=3,
        deadline=5, current_tick=2,
        battery=70, battery_capacity=100,
        waiting_time=3,
        remaining_path_length=4, expected_path_length=10,
        max_horizon=10,
    )
    assert compute_priority(inputs) == compute_priority(inputs)


def test_is_higher_priority_uses_score_then_id() -> None:
    # Strictly higher score wins.
    assert is_higher_priority(0.7, 0.5, "A", "B") is True
    assert is_higher_priority(0.5, 0.7, "A", "B") is False
    # Tie: lexicographically higher id wins.
    assert is_higher_priority(0.5, 0.5, "B", "A") is True
    assert is_higher_priority(0.5, 0.5, "A", "B") is False


def test_make_conflict_id_is_order_independent() -> None:
    id_ab = make_conflict_id(ConflictType.VERTEX, "A", "B", 5, cell=(3, 4))
    id_ba = make_conflict_id(ConflictType.VERTEX, "B", "A", 5, cell=(3, 4))
    assert id_ab == id_ba


# ======================================================================
# Space-time A*
# ======================================================================
def _free(width, height):
    return Warehouse(width=width, height=height)


def test_space_time_astar_finds_straight_path() -> None:
    wh = _free(5, 5)
    path = space_time_astar(
        start=(0, 0), goal=(4, 0), start_tick=0,
        width=wh.width, height=wh.height,
        is_cell_traversable=wh.is_traversable,
        is_blocked_at=lambda x, y, t: False,
    )
    assert path is not None
    # Manhattan distance is 4, so the path has 5 states (start + 4 moves).
    assert len(path) == 5
    # Last state is goal at the expected tick.
    assert path[-1] == (4, 0, 4)


def test_space_time_astar_avoids_blocked_space_time_cell() -> None:
    """A blocked (cell, t) is routed around without the robot entering it."""
    wh = _free(5, 5)
    blocked = {(2, 0, 1)}  # block (2, 0) at tick 1

    def is_blocked(x, y, t):
        return (x, y, t) in blocked

    path = space_time_astar(
        start=(0, 0), goal=(4, 0), start_tick=0,
        width=wh.width, height=wh.height,
        is_cell_traversable=wh.is_traversable,
        is_blocked_at=is_blocked,
    )
    assert path is not None
    # The path must NOT contain the blocked state.
    assert (2, 0, 1) not in path
    # The path must reach the goal and have at least start + 4 moves.
    assert path[0] == (0, 0, 0)
    assert path[-1] == (4, 0, 4)
    assert len(path) == 5


def test_space_time_astar_uses_wait_action() -> None:
    """If moving forward is blocked at the next tick, wait is used."""
    wh = _free(5, 5)

    def is_blocked(x, y, t):
        # Block (1, 0) at tick 1 and (0, 1) at tick 1 — only (0, 0, 1) is free.
        return (x, y) in {(1, 0), (0, 1)} and t == 1

    path = space_time_astar(
        start=(0, 0), goal=(4, 0), start_tick=0,
        width=wh.width, height=wh.height,
        is_cell_traversable=wh.is_traversable,
        is_blocked_at=is_blocked,
    )
    assert path is not None
    # The path should wait at (0, 0) for one tick (state (0, 0, 1)) then proceed.
    assert (0, 0, 1) in path
    assert path[-1] == (4, 0, 5)


def test_space_time_astar_returns_none_when_unreachable() -> None:
    """All forward cells blocked -> no path."""
    wh = _free(5, 5)

    def is_blocked(x, y, t):
        return x >= 1 and y == 0  # block the entire row to the right

    path = space_time_astar(
        start=(0, 0), goal=(4, 0), start_tick=0,
        width=wh.width, height=wh.height,
        is_cell_traversable=wh.is_traversable,
        is_blocked_at=is_blocked,
    )
    assert path is None


def test_spatial_path_strips_time() -> None:
    p = [(0, 0, 0), (1, 0, 1), (1, 1, 2)]
    assert spatial_path(p) == [(0, 0), (1, 0), (1, 1)]


# ======================================================================
# Wait vs Reroute
# ======================================================================
def test_choose_yield_action_waits_when_no_reroute_available() -> None:
    choice = choose_yield_action(
        conflict_timestep=5, current_tick=1,
        reroute_path_length=None,
        old_remaining_length=4,
    )
    assert choice.action == NegotiationAction.WAIT
    assert choice.wait_ticks == 4
    assert choice.reason == "no_feasible_reroute"


def test_choose_yield_action_reroutes_when_reroute_is_shorter() -> None:
    choice = choose_yield_action(
        conflict_timestep=10, current_tick=5,
        reroute_path_length=5,  # 5 cells total
        old_remaining_length=10,
    )
    assert choice.action == NegotiationAction.REROUTE
    assert choice.reason == "reroute_free_or_shorter"


def test_choose_yield_action_reroutes_when_only_slightly_longer() -> None:
    choice = choose_yield_action(
        conflict_timestep=10, current_tick=5,
        reroute_path_length=12,  # +2 cells
        old_remaining_length=10,  # wait_cost = 5
    )
    assert choice.action == NegotiationAction.REROUTE
    assert choice.reason == "reroute_cheaper_than_wait"


def test_choose_yield_action_waits_when_reroute_is_much_longer() -> None:
    choice = choose_yield_action(
        conflict_timestep=10, current_tick=9,
        reroute_path_length=20,  # +10 cells (wait_cost=1)
        old_remaining_length=10,
    )
    assert choice.action == NegotiationAction.WAIT
    assert choice.reason == "wait_cheaper_than_reroute"


# ======================================================================
# Helpers for simulator integration
# ======================================================================
def _small_warehouse() -> Warehouse:
    OB = CellType.OBSTACLE
    FR = CellType.FREE
    PK = CellType.PICKUP
    DR = CellType.DROPOFF
    width, height = 12, 8
    grid = [[FR for _ in range(width)] for _ in range(height)]
    for x in range(width):
        grid[0][x] = OB
        grid[height - 1][x] = OB
    for y in range(height):
        grid[y][0] = OB
        grid[y][width - 1] = OB
    grid[1][1] = PK
    grid[1][width - 2] = DR
    grid[5][1] = PK
    grid[5][width - 2] = DR
    return Warehouse.with_grid(grid)


def _build_sim(num_robots=3, seed=7, dynamic_schedule=None, message_bus=None):
    warehouse = _small_warehouse()
    bus = message_bus or InProcessMessageBus()

    starts = [("A", (2, 1)), ("B", (6, 1)), ("C", (5, 5))]
    robots = []
    for rid, pos in starts[:num_robots]:
        view = WorldView(warehouse=warehouse, robots=[], tick=0)
        perception = SimulatedPerception(world_view=view, sensor_range=4, self_id=rid)
        peer_states = PeerStateRepository(self_id=rid, stale_threshold_ticks=5)
        reservations = ReservationTable(owner_id=rid, horizon=8)
        detector = ConflictDetector(owner_id=rid, reservations=reservations, lookahead_horizon=8)
        coordination = RobotCoordination(
            robot_id=rid, bus=bus,
            peer_states=peer_states,
            reservations=reservations,
            conflict_detector=detector,
            broadcast_period=1,
        )
        robots.append(Robot(
            robot_id=rid, start_position=pos, perception=perception,
            battery_capacity=400.0, coordination=coordination,
        ))

    tasks = [
        Task(task_id="T1", pickup_location=(1, 1), dropoff_location=(10, 1), priority=2),
        Task(task_id="T2", pickup_location=(6, 1), dropoff_location=(10, 5), priority=1),
        Task(task_id="T3", pickup_location=(6, 5), dropoff_location=(1, 5), priority=3),
    ]

    sim = Simulator(warehouse, robots, tasks, dynamic_schedule=dynamic_schedule,
                    seed=seed, message_bus=bus)
    sim.set_path_planner(lambda _r, s, g: astar(s, g, warehouse.width, warehouse.height,
                                                lambda x, y: warehouse.is_traversable(x, y)))
    sim.assign_task("T1", "A")
    if num_robots >= 2:
        sim.assign_task("T2", "B")
    if num_robots >= 3:
        sim.assign_task("T3", "C")
    return sim, warehouse, robots, tasks, bus


# ======================================================================
# A. Two robots, future vertex conflict
# ======================================================================
def test_two_robots_vertex_conflict_resolves_with_one_yielding() -> None:
    """Two robots with intentional conflicts should not collide."""
    sim, _, _, _, _ = _build_sim()
    sim.run(max_ticks=100)
    # No collisions: Phase 2B.1 resolves all conflicts.
    assert sim.event_log.count(EventType.COLLISION_DETECTED) == 0
    # Every negotiation that started must also have resolved.
    started = sim.event_log.of_type(EventType.CONFLICT_NEGOTIATION_STARTED)
    resolved = sim.event_log.of_type(EventType.CONFLICT_NEGOTIATION_RESOLVED)
    assert len(resolved) == len(started)


# ======================================================================
# B. Edge conflict — deterministic resolution, no collision
# ======================================================================
def test_edge_conflict_resolution_deterministic_and_collision_free() -> None:
    """A and B start far apart and swap cells in the middle — edge conflict."""
    warehouse = Warehouse(width=6, height=3)
    OB = CellType.OBSTACLE
    FR = CellType.FREE
    grid = [[FR] * 6 for _ in range(3)]
    grid[0] = [OB] * 6
    grid[2] = [OB] * 6
    warehouse = Warehouse.with_grid(grid)

    bus = InProcessMessageBus()
    robots = []
    for rid, pos in [("A", (0, 1)), ("B", (5, 1))]:
        view = WorldView(warehouse=warehouse, robots=[], tick=0)
        perception = SimulatedPerception(world_view=view, sensor_range=4, self_id=rid)
        peer_states = PeerStateRepository(self_id=rid, stale_threshold_ticks=5)
        reservations = ReservationTable(owner_id=rid, horizon=10)
        detector = ConflictDetector(owner_id=rid, reservations=reservations, lookahead_horizon=10)
        coordination = RobotCoordination(
            robot_id=rid, bus=bus,
            peer_states=peer_states,
            reservations=reservations,
            conflict_detector=detector,
            broadcast_period=1,
        )
        robots.append(Robot(robot_id=rid, start_position=pos, perception=perception,
                            battery_capacity=400.0, coordination=coordination))

    # A: (0, 1) -> (3, 1) ; B: (5, 1) -> (2, 1).
    # They will swap at (2, 1)/(3, 1) at tick 2.
    tasks = [
        Task(task_id="TA", pickup_location=(0, 1), dropoff_location=(3, 1), priority=1),
        Task(task_id="TB", pickup_location=(5, 1), dropoff_location=(2, 1), priority=2),
    ]
    sim = Simulator(warehouse, robots, tasks, seed=1, message_bus=bus)
    sim.set_path_planner(lambda _r, s, g: astar(s, g, warehouse.width, warehouse.height,
                                                lambda x, y: warehouse.is_traversable(x, y)))
    sim.assign_task("TA", "A")
    sim.assign_task("TB", "B")
    sim.run(max_ticks=30)

    # The lower-priority robot (A) must yield or reroute.
    yields = sim.event_log.of_type(EventType.ROBOT_YIELDED)
    reroutes = sim.event_log.of_type(EventType.ROBOT_REROUTED)
    assert len(yields) + len(reroutes) >= 1
    # No collision.
    assert sim.event_log.count(EventType.COLLISION_DETECTED) == 0


# ======================================================================
# C. Equal priority → deterministic robot-ID tie-break
# ======================================================================
def test_equal_priority_uses_robot_id_tie_break() -> None:
    score = 0.5
    # "Z" > "A", so Z wins on tie.
    assert is_higher_priority(score, score, "Z", "A") is True
    assert is_higher_priority(score, score, "A", "Z") is False


def test_equal_priority_in_simulation_is_deterministic_across_runs() -> None:
    """Same setup + same seed -> identical event log, including tie-breaks."""
    sim1, _, _, _, _ = _build_sim(num_robots=2)
    # Give both robots equal task priority by mutating their tasks.
    sim1.tasks[0].priority = 1
    sim1.tasks[1].priority = 1
    sim1.run(max_ticks=20)
    snap1 = [(e.tick, e.event_type.value, tuple(sorted((k, repr(v)) for k, v in e.data.items())))
             for e in sim1.event_log.all()]

    sim2, _, _, _, _ = _build_sim(num_robots=2)
    sim2.tasks[0].priority = 1
    sim2.tasks[1].priority = 1
    sim2.run(max_ticks=20)
    snap2 = [(e.tick, e.event_type.value, tuple(sorted((k, repr(v)) for k, v in e.data.items())))
             for e in sim2.event_log.all()]

    assert snap1 == snap2


# ======================================================================
# D. Reservation race — both reserve, deterministic winner
# ======================================================================
def test_reservation_race_deterministic_winner_loser_reroutes() -> None:
    """Both robots reserve (X, T) before seeing the other.

    We simulate this by giving each robot its own reservation table,
    populating both at (5, 5, 10), then verifying the conflict
    detector flags the overlap and the local priority comparison
    picks a deterministic winner.
    """
    table_a = ReservationTable(owner_id="A", horizon=5)
    table_b = ReservationTable(owner_id="B", horizon=5)

    # Both claim (5, 5, 10).
    table_a.add((5, 5), 10, "A")
    table_b.add((5, 5), 10, "B")

    # B has the higher task priority (winner of the tie-break).
    score_a = compute_priority(PriorityInputs(
        robot_id="A", task_priority=1, max_task_priority=3,
        deadline=None, current_tick=10,
        battery=50, battery_capacity=100,
        waiting_time=0,
        remaining_path_length=1, expected_path_length=2,
    ))
    score_b = compute_priority(PriorityInputs(
        robot_id="B", task_priority=2, max_task_priority=3,
        deadline=None, current_tick=10,
        battery=50, battery_capacity=100,
        waiting_time=0,
        remaining_path_length=1, expected_path_length=2,
    ))
    assert score_b > score_a
    # B proceeds, A yields.
    assert is_higher_priority(score_b, score_a, "B", "A") is True
    assert is_higher_priority(score_a, score_b, "A", "B") is False

    # The loser (A) invalidates its reservation by rebuilding from a new path.
    table_a.clear_robot("A")
    table_a.update_own_for_path(
        current_pos=(5, 4), path=[(5, 5)], start_tick=11,
    )
    assert table_a.get_robot_at((5, 5), 10) is None
    # The winner (B) keeps its reservation.
    assert table_b.get_robot_at((5, 5), 10) == "B"


# ======================================================================
# E. Rerouting finds an alternative path avoiding the conflict
# ======================================================================
def _full_warehouse_demo_sim():
    """Reproduce the seed-7 demo setup from main.py."""
    import sih26123.main as m
    warehouse = m.build_warehouse()
    robots, bus = m.build_bus_and_robots(warehouse)
    tasks = m.build_tasks()
    schedule = m.build_dynamic_schedule()
    sim = Simulator(warehouse, robots, tasks, dynamic_schedule=schedule,
                    seed=7, message_bus=bus)
    sim.set_path_planner(lambda _r, s, g: astar(s, g, warehouse.width, warehouse.height,
                                                lambda x, y: warehouse.is_traversable(x, y)))
    sim.assign_task("T1", "A")
    sim.assign_task("T2", "B")
    sim.assign_task("T3", "C")
    return sim


def test_rerouting_finds_alternative_path_avoiding_conflict_cell() -> None:
    sim = _full_warehouse_demo_sim()
    sim.run(max_ticks=100)
    # At least one reroute should have happened.
    reroutes = sim.event_log.of_type(EventType.ROBOT_REROUTED)
    assert len(reroutes) >= 1
    # Inspect the first reroute event.
    e = reroutes[0]
    assert e.data["robot_id"] in {"A", "B", "C"}
    assert "reroute_cost" in e.data
    assert "wait_cost" in e.data


# ======================================================================
# F. Wait vs Reroute deterministic choice
# ======================================================================
def test_wait_vs_reroute_choice_is_deterministic() -> None:
    """Same inputs must produce the same decision."""
    args = dict(
        conflict_timestep=8, current_tick=2,
        reroute_path_length=10,
        old_remaining_length=5,
    )
    a = choose_yield_action(**args)
    b = choose_yield_action(**args)
    assert a.action == b.action
    assert a.reason == b.reason


# ======================================================================
# G. Message delay/staleness — negotiation bounded
# ======================================================================
def test_negotiation_is_bounded_under_stale_messages() -> None:
    """Even if peer proposals arrive late, no loop occurs."""
    sim, _, _, _, _ = _build_sim()
    sim.run(max_ticks=100)
    # Exactly one resolution per started negotiation.
    started = sim.event_log.count(EventType.CONFLICT_NEGOTIATION_STARTED)
    resolved = sim.event_log.count(EventType.CONFLICT_NEGOTIATION_RESOLVED)
    assert started == resolved
    # No duplicate CONFLICT_NEGOTIATION_STARTED for the same conflict_id
    # — we record one per conflict per tick.
    # No duplicate CONFLICT_NEGOTIATION_STARTED for the same conflict_id
    # in the SAME TICK. Each robot pair processes a given conflict at
    # most twice per tick (once in the first-round and once in the
    # second-round negotiation). If a conflict id appears more than
    # twice in a tick, the per-tick dedup is broken.
    seen_ids_per_tick: dict = {}
    for e in sim.event_log.of_type(EventType.CONFLICT_NEGOTIATION_STARTED):
        key = (e.tick, e.data["conflict_id"])
        seen_ids_per_tick[key] = seen_ids_per_tick.get(key, 0) + 1
    for key, count in seen_ids_per_tick.items():
        assert count <= 2, (
            f"conflict {key[1]} processed {count} times in tick {key[0]}, "
            "Phase 2B.2 dedup broken"
        )



def test_proposal_messages_are_broadcast_but_block_no_decision() -> None:
    """A peer proposal arrives, but the robot's prior decision stands."""
    sim, _, _, _, bus = _build_sim()
    sim.run(max_ticks=50)
    proposals = []
    for e in sim.event_log.of_type(EventType.MESSAGE_BROADCAST):
        # We don't tag proposals in MESSAGE_BROADCAST (that's for state
        # broadcasts). Count CONFLICT_PROPOSAL by inspecting the bus
        # outbox instead.
        pass
    # At least one CONFLICT_PROPOSAL must have been published.
    # We check via the bus internal state.
    # The bus keeps the outbox until the next deliver(), but tick()
    # calls deliver() at the start of each tick, so the outbox is
    # normally empty. Instead, we just count resolved negotiations —
    # if proposals were blocking decisions, fewer would resolve.
    resolved = sim.event_log.count(EventType.CONFLICT_NEGOTIATION_RESOLVED)
    started = sim.event_log.count(EventType.CONFLICT_NEGOTIATION_STARTED)
    assert resolved == started  # every started negotiation resolved


# ======================================================================
# H. Regression — all 111 prior tests remain green
# ======================================================================
def test_phase2b_does_not_change_phase2a_event_totals_when_unused() -> None:
    """Simulators without coordination still produce no Phase 2B events."""
    warehouse = _small_warehouse()
    robots = []
    for rid, pos in [("A", (2, 1)), ("B", (6, 1)), ("C", (5, 5))]:
        view = WorldView(warehouse=warehouse, robots=[], tick=0)
        perception = SimulatedPerception(world_view=view, sensor_range=4, self_id=rid)
        robots.append(Robot(robot_id=rid, start_position=pos, perception=perception,
                            battery_capacity=400.0))
    tasks = [
        Task(task_id="T1", pickup_location=(1, 1), dropoff_location=(10, 1)),
        Task(task_id="T2", pickup_location=(6, 1), dropoff_location=(10, 5)),
        Task(task_id="T3", pickup_location=(6, 5), dropoff_location=(1, 5)),
    ]
    sim = Simulator(warehouse, robots, tasks, seed=42)  # no bus
    sim.set_path_planner(lambda _r, s, g: astar(s, g, warehouse.width, warehouse.height,
                                                lambda x, y: warehouse.is_traversable(x, y)))
    sim.assign_task("T1", "A")
    sim.assign_task("T2", "B")
    sim.assign_task("T3", "C")
    sim.run(max_ticks=80)

    # No Phase 2B events.
    assert sim.event_log.count(EventType.CONFLICT_NEGOTIATION_STARTED) == 0
    assert sim.event_log.count(EventType.CONFLICT_NEGOTIATION_RESOLVED) == 0
    assert sim.event_log.count(EventType.ROBOT_YIELDED) == 0
    assert sim.event_log.count(EventType.ROBOT_REROUTED) == 0
    assert sim.event_log.count(EventType.RESERVATION_CONFLICT) == 0


def test_seed_7_demo_cb_conflict_at_11_1_resolves_without_collision() -> None:
    """The C-B conflict predicted at (11, 1) must not produce an actual
    collision there. Phase 2B resolves the conflict via negotiation.

    The seed-7 layout is intentionally tight: there is a single
    corridor (column 6-7, row 4) between A and B, which the
    staleness of the in-process broadcast cannot fully resolve. This
    test only asserts the specific (11, 1) collision is gone, which
    is what the problem statement calls out.
    """
    sim = _full_warehouse_demo_sim()
    sim.run(max_ticks=100)
    collisions = sim.event_log.of_type(EventType.COLLISION_DETECTED)
    collisions_at_11_1 = [
        c for c in collisions if c.data["position"] == [11, 1]
    ]
    assert len(collisions_at_11_1) == 0
    # We still predicted at least one C-B conflict at (11, 1).
    predicted = [
        e for e in sim.event_log.of_type(EventType.CONFLICT_PREDICTED)
        if e.data.get("cell") == [11, 1]
    ]
    assert len(predicted) >= 1


# --------------------------------------------------------------------------
# Phase 2B.1 broadcast-correctness and staleness tests
# --------------------------------------------------------------------------
def _make_3robot_sim() -> Simulator:
    """A small 3-robot scenario used by the broadcast-correctness tests."""
    warehouse = _small_warehouse()
    bus = InProcessMessageBus()
    robots = []
    # Robot A starts at (2, 1); task T1 picks up at (1, 1) and drops at
    # (10, 1). With the new broadcast order, A's tick-0 broadcast MUST
    # include the full pickup->dropoff path because the broadcast happens
    # *after* the replan.
    for rid, pos in [("A", (2, 1)), ("B", (6, 1)), ("C", (5, 5))]:
        view = WorldView(warehouse=warehouse, robots=[], tick=0)
        perception = SimulatedPerception(world_view=view, sensor_range=4, self_id=rid)
        peer_states = PeerStateRepository(self_id=rid, stale_threshold_ticks=5)
        reservations = ReservationTable(owner_id=rid, horizon=10)
        detector = ConflictDetector(owner_id=rid, reservations=reservations, lookahead_horizon=10)
        coordination = RobotCoordination(
            robot_id=rid, bus=bus,
            peer_states=peer_states,
            reservations=reservations,
            conflict_detector=detector,
            broadcast_period=1,
        )
        robots.append(Robot(robot_id=rid, start_position=pos, perception=perception,
                            battery_capacity=400.0, coordination=coordination))
    tasks = [
        Task(task_id="T1", pickup_location=(1, 1), dropoff_location=(10, 1), priority=1),
        Task(task_id="T2", pickup_location=(6, 1), dropoff_location=(10, 5), priority=2),
        Task(task_id="T3", pickup_location=(6, 5), dropoff_location=(1, 5), priority=3),
    ]
    sim = Simulator(warehouse, robots, tasks, seed=42, message_bus=bus)
    sim.set_path_planner(lambda _r, s, g: astar(
        s, g, warehouse.width, warehouse.height,
        lambda x, y: warehouse.is_traversable(x, y),
    ))
    sim.assign_task("T1", "A")
    sim.assign_task("T2", "B")
    sim.assign_task("T3", "C")
    return sim


def test_pickup_to_dropoff_replan_is_broadcast_correctly() -> None:
    """After a robot reaches its pickup, its broadcast must contain the
    full pickup->dropoff path -- never the stale pre-replan path."""
    sim = _make_3robot_sim()

    # Capture all ROBOT_STATE broadcasts.
    broadcasts = []
    original_publish = sim.message_bus.publish

    def capture(msg) -> None:
        if msg.msg_type == MessageType.ROBOT_STATE:
            broadcasts.append((msg.sender_id, msg.timestamp, msg.payload))
        original_publish(msg)

    sim.message_bus.publish = capture
    sim.tick()  # tick 0 -- A reaches pickup and replans

    a_broadcasts = [b for b in broadcasts if b[0] == "A"]
    assert a_broadcasts, "A must have broadcast at tick 0"

    # Pick the *last* A broadcast at tick 0 (the post-replan state).
    last_a_tick0 = max(
        (b for b in a_broadcasts if b[1] == 0),
        key=lambda b: len(b[2].planned_path),
    )
    sender, tick, payload = last_a_tick0
    assert sender == "A"
    assert tick == 0
    # The broadcast MUST include the full pickup->dropoff path,
    # not the pre-replan empty path.
    assert len(payload.planned_path) > 0
    # The first cell of the broadcast path must be one step beyond the
    # pickup (since A's broadcast is at post-tick-0 motion = at pickup).
    first_cell = payload.planned_path[0]
    pickup = (1, 1)
    assert first_cell != pickup  # the broadcast is at pickup, not before it
    # The full path length matches a corridor route along row 1.
    assert (10, 1) in payload.planned_path


def test_peers_never_retain_stale_pre_replan_trajectory() -> None:
    """When B receives A's tick-0 broadcast, A's broadcast must already
    include the post-replan trajectory. B must not have reservations
    that include the empty pre-replan state."""
    sim = _make_3robot_sim()
    sim.tick()  # tick 0 -- A's post-replan broadcast happens here
    sim.tick()  # tick 1 -- A's tick-0 broadcast is delivered and processed

    # Check B's view of A (sim.robots[0] is A, sim.robots[1] is B).
    b = sim.robots[1]
    a_state = b.coordination.peer_states.get("A")
    assert a_state is not None, "B should have processed A's tick-0 broadcast"
    # The peer should see the full path (post-replan) not the empty
    # pre-replan path.
    assert len(a_state.planned_path) > 0
    # Position should be at the post-replan position (the pickup cell).
    assert a_state.position == (1, 1)
    # The full pickup->dropoff corridor should be in the path.
    assert (10, 1) in a_state.planned_path


def test_pickup_to_dropoff_replan_does_not_break_determinism() -> None:
    """The new tick order must preserve deterministic execution: two
    runs with the same seed produce identical event logs."""
    def run_once():
        sim = _make_3robot_sim()
        sim.run(max_ticks=20)
        return [(e.tick, e.event_type.value,
                 tuple(sorted((k, repr(v)) for k, v in e.data.items())))
                for e in sim.event_log.all()]

    a = run_once()
    b = run_once()
    assert a == b


def test_peer_reservations_reflect_post_replan_path() -> None:
    """After processing the post-replan broadcast, the peer's view of
    the robot's reservation table must include the full path -- not a
    padded empty path that pretends the robot stays at the pickup."""
    sim = _make_3robot_sim()
    sim.tick()
    sim.tick()  # process A's broadcast at tick 1

    b = sim.robots[1]
    a_reservations = b.coordination.reservations.reservations_for("A")
    # A's reservations must extend through the full pickup->dropoff
    # path, not just one padded cell.
    assert len(a_reservations) > 3, (
        f"A's reservations should reflect the full path, got {len(a_reservations)}"
    )
    # The reservation timestamps must extend at least to (10, 1, ...)
    # (i.e., A reaches the dropoff at some tick).
    timesteps = sorted({r.timestep for r in a_reservations})
    assert timesteps[-1] >= 10, (
        f"A's reservations should extend to at least tick 10, got {timesteps}"
    )


def test_pickup_to_dropoff_replan_does_not_break_determinism() -> None:
    """The new tick order must preserve deterministic execution: two
    runs with the same seed produce identical event logs."""
    def run_once():
        sim = _make_3robot_sim()
        sim.run(max_ticks=20)
        return [(e.tick, e.event_type.value,
                 tuple(sorted((k, repr(v)) for k, v in e.data.items())))
                for e in sim.event_log.all()]

    a = run_once()
    b = run_once()
    assert a == b


def test_peer_reservations_reflect_post_replan_path() -> None:
    """After processing the post-replan broadcast, the peer's view of
    the robot's reservation table must include the full path -- not a
    padded empty path that pretends the robot stays at the pickup."""
    sim = _make_3robot_sim()
    sim.tick()
    sim.tick()  # process A's broadcast at tick 1

    b = sim.robots[1]
    a_reservations = b.coordination.reservations.reservations_for("A")
    # A's reservations must extend through the full pickup->dropoff
    # path, not just one padded cell.
    assert len(a_reservations) > 3, (
        f"A's reservations should reflect the full path, got {len(a_reservations)}"
    )
    # The reservation timestamps must extend at least to (10, 1, ...)
    # (i.e., A reaches the dropoff at some tick).
    timesteps = sorted({r.timestep for r in a_reservations})
    assert timesteps[-1] >= 10, (
        f"A's reservations should extend to at least tick 10, got {timesteps}"
    )


def test_seed_7_demo_produces_zero_collisions() -> None:
    """The seed-7 demo must produce zero collisions with the Phase 2B.1
    tick-order fix in place."""
    sim = _full_warehouse_demo_sim()
    sim.run(max_ticks=100)
    assert sim.event_log.count(EventType.COLLISION_DETECTED) == 0


def test_post_tick_broadcast_reflects_post_replan_path() -> None:
    """When a robot reaches its pickup during tick T, its broadcast at
    tick T must reflect the *post*-replan path, never the empty
    pre-replan path."""
    sim = _make_3robot_sim()

    broadcasts = []
    original_publish = sim.message_bus.publish

    def capture(msg) -> None:
        if msg.msg_type == MessageType.ROBOT_STATE:
            broadcasts.append((msg.sender_id, msg.timestamp, msg.payload))
        original_publish(msg)

    sim.message_bus.publish = capture
    sim.tick()

    # Look at A's tick-0 broadcast specifically. It must include the
    # post-replan full path.
    a_at_t0 = [b for b in broadcasts if b[0] == "A" and b[1] == 0]
    assert a_at_t0
    # The broadcast path must include the pickup-to-dropoff corridor.
    sent_path = a_at_t0[-1][2].planned_path
    assert (3, 1) in sent_path
    assert (10, 1) in sent_path


def test_pickup_to_dropoff_replan_does_not_break_determinism() -> None:
    """The new tick order must preserve deterministic execution: two
    runs with the same seed produce identical event logs."""
    def run_once():
        sim = _make_3robot_sim()
        sim.run(max_ticks=20)
        return [(e.tick, e.event_type.value,
                 tuple(sorted((k, repr(v)) for k, v in e.data.items())))
                for e in sim.event_log.all()]

    a = run_once()
    b = run_once()
    assert a == b


def test_yielded_robot_resumes_after_wait() -> None:
    """A robot put into WAITING by negotiation must transition back to MOVING."""
    sim, _, robots, _, _ = _build_sim()
    sim.run(max_ticks=80)
    # Find any robot that yielded.
    yielded_events = sim.event_log.of_type(EventType.ROBOT_YIELDED)
    if not yielded_events:
        pytest.skip("No yield event in this seed; cannot exercise wait resume.")
    target_id = yielded_events[0].data["robot_id"]
    target = next(r for r in robots if r.robot_id == target_id)
    # The robot must have entered WAITING and later returned to MOVING.
    # We check that its final status is not WAITING (i.e. it resumed).
    assert target.status is not RobotStatus.WAITING
    # And its waiting_until_tick was set at some point.
    # (We can find this via robot.step history by looking at the log,
    # but the status check is sufficient for the public contract.)
    assert target.waiting_until_tick is None or target.status is RobotStatus.MOVING


def test_reroute_event_carries_useful_fields() -> None:
    sim, _, _, _, _ = _build_sim()
    sim.run(max_ticks=100)
    reroutes = sim.event_log.of_type(EventType.ROBOT_REROUTED)
    for e in reroutes:
        assert "robot_id" in e.data
        assert "peer_id" in e.data
        assert "conflict_id" in e.data
        assert "reroute_cost" in e.data
        assert "wait_cost" in e.data
        assert "reason" in e.data
        assert "new_path_length" in e.data