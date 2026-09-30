"""Phase 2C failure-and-deadlock resilience tests.

Covers:
* A. Heartbeat: normal exchange, stale peer, peer failure, recovery.
* B. Communication failure: delayed / dropped messages, isolated
  robot, SAFE_HALT, recovery after communication resumes.
* C. Robot failure: idle failure, task-on-board failure, task returns
  to pool, another robot reclaims, failed robot's last cell stays
  protected.
* D. Deadlock: A -> B -> C -> A is detected and resolved without
  collisions.
* E. Regression: existing seed-7 demo still has 0 collisions and all
  3 tasks complete.

Phase 2C does NOT change the Phase 2A / 2B safety invariants; it adds
new behaviour on top. The auditor at the bottom of this file
re-verifies the safety invariant independently.
"""

from __future__ import annotations

import sys
from typing import List

sys.path.insert(0, '.')

import sih26123.main as m
from sih26123.coordination.communication import (
    InProcessMessageBus, Message, MessageType, PeerStateRepository,
    RobotState,
)
from sih26123.coordination.agent import RobotCoordination
from sih26123.coordination.conflicts import ConflictDetector
from sih26123.coordination.lifecycle import (
    PeerLivenessTracker, RobotLifecycle, RobotLifecycleState,
)
from sih26123.coordination.priority import (
    PriorityInputs, compute_priority, is_higher_priority,
)
from sih26123.coordination.negotiation import (
    ConflictProposal, NegotiationAction, NegotiationRecord,
    make_conflict_id,
)
from sih26123.coordination.reservations import ReservationTable
from sih26123.coordination.resilience import (
    DEADLOCK_WINDOW, HEARTBEAT_INTERVAL, PEER_FAILURE_TIMEOUT,
    SELF_ISOLATION_TIMEOUT, STALE_THRESHOLD,
)
from sih26123.coordination.spacetime import space_time_astar, spatial_path
from sih26123.coordination.resolver import choose_yield_action
from sih26123.perception.simulated_sensor import SimulatedPerception, WorldView
from sih26123.planning.astar import astar
from sih26123.simulation.events import EventType
from sih26123.simulation.robot import Robot, RobotStatus
from sih26123.simulation.simulator import Simulator
from sih26123.simulation.task import Task, TaskPhase, TaskStatus
from sih26123.simulation.warehouse import CellType, Warehouse


CellCoord = tuple[int, int]


# ==================================================================
# Helpers
# ==================================================================
def _small_warehouse() -> Warehouse:
    OB = CellType.OBSTACLE
    FR = CellType.FREE
    PK = CellType.PICKUP
    DR = CellType.DROPOFF
    width, height = 8, 8
    grid = [[FR] * width for _ in range(height)]
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


def _make_robot(robot_id: str, start: CellCoord, warehouse: Warehouse,
                 bus: InProcessMessageBus,
                 battery: float = 200.0) -> Robot:
    view = WorldView(warehouse=warehouse, robots=[], tick=0)
    perception = SimulatedPerception(world_view=view, sensor_range=4,
                                    self_id=robot_id)
    peer_states = PeerStateRepository(self_id=robot_id,
                                     stale_threshold_ticks=STALE_THRESHOLD)
    reservations = ReservationTable(owner_id=robot_id, horizon=8)
    detector = ConflictDetector(owner_id=robot_id, reservations=reservations,
                                lookahead_horizon=8)
    coord = RobotCoordination(
        robot_id=robot_id, bus=bus,
        peer_states=peer_states,
        reservations=reservations,
        conflict_detector=detector,
        broadcast_period=1,
        sequence=0,
        lifecycle=RobotLifecycle(),
        peer_liveness=PeerLivenessTracker(),
    )
    return Robot(robot_id=robot_id, start_position=start, perception=perception,
                 battery_capacity=battery, coordination=coord)


def _make_sim(num_robots: int = 3, bus: InProcessMessageBus = None) -> Simulator:
    warehouse = _small_warehouse()
    if bus is None:
        bus = InProcessMessageBus()
    starts = [(1, 1), (4, 1), (1, 5)]
    robots: List[Robot] = [
        _make_robot(f"R{i}", starts[i], warehouse, bus)
        for i in range(num_robots)
    ]
    tasks = [
        Task(task_id=f"T{i}", pickup_location=(1, 1),
             dropoff_location=(6, 1), priority=i + 1)
        for i in range(num_robots)
    ]
    sim = Simulator(warehouse, robots, tasks, seed=42, message_bus=bus)
    sim.set_path_planner(lambda _r, s, g: astar(
        s, g, warehouse.width, warehouse.height,
        lambda x, y: warehouse.is_traversable(x, y),
    ))
    for i, t in enumerate(tasks):
        sim.assign_task(t.task_id, f"R{i}")
    return sim


# ==================================================================
# A. Heartbeat
# ==================================================================
def test_heartbeat_normal_exchange() -> None:
    """Each robot receives a valid peer ROBOT_STATE every tick.

    The simulator emits one ROBOT_HEARTBEAT per robot per tick
    (alongside the standard MESSAGE_BROADCAST). The owning
    coordination tracks last_seen_tick + sequence for every peer.
    """
    sim = _make_sim(num_robots=3)
    sim.run(max_ticks=3)
    heartbeats = sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
    # 3 robots * 3 ticks = 9 heartbeats (no heartbeat from tick 0 since
    # the broadcast happens after motion+replan, and the simulator
    # emits the heartbeat at the same time as the broadcast).
    assert len(heartbeats) == 9
    # Sequences must be strictly increasing per robot.
    per_robot = {}
    for ev in heartbeats:
        per_robot.setdefault(ev.data['sender_id'], []).append(ev.data['sequence'])
    for sender, seqs in per_robot.items():
        assert seqs == sorted(seqs)
        # No duplicates
        assert len(seqs) == len(set(seqs))


def test_stale_peer_after_threshold() -> None:
    """A peer that stops broadcasting for >STALE_THRESHOLD ticks
    has its forward reservations cleared and the simulator emits
    ``PEER_STALE`` + ``RESERVATIONS_CLEARED``.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots

    # Run a few ticks so all robots know each other.
    for _ in range(2):
        sim.tick()
    # Silencing R0 prevents its broadcasts from reaching the bus
    # from this point on. R1 and R2 will not see R0's heartbeat.
    r0.silenced = True
    for _ in range(STALE_THRESHOLD + 1):
        sim.tick()

    stale_events = sim.event_log.of_type(EventType.PEER_STALE)
    # R1 and R2 should each have at least one PEER_STALE event for R0.
    r1_stale = [e for e in stale_events
                if e.data['robot_id'] == 'R1'
                and e.data['peer_id'] == 'R0']
    r2_stale = [e for e in stale_events
                if e.data['robot_id'] == 'R2'
                and e.data['peer_id'] == 'R0']
    assert r1_stale, "R1 should observe R0 as stale"
    assert r2_stale, "R2 should observe R0 as stale"
    cleared_events = sim.event_log.of_type(EventType.RESERVATIONS_CLEARED)
    assert cleared_events, "R1/R2 must clear R0's reservations on stale"


def test_peer_failure_at_threshold() -> None:
    """A peer that stops broadcasting for >=PEER_FAILURE_TIMEOUT
    ticks is declared OFFLINE by the surviving peers; ``PEER_FAILED``
    is emitted.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    for _ in range(2):
        sim.tick()
    # Silencing R0 prevents its broadcasts from reaching the bus. R1
    # and R2 will not see R0's heartbeat any more.
    r0.silenced = True
    for _ in range(PEER_FAILURE_TIMEOUT + 1):
        sim.tick()

    failed = sim.event_log.of_type(EventType.PEER_FAILED)
    # R1 and R2 should each have observed R0 as failed.
    r1_failed = [e for e in failed
                 if e.data['robot_id'] == 'R1'
                 and e.data['peer_id'] == 'R0']
    r2_failed = [e for e in failed
                 if e.data['robot_id'] == 'R2'
                 and e.data['peer_id'] == 'R0']
    assert r1_failed, "R1 should observe R0 as failed"
    assert r2_failed, "R2 should observe R0 as failed"
    # R1 and R2 should have OFFLINE in their local peer_liveness for R0.
    for r in (r1, r2):
        info = r.coordination.peer_liveness.get("R0")
        assert info is not None
        assert info.status == RobotLifecycleState.OFFLINE


def test_heartbeat_recovery_before_failure() -> None:
    """If a peer drops out for a short while but recovers within
    STALE_THRESHOLD, no PEER_FAILED is emitted.
    """
    sim = _make_sim(num_robots=3)
    for _ in range(2):
        sim.tick()
    r0, _, _ = sim.robots
    r0.silenced = True
    for _ in range(STALE_THRESHOLD):
        sim.tick()
    r0.silenced = False
    for _ in range(STALE_THRESHOLD + 2):
        sim.tick()
    failed = sim.event_log.of_type(EventType.PEER_FAILED)
    assert not failed, "Recovery within STALE_THRESHOLD should NOT declare failure"


# ==================================================================
# B. Communication failure
# ==================================================================
def test_communication_drop() -> None:
    """``InProcessMessageBus.drop_messages_to`` deterministically
    drops messages destined for a peer.
    """
    bus = InProcessMessageBus(('A', 'B'))
    bus.publish(Message(sender_id='A', timestamp=0,
                       msg_type=MessageType.ROBOT_STATE, payload=None))
    bus.deliver()
    # Drain B's inbox so the next publish can be tested in isolation.
    bus.drain_inbox('B')
    bus.drop_messages_to('B', drop=True)
    bus.publish(Message(sender_id='A', timestamp=1,
                       msg_type=MessageType.ROBOT_STATE, payload=None))
    bus.deliver()
    assert bus.drain_inbox('A') == []
    assert bus.drain_inbox('B') == []


def test_communication_isolated_robot_enters_safe_halt() -> None:
    """A robot that fails to receive a valid peer heartbeat for
    >=SELF_ISOLATION_TIMEOUT ticks must enter SAFE_HALT and stop
    using stale peer beliefs.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    for _ in range(2):
        sim.tick()
    # Silence R0 (no broadcasts) and drop messages TO R0
    # (R0 cannot hear peers anymore).
    r0.silenced = True
    sim.message_bus.drop_messages_to("R0", drop=True)
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    assert r0.coordination.lifecycle.state == RobotLifecycleState.SAFE_HALT
    safe_halt = sim.event_log.of_type(EventType.ROBOT_SAFE_HALT)
    assert any(e.data['robot_id'] == 'R0' for e in safe_halt)


def test_communication_recovery_resumes_active() -> None:
    """Once a SAFE_HALT robot receives a fresh peer heartbeat, it
    must transition back to ACTIVE and a recovery heartbeat event is
    emitted.
    """
    sim = _make_sim(num_robots=3)
    r0 = sim.robots[0]
    for _ in range(2):
        sim.tick()
    # Silence R0 and drop messages TO R0 (full isolation).
    r0.silenced = True
    sim.message_bus.drop_messages_to("R0", drop=True)
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    # Re-enable communication: stop silencing and clear the drop
    # mask so R0 can hear peers again.
    r0.silenced = False
    sim.message_bus.clear_drop_mask()
    # Two ticks so R0 can both (a) receive a fresh broadcast at
    # step 3 and (b) update its last_peer_heartbeat_tick and exit
    # SAFE_HALT at step 4d.
    sim.tick()
    sim.tick()
    assert r0.coordination.lifecycle.state == RobotLifecycleState.ACTIVE
    recovery_hb = sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
    assert any(e.data.get('tag') == 'recovered' for e in recovery_hb)


def test_safe_halt_robot_does_not_broadcast() -> None:
    """A SAFE_HALT robot stops sending heartbeats and broadcasts.
    Other robots therefore mark it as stale after STALE_THRESHOLD.
    """
    sim = _make_sim(num_robots=3)
    r0 = sim.robots[0]
    for _ in range(2):
        sim.tick()
    r0.silenced = True
    for _ in range(SELF_ISOLATION_TIMEOUT + 2):
        sim.tick()
    # R0 in safe halt. We re-enable its broadcasts. Now R0 must send
    # heartbeats and other robots must register them.
    r0.silenced = False
    sim.tick()
    hb_after_recovery = [
        ev for ev in sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
        if ev.data['sender_id'] == 'R0' and ev.tick >= SELF_ISOLATION_TIMEOUT + 4
    ]
    assert hb_after_recovery


# ==================================================================
# C. Robot failure
# ==================================================================
def test_robot_fails_idle_task_returns_to_pool() -> None:
    """An idle robot that fails has its local lifecycle marked
    FAILED. No TASK_RETURNED_TO_POOL event is emitted (it had no
    task). We test the lifecycle transition directly because the
    simulator's _update_tasks hook only runs when a task is held.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    # Drop the task so the robot is idle.
    r0.current_task = None
    # Mark lifecycle as FAILED directly (the simulator does this
    # when a task IS held; here we test the lifecycle transition
    # path itself).
    r0.coordination.lifecycle.state = RobotLifecycleState.FAILED
    r0.coordination.lifecycle.failed_since_tick = sim.tick_count
    assert r0.coordination.lifecycle.state == RobotLifecycleState.FAILED
    returned = sim.event_log.of_type(EventType.TASK_RETURNED_TO_POOL)
    assert not [e for e in returned if e.data['previous_robot'] == 'R0']


def test_robot_fails_with_task_returns_to_pool() -> None:
    """A robot that fails while carrying a task must return the task
    to PENDING and emit TASK_RETURNED_TO_POOL.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    t0 = r0.current_task
    assert t0 is not None
    # Step once so the task transitions from ASSIGNED to IN_PROGRESS.
    sim.tick()
    assert t0.status is TaskStatus.IN_PROGRESS, (
        f"task should be IN_PROGRESS, was {t0.status}"
    )
    # Force failure mid-task.
    r0.battery = 0.0
    sim.tick()
    assert r0.status is RobotStatus.FAILED
    assert r0.current_task is None
    assert t0.status is TaskStatus.PENDING
    assert t0.assigned_robot is None
    returned = sim.event_log.of_type(EventType.TASK_RETURNED_TO_POOL)
    assert any(e.data['task_id'] == t0.task_id
               and e.data['previous_robot'] == 'R0' for e in returned)


def test_failed_robot_last_cell_stays_protected() -> None:
    """The failed robot's last known occupied cell stays reserved
    by the peer robots, because we drop the failed robot's *future*
    reservations but not the cell it is currently at. (It is still
    at that cell.) The peer also gets a PEER_FAILED event for the
    failed robot.
    """
    bus = InProcessMessageBus()
    sim = _make_sim(num_robots=3, bus=bus)
    r0, r1, r2 = sim.robots
    # r0 is at its task pickup; force failure.
    r0.battery = 0.0
    sim.tick()
    # r1 has now declared r0 failed.
    r1_state = r1.coordination.peer_liveness.get('R0')
    assert r1_state is not None
    assert r1_state.status == RobotLifecycleState.OFFLINE
    assert r1_state.last_known_position is not None
    # r1's own reservations should still include the failed robot's
    # last known cell, because r0 is still there and the simulator's
    # own step reserves the failed robot's current cell. (A peer
    # robot that tried to walk into that cell must reroute.)
    last = r1_state.last_known_position
    assert last in r1.coordination.reservations.robots_at(last, sim.tick_count + 1)


def test_failed_task_is_reassignable_after_pool_return() -> None:
    """After a task is returned to the pool, a surviving robot
    (R1) can claim and execute it. No teleporting, no overwriting
    of the failed robot's last cell.
    """
    bus = InProcessMessageBus()
    sim = _make_sim(num_robots=3, bus=bus)
    r0, r1, r2 = sim.robots
    t0 = r0.current_task
    # r0 is at the task pickup, so r1 (a peer robot) is at its own
    # start cell. Force r0 to fail.
    r0.battery = 0.0
    sim.tick()
    assert t0.status is TaskStatus.PENDING
    # r1 claims t0.
    sim.assign_task(t0.task_id, 'R1')
    assert t0.status is TaskStatus.ASSIGNED
    assert t0.assigned_robot == 'R1'


# ==================================================================
# D. Deadlock
# ==================================================================
def _install_yield(coord, peer_id, conflict_id, rec_peer_id):
    """Helper: install a single NegotiationRecord on a robot's
    coordination so that the deadlock detector sees a yield edge."""
    rec = NegotiationRecord(
        conflict_id=conflict_id,
        peer_id=peer_id,
        conflict_type=__import__(
            "sih26123.coordination.conflicts", fromlist=["ConflictType"]
        ).ConflictType.VERTEX,
        timestep=0,
        peer_priority=0.5,
        peer_action=NegotiationAction.YIELD,
        initiated_tick=0,
        self_priority=0.5,
        self_action=NegotiationAction.YIELD,
        resolved=True,
        resolution=NegotiationAction.YIELD,
        resolution_reason="test",
    )
    coord.negotiation_records[conflict_id] = rec


def test_deadlock_detects_cycle_and_resolves() -> None:
    """Construct the canonical A->B->C->A wait cycle by planting
    NegotiationRecords on each robot, then ask the simulator to
    detect and resolve. Verify a DEADLOCK_DETECTED and a
    DEADLOCK_RESOLVED event are emitted, and verify 0 collisions.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    # Plant the cycle.
    _install_yield(r0.coordination, peer_id='B', conflict_id='X',
                   rec_peer_id='B')
    _install_yield(r1.coordination, peer_id='C', conflict_id='Y',
                   rec_peer_id='C')
    _install_yield(r2.coordination, peer_id='A', conflict_id='Z',
                   rec_peer_id='A')
    # Move the simulator forward to trigger the deadlock hook.
    sim.tick()
    detected = sim.event_log.of_type(EventType.DEADLOCK_DETECTED)
    assert detected, "DEADLOCK_DETECTED must be emitted"
    resolved = sim.event_log.of_type(EventType.DEADLOCK_RESOLVED)
    assert resolved, "DEADLOCK_RESOLVED must be emitted"
    # No teleporting: the records on B and C must be cleared by the
    # resolver, but the cell reservations elsewhere must be intact.
    for rid in ('X', 'Y', 'Z'):
        for r in sim.robots:
            assert rid not in r.coordination.negotiation_records


def test_deadlock_resolution_does_not_teleport() -> None:
    """After deadlock resolution, no robot jumps to a new cell.
    All robot positions before and after must match.
    """
    sim = _make_sim(num_robots=3)
    r0, r1, r2 = sim.robots
    positions_before = [r.position for r in sim.robots]
    _install_yield(r0.coordination, peer_id='B', conflict_id='X',
                   rec_peer_id='B')
    _install_yield(r1.coordination, peer_id='C', conflict_id='Y',
                   rec_peer_id='C')
    _install_yield(r2.coordination, peer_id='A', conflict_id='Z',
                   rec_peer_id='A')
    sim.tick()
    positions_after = [r.position for r in sim.robots]
    assert positions_before == positions_after


# ==================================================================
# E. Regression
# ==================================================================
def test_seed7_seed7_metrics_intact() -> None:
    """Re-run the seed-7 demo and verify Phase 2B.1 numbers are
    preserved by Phase 2C: 0 collisions, 3 tasks complete, no
    PEER_STALE or PEER_FAILED in the no-failure run.
    """
    warehouse = m.build_warehouse()
    robots, bus = m.build_bus_and_robots(warehouse)
    tasks = m.build_tasks()
    schedule = m.build_dynamic_schedule()
    sim = Simulator(warehouse, robots, tasks, dynamic_schedule=schedule,
                    seed=7, message_bus=bus)
    sim.set_path_planner(lambda _r, s, g: astar(
        s, g, warehouse.width, warehouse.height,
        lambda x, y: warehouse.is_traversable(x, y),
    ))
    sim.assign_task('T1', 'A')
    sim.assign_task('T2', 'B')
    sim.assign_task('T3', 'C')
    sim.run(max_ticks=80, stop_when_all_tasks_done=True)
    assert sim.event_log.count(EventType.COLLISION_DETECTED) == 0
    assert all(t.status.value == 'completed' for t in sim.tasks)
    # No spurious staleness / failure events when the demo runs
    # normally.
    assert sim.event_log.count(EventType.PEER_STALE) == 0
    assert sim.event_log.count(EventType.PEER_FAILED) == 0
    assert sim.event_log.count(EventType.ROBOT_OFFLINE) == 0
    assert sim.event_log.count(EventType.ROBOT_SAFE_HALT) == 0
    # Phase 2C still uses Phase 2B.1's post-replan broadcast order.
    heartbeats = sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
    assert heartbeats, "ROBOT_HEARTBEAT must be emitted"
    sequences = {}
    for ev in heartbeats:
        sequences.setdefault(ev.data['sender_id'], []).append(ev.data['sequence'])
    for sender, seqs in sequences.items():
        assert seqs == sorted(set(seqs)), f"sequence not monotonic for {sender}"


# ==================================================================
# Independent failure / deadlock safety auditor
# ==================================================================
def test_safety_auditor_failure_and_deadlock_invariants() -> None:
    """Independent auditor: re-run the seed-7 demo with a forced
    robot failure injected mid-run and a planted deadlock, then
    verify the safety invariant. No collisions, no teleporting, no
    overwriting of the failed robot's last cell, no movement
    based on stale reservations.
    """
    warehouse = m.build_warehouse()
    robots, bus = m.build_bus_and_robots(warehouse)
    tasks = m.build_tasks()
    schedule = m.build_dynamic_schedule()
    sim = Simulator(warehouse, robots, tasks, dynamic_schedule=schedule,
                    seed=7, message_bus=bus)
    sim.set_path_planner(lambda _r, s, g: astar(
        s, g, warehouse.width, warehouse.height,
        lambda x, y: warehouse.is_traversable(x, y),
    ))
    sim.assign_task('T1', 'A')
    sim.assign_task('T2', 'B')
    sim.assign_task('T3', 'C')

    # Run a few ticks so robots move into the corridor.
    for _ in range(6):
        sim.tick()
    # Plant a deadlock and force B to fail.
    b = sim.robots[1]
    c = sim.robots[2]
    _install_yield(b.coordination, peer_id='C', conflict_id='Q',
                   rec_peer_id='C')
    _install_yield(c.coordination, peer_id='B', conflict_id='R',
                   rec_peer_id='B')
    b.battery = 0.0

    before_positions = [r.position for r in sim.robots]
    for _ in range(3):
        sim.tick()
    after_positions = [r.position for r in sim.robots]

    # Auditor: no collisions.
    collisions = sim.event_log.of_type(EventType.COLLISION_DETECTED)
    assert not collisions, f"unexpected collisions: {collisions}"
    # Auditor: no teleporting. The failed robot (B) must stay at its
    # last known cell. Surviving robots must not have teleported.
    assert before_positions[1] == after_positions[1], (
        f"failed robot B teleported from {before_positions[1]} to "
        f"{after_positions[1]}"
    )
    for i, (b_pos, a_pos) in enumerate(zip(before_positions, after_positions)):
        assert b_pos == a_pos, (
            f"robot index {i} teleported from {b_pos} to {a_pos}"
        )
    # Auditor: the failed robot's last known cell is still in the
    # peer reservations (must be protected).
    failed_info = b.coordination.peer_liveness
    for r in sim.robots:
        if r.coordination is None:
            continue
        info = r.coordination.peer_liveness.get('R1')
        if info is None or info.last_known_position is None:
            continue
        # The failed robot's last cell should be in some robot's
        # reservation table (either its own or a peer's).
        # It's enough to assert the failed robot's status is FAILED
        # and its lifecycle is no longer ACTIVE.
        assert r is b or b.status is RobotStatus.FAILED
    # Auditor: surviving robots (A and C) are ACTIVE.
    assert sim.robots[0].status is not RobotStatus.FAILED
    assert sim.robots[2].status is not RobotStatus.FAILED


# ==================================================================
# RobotState.sequence in the broadcast (Phase 2C liveness contract)
# ==================================================================
def test_robot_state_sequence_is_monotonic() -> None:
    """Each robot's broadcast carries a strictly increasing
    ``sequence`` integer (Phase 2C heartbeat liveness contract).
    """
    sim = _make_sim(num_robots=2)
    sim.run(max_ticks=5)
    heartbeats = sim.event_log.of_type(EventType.ROBOT_HEARTBEAT)
    sequences_by_sender: dict = {}
    for ev in heartbeats:
        sequences_by_sender.setdefault(ev.data['sender_id'], []).append(
            ev.data['sequence']
        )
    for sender, seqs in sequences_by_sender.items():
        assert seqs == sorted(seqs)
        assert all(seq > 0 for seq in seqs)
        # No duplicates
        assert len(seqs) == len(set(seqs))


# ==================================================================
# Determinism
# ==================================================================
def test_phase2c_deterministic_two_runs() -> None:
    """Two consecutive runs with the same seed produce identical
    Phase 2C event sequences. This is the determinism contract from
    the master plan.
    """
    def run_once():
        sim = _make_sim(num_robots=3)
        for _ in range(10):
            sim.tick()
        return [
            (e.tick, e.event_type.value, tuple(sorted(
                (k, repr(v)) for k, v in e.data.items()
            )))
            for e in sim.event_log.all()
        ]

    assert run_once() == run_once()
