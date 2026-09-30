"""Deterministic demonstration of the SIH26123 simulation foundation.

Running this module prints a tick-by-tick log of a three-robot
warehouse run and (optionally) renders an ASCII map each tick.

Example:
    python main.py

The demo is fully deterministic: same seed in -> same log out.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

from sih26123.coordination.agent import RobotCoordination
from sih26123.coordination.communication import (
    InProcessMessageBus,
    PeerStateRepository,
)
from sih26123.coordination.conflicts import ConflictDetector
from sih26123.coordination.reservations import ReservationTable
from sih26123.perception.simulated_sensor import SimulatedPerception, WorldView
from sih26123.planning.astar import astar
from sih26123.simulation.events import EventType
from sih26123.simulation.robot import Robot, RobotStatus
from sih26123.simulation.simulator import DynamicObstacleSchedule, Simulator
from sih26123.simulation.task import Task, TaskPhase, TaskStatus
from sih26123.simulation.warehouse import CellType, Warehouse


CellCoord = Tuple[int, int]


def build_warehouse() -> Warehouse:
    """Build a small but realistic warehouse layout.

    The layout is 14 columns by 8 rows. Two parallel aisles are
    separated by a center wall. There are two pickup cells on the left
    and two drop-off cells on the right, plus a charging bay in the
    bottom-right.
    """
    OB = CellType.OBSTACLE
    PK = CellType.PICKUP
    DR = CellType.DROPOFF
    CH = CellType.CHARGING
    FR = CellType.FREE

    width, height = 14, 8
    grid = [[FR for _ in range(width)] for _ in range(height)]

    # Outer walls.
    for x in range(width):
        grid[0][x] = OB
        grid[height - 1][x] = OB
    for y in range(height):
        grid[y][0] = OB
        grid[y][width - 1] = OB

    # Central divider wall (columns 6 and 7) with a single opening at row 4.
    for y in range(1, height - 1):
        grid[y][6] = OB
        grid[y][7] = OB
    grid[4][6] = FR
    grid[4][7] = FR

    # Pickup cells on the left aisle.
    grid[1][2] = PK
    grid[5][2] = PK
    # Drop-off cells on the right aisle.
    grid[1][11] = DR
    grid[5][11] = DR
    # Charging bay (still traversable).
    grid[6][11] = CH

    return Warehouse.with_grid(grid)


def build_bus_and_robots(warehouse: Warehouse) -> Tuple[List[Robot], InProcessMessageBus]:
    """Create three robots with deliberately conflicting routes.

    Each robot is wired to its own :class:`RobotCoordination` and to
    the shared in-process message bus.
    """
    bus = InProcessMessageBus()
    robots: List[Robot] = []
    # Robot A starts in the upper-left aisle, heading to bottom-right.
    # Robot B starts in the upper-right aisle, heading to bottom-left.
    # Robot C starts in the lower-right, heading to upper-left.
    for rid, pos in [("A", (2, 2)), ("B", (10, 2)), ("C", (10, 5))]:
        view = WorldView(warehouse=warehouse, robots=[], tick=0)
        perception = SimulatedPerception(world_view=view, sensor_range=4, self_id=rid)
        peer_states = PeerStateRepository(self_id=rid, stale_threshold_ticks=5)
        reservations = ReservationTable(owner_id=rid, horizon=8)
        detector = ConflictDetector(owner_id=rid, reservations=reservations, lookahead_horizon=8)
        coordination = RobotCoordination(
            robot_id=rid,
            bus=bus,
            peer_states=peer_states,
            reservations=reservations,
            conflict_detector=detector,
            broadcast_period=1,
        )
        robots.append(Robot(
            robot_id=rid,
            start_position=pos,
            perception=perception,
            battery_capacity=400.0,
            coordination=coordination,
        ))
    return robots, bus


def build_tasks() -> List[Task]:
    """Pickup-and-delivery tasks. Routes intentionally cross."""
    return [
        Task(task_id="T1", pickup_location=(2, 1), dropoff_location=(11, 5), priority=2),
        Task(task_id="T2", pickup_location=(11, 1), dropoff_location=(2, 5), priority=1),
        Task(task_id="T3", pickup_location=(10, 5), dropoff_location=(11, 1), priority=3),
    ]


def build_dynamic_schedule() -> DynamicObstacleSchedule:
    """A deterministic schedule that drops obstacles on planned paths.

    Robot A's post-pickup path to (11, 5) passes through (5, 3). We
    drop a dynamic obstacle there at tick 4 so that A sees the
    blocker before stepping into it and transitions to WAITING. A few
    ticks later the obstacle is removed and A resumes.

    Robot B's path back to the left passes near (8, 1); we drop a
    second obstacle there to exercise the perception/wait logic on a
    second agent.
    """
    return DynamicObstacleSchedule(
        entries=[
            (4, "add", (5, 3)),     # blocks A on the way to (11, 5)
            (4, "add", (8, 1)),     # blocks B on the way back to (2, 5)
            (10, "remove", (5, 3)),  # clears A's path
            (10, "remove", (8, 1)),  # clears B's path
        ]
    )


def render_ascii(warehouse: Warehouse, robots) -> str:
    """Render the warehouse as ASCII for the headless demo."""
    pos_map = {r.position: r.robot_id for r in robots}
    obstacle_set = warehouse.dynamic_obstacle_positions()
    lines: List[str] = []
    for y in range(warehouse.height):
        row: List[str] = []
        for x in range(warehouse.width):
            if (x, y) in obstacle_set:
                row.append("X")
                continue
            cell = warehouse.get_cell_type(x, y)
            if cell is CellType.OBSTACLE:
                row.append("#")
            elif cell is CellType.PICKUP:
                row.append("P")
            elif cell is CellType.DROPOFF:
                row.append("D")
            elif cell is CellType.CHARGING:
                row.append("C")
            else:
                row.append(".")
        for x, ch in enumerate(row):
            if (x, y) in pos_map:
                row[x] = pos_map[(x, y)]
        lines.append("".join(row))
    return "\n".join(lines)


def summarize_event(event) -> str:
    """Compact one-line representation of an event for the demo log."""
    data = {k: v for k, v in event.data.items() if k not in {"tick"}}
    return f"  [{event.tick:03d}] {event.event_type.value:24s} {data}"


def run_demo(max_ticks: int = 60, render_every: int = 5, seed: int = 7, headless: bool = False) -> None:
    warehouse = build_warehouse()
    robots, bus = build_bus_and_robots(warehouse)
    tasks = build_tasks()
    schedule = build_dynamic_schedule()

    sim = Simulator(warehouse, robots, tasks, dynamic_schedule=schedule, seed=seed, message_bus=bus)

    def planner(_robot, start: CellCoord, goal: CellCoord) -> Optional[List[CellCoord]]:
        return astar(start, goal, warehouse.width, warehouse.height, warehouse.is_traversable)

    sim.set_path_planner(planner)

    print("=" * 60)
    print("SIH26123 demo — Phase 2B: decentralized conflict resolution")
    print("=" * 60)
    print(f"Warehouse: {warehouse.width} x {warehouse.height}, seed={seed}")
    print()
    print("Initial layout:")
    print(render_ascii(warehouse, robots))
    print()

    sim.assign_task("T1", "A")
    sim.assign_task("T2", "B")
    sim.assign_task("T3", "C")

    # Record the SIMULATION_STARTED event for the log.
    sim.event_log.record(sim.tick_count, EventType.SIMULATION_STARTED, num_robots=len(robots), num_tasks=len(tasks))

    all_events: List = []

    # Run tick-by-tick so we can capture snapshots at chosen intervals.
    stop = False
    for tick_index in range(max_ticks):
        tick_events = sim.tick()
        all_events.extend(tick_events)
        if not headless and tick_index % render_every == 0:
            print(f"--- snapshot at tick {tick_index} ---")
            print(render_ascii(warehouse, robots))
            print()
        if sim._should_stop():
            stop = True
            break

    sim.event_log.record(sim.tick_count, EventType.SIMULATION_ENDED, total_ticks=sim.tick_count, stopped_early=stop)

    # Print a compact per-tick event log.
    print("-" * 60)
    print("Event log:")
    print("-" * 60)
    for e in all_events:
        if e.event_type in (
            EventType.ROBOT_MOVED,
            EventType.ROBOT_WAITING,
            EventType.ROBOT_ARRIVED,
            EventType.ROBOT_FAILED,
            EventType.TASK_ASSIGNED,
            EventType.TASK_PICKED_UP,
            EventType.TASK_COMPLETED,
            EventType.TASK_FAILED,
            EventType.OBSTACLE_ADDED,
            EventType.OBSTACLE_REMOVED,
            EventType.COLLISION_DETECTED,
            EventType.CONFLICT_PREDICTED,
            EventType.CONFLICT_NEGOTIATION_STARTED,
            EventType.CONFLICT_NEGOTIATION_RESOLVED,
            EventType.RESERVATION_CONFLICT,
            EventType.ROBOT_YIELDED,
            EventType.ROBOT_REROUTED,
            EventType.PEER_STATE_UPDATED,
            EventType.MESSAGE_BROADCAST,
        ):
            print(summarize_event(e))

    # Final summary.
    print()
    print("-" * 60)
    print("Final task statuses:")
    print("-" * 60)
    for t in tasks:
        print(f"  {t.task_id}: {t.status.value:12s} phase={t.phase.value}  robot={t.assigned_robot}")

    print()
    print("Final robot statuses:")
    for r in robots:
        print(f"  {r.robot_id}: pos={r.position} status={r.status.value} battery={r.battery:.1f}")

    collisions = sim.event_log.of_type(EventType.COLLISION_DETECTED)
    if collisions:
        print()
        print(f"Collisions detected: {len(collisions)} (foundation only logs; no resolution)")
        for c in collisions:
            print(f"  {summarize_event(c)}")

    predicted = sim.event_log.of_type(EventType.CONFLICT_PREDICTED)
    if predicted:
        print()
        print(f"Predicted conflicts: {len(predicted)} (Phase 2A detection)")
        # Show at most 5 to keep the log readable.
        for c in predicted[:5]:
            print(f"  {summarize_event(c)}")
        if len(predicted) > 5:
            print(f"  ... and {len(predicted) - 5} more")

    started = sim.event_log.of_type(EventType.CONFLICT_NEGOTIATION_STARTED)
    resolved = sim.event_log.of_type(EventType.CONFLICT_NEGOTIATION_RESOLVED)
    yields = sim.event_log.of_type(EventType.ROBOT_YIELDED)
    reroutes = sim.event_log.of_type(EventType.ROBOT_REROUTED)
    if started or resolved:
        print()
        print("Phase 2B negotiation summary:")
        print(f"  Negotiations started: {len(started)}")
        print(f"  Negotiations resolved: {len(resolved)}")
        print(f"  Robot yielded (WAIT): {len(yields)}")
        print(f"  Robot rerouted:       {len(reroutes)}")

    print()
    print(f"Total ticks: {sim.tick_count}")
    print(f"Total events: {len(sim.event_log.all())}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="SIH26123 Phase 2A demo")
    parser.add_argument("--max-ticks", type=int, default=80, help="Maximum simulation ticks.")
    parser.add_argument("--render-every", type=int, default=10, help="ASCII render frequency.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--headless", action="store_true", help="Suppress ASCII rendering.")
    args = parser.parse_args(argv)

    run_demo(max_ticks=args.max_ticks, render_every=args.render_every, seed=args.seed, headless=args.headless)


if __name__ == "__main__":
    main()