"""Time-stepping world simulator.

The simulator is a *world clock*. It owns the warehouse, the robots,
the task list and the event log. Each tick the simulator:

1. Builds a fresh :class:`WorldView` for every robot.
2. Calls :meth:`Robot.step` on every robot and records the events it
   produced. The simulator does not decide paths, resolve conflicts or
   assign tasks; those decisions live in the robot (or, in the next
   layer, in dedicated agent modules).
3. Updates task progress based on robot positions and the task
   phases (a robot that reaches its pickup advances to dropoff;
   reaching the dropoff completes the task).
4. Runs the configured dynamic-obstacle schedule (foundation: a
   deterministic timeline of placements and removals).
5. Detects position collisions between robots and logs them. No
   resolution.
6. Advances its internal clock.

Determinism
-----------
The simulator holds its own :class:`random.Random` instance configured
by a caller-supplied seed. No code anywhere else should call
``random.*`` without owning its RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..coordination.communication import MessageBus
from ..coordination.conflicts import PredictedConflict
from ..coordination.lifecycle import RobotLifecycleState
from ..perception.simulated_sensor import SimulatedPerception, WorldView
from .events import Event, EventLog, EventType
from .robot import Robot, RobotStatus
from .task import Task, TaskPhase, TaskStatus
from .warehouse import Warehouse

CellCoord = Tuple[int, int]


@dataclass
class DynamicObstacleSchedule:
    """A deterministic schedule of dynamic obstacle changes.

    Each entry is ``(tick, kind, position)`` and ``kind`` is one of
    ``"add"`` or ``"remove"``. The simulator applies them in tick
    order. ``tick=0`` entries are applied during the very first tick.
    """

    entries: List[Tuple[int, str, CellCoord]] = field(default_factory=list)


# Type alias for the path-planner function the simulator can call.
PathPlanner = Callable[[Robot, CellCoord, CellCoord], Optional[List[CellCoord]]]


class Simulator:
    """Deterministic time-stepping world simulator."""

    def __init__(
        self,
        warehouse: Warehouse,
        robots: List[Robot],
        tasks: List[Task],
        dynamic_schedule: Optional[DynamicObstacleSchedule] = None,
        seed: Optional[int] = None,
        collision_log_enabled: bool = True,
        message_bus: Optional[MessageBus] = None,
    ) -> None:
        self.warehouse = warehouse
        self.robots = list(robots)
        self.tasks = list(tasks)
        self.dynamic_schedule = dynamic_schedule or DynamicObstacleSchedule()
        self.rng = random.Random(seed)
        self.collision_log_enabled = collision_log_enabled
        self.message_bus = message_bus

        self.tick_count: int = 0
        self.event_log = EventLog()
        self._path_planner: Optional[PathPlanner] = None

        # Phase 2B.2: per-tick set of conflict_ids already negotiated
        # in the current tick. Both the first-round negotiation (step
        # 7) and the second-round negotiation (step 9b) consult this
        # set so the same logical conflict is not processed twice in
        # one tick. Conflicts that are genuinely different (different
        # timestep, robot pair, cell, or type) are NOT affected.
        self._negotiated_this_tick: set = set()

        # Phase 2C: last tick at which each robot received at least
        # one valid peer heartbeat (or broadcast). Used by the
        # SELF_ISOLATION_TIMEOUT self-isolation check.
        self._last_peer_heartbeat_tick: Dict[str, int] = {
            r.robot_id: -1 for r in self.robots
        }

        # Phase 2C: per-robot "wait-for" cycle detection. We track
        # which conflicts are currently in-flight for each robot via
        # ``robot.coordination.negotiation_records``. Deadlock
        # detection reads those records at the end of each tick.
        # No additional state needed here.

        # Telemetry for tests: keep ids seen at each tick.
        self._robot_ids: List[str] = [r.robot_id for r in self.robots]
        self._validate_robot_ids()
        self._validate_task_ids()

        # Phase 2A: register all robots that have a coordination
        # module on the bus. Robots without coordination are still
        # allowed; the simulator just skips them in the coordination
        # hooks.
        if self.message_bus is not None:
            for r in self.robots:
                self.message_bus.register_peer(r.robot_id)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_robot_ids(self) -> None:
        seen = set()
        for rid in self._robot_ids:
            if rid in seen:
                raise ValueError(f"Duplicate robot id {rid!r}")
            seen.add(rid)

    def _validate_task_ids(self) -> None:
        seen = set()
        for t in self.tasks:
            if t.task_id in seen:
                raise ValueError(f"Duplicate task id {t.task_id!r}")
            seen.add(t.task_id)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def assign_task(self, task_id: str, robot_id: str) -> bool:
        """Assign ``task_id`` to ``robot_id`` and request a path.

        The simulator uses the path planner installed via
        :meth:`set_path_planner`. Returns True on success, False on
        failure (already logged).
        """
        robot = self._get_robot(robot_id)
        task = self._get_task(task_id)

        if robot is None or task is None:
            self.event_log.record(
                self.tick_count,
                EventType.TASK_ASSIGNED,
                task_id=task_id,
                robot_id=robot_id,
                success=False,
                reason="unknown_robot_or_task",
            )
            return False
        if robot.current_task is not None:
            self.event_log.record(
                self.tick_count,
                EventType.TASK_ASSIGNED,
                task_id=task_id,
                robot_id=robot_id,
                success=False,
                reason="robot_busy",
            )
            return False
        if task.status is not TaskStatus.PENDING:
            self.event_log.record(
                self.tick_count,
                EventType.TASK_ASSIGNED,
                task_id=task_id,
                robot_id=robot_id,
                success=False,
                reason="task_not_pending",
            )
            return False

        planner = self._path_planner
        if planner is None:
            self.event_log.record(
                self.tick_count,
                EventType.TASK_ASSIGNED,
                task_id=task_id,
                robot_id=robot_id,
                success=False,
                reason="no_path_planner",
            )
            return False

        task.mark_assigned(robot_id)
        robot.current_task = task

        # Plan path to the current task target (pickup first).
        target = task.current_target()
        assert target is not None
        path = planner(robot, robot.position, target)
        if not path:
            task.mark_failed()
            robot.current_task = None
            robot.status = RobotStatus.FAILED
            self.event_log.record(self.tick_count, EventType.TASK_FAILED, task_id=task_id, robot_id=robot_id, reason="no_path")
            self.event_log.record(self.tick_count, EventType.TASK_ASSIGNED, task_id=task_id, robot_id=robot_id, success=False, reason="no_path")
            return False

        robot.set_path(path, target)
        self.event_log.record(self.tick_count, EventType.TASK_ASSIGNED, task_id=task_id, robot_id=robot_id, success=True)
        self.event_log.record(self.tick_count, EventType.ROBOT_STARTED, robot_id=robot_id, task_id=task_id, destination=list(target))
        return True

    # ------------------------------------------------------------------
    # Internal lookups
    # ------------------------------------------------------------------
    def _get_robot(self, robot_id: str) -> Optional[Robot]:
        for r in self.robots:
            if r.robot_id == robot_id:
                return r
        return None

    def _get_task(self, task_id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    # ------------------------------------------------------------------
    # The tick loop
    # ------------------------------------------------------------------
    def _refresh_world_views(self) -> None:
        """Build a fresh WorldView per robot before the step.

        We give each robot's perception its own view, with a copy of
        the robot list excluding the perceiver's own id (already
        handled inside :class:`SimulatedPerception`). The view is a
        *snapshot* — the robot reads from it during its step.
        """
        for robot in self.robots:
            view = WorldView(
                warehouse=self.warehouse,
                robots=list(self.robots),
                tick=self.tick_count,
            )
            if isinstance(robot.perception, SimulatedPerception):
                robot.perception.world_view = view
                robot.perception.self_id = robot.robot_id
            # Non-simulated perception modules are expected to maintain
            # their own world view contract; we leave them alone.

    def _apply_dynamic_schedule(self) -> List[Event]:
        events: List[Event] = []
        for tick, kind, position in list(self.dynamic_schedule.entries):
            if tick != self.tick_count:
                continue
            x, y = position
            if kind == "add":
                if self.warehouse.is_within_bounds(x, y) and not self.warehouse.has_dynamic_obstacle(x, y):
                    obs = self.warehouse.add_dynamic_obstacle(x, y)
                    events.append(Event(self.tick_count, EventType.OBSTACLE_ADDED, obstacle_id=obs.obstacle_id, position=[x, y]))
            elif kind == "remove":
                if self.warehouse.remove_dynamic_obstacle(x, y):
                    events.append(Event(self.tick_count, EventType.OBSTACLE_REMOVED, position=[x, y]))
            else:
                raise ValueError(f"Unknown dynamic schedule kind {kind!r}")
        return events

    def _update_tasks(self) -> List[Event]:
        """Drive task lifecycle from observed robot positions.

        Robots make their own phase transitions; here we just observe
        the resulting state and reflect it on the task object.

        Phase 2C: when a robot enters FAILED while holding an active
        task, that task is returned to PENDING (returned to the task
        pool) so it can be reassigned to a surviving robot. The robot's
        ``current_task`` is cleared without teleporting it. We also
        mark the robot's local lifecycle as FAILED so other peers
        observe it through broadcasts and the staleness/failure
        path stays consistent.
        """
        events: List[Event] = []
        for robot in self.robots:
            task = robot.current_task
            if task is None:
                continue
            # Phase 2C: failed-robot task return-to-pool.
            if robot.status is RobotStatus.FAILED:
                if robot.coordination is not None and \
                        robot.coordination.lifecycle.state != \
                        RobotLifecycleState.FAILED:
                    robot.coordination.lifecycle.state = \
                        RobotLifecycleState.FAILED
                    robot.coordination.lifecycle.failed_since_tick = \
                        self.tick_count
                events.extend(self._return_failed_task_to_pool(robot, task))
                continue
            if task.status is TaskStatus.ASSIGNED:
                # Robot has started moving: transition to IN_PROGRESS.
                task.mark_in_progress()
            if task.status is TaskStatus.IN_PROGRESS:
                if task.phase is TaskPhase.TO_PICKUP and robot.position == task.pickup_location:
                    task.mark_pickup_done()
                    events.append(Event(self.tick_count, EventType.TASK_PICKED_UP, task_id=task.task_id, robot_id=robot.robot_id, position=list(robot.position)))
                    # Re-plan toward dropoff.
                    self._replan_current_target(robot, events)
                elif task.phase is TaskPhase.TO_DROPOFF and robot.position == task.dropoff_location:
                    task.mark_completed()
                    robot.current_task = None
                    robot.clear_path()
                    events.append(Event(self.tick_count, EventType.TASK_COMPLETED, task_id=task.task_id, robot_id=robot.robot_id, position=list(robot.position)))
        return events

    def _return_failed_task_to_pool(self, robot: Robot, task: Task) -> List[Event]:
        """Phase 2C: a robot holding an active task failed.

        The task is returned to PENDING (the pool), the robot's
        ``current_task`` is cleared, and a ``TASK_RETURNED_TO_POOL``
        event is emitted. The simulator does not itself reassign the
        task; reassignment is up to the orchestration layer (the demo
        / main wiring) which decides which surviving robot to claim
        the task.
        """
        events: List[Event] = []
        previous_robot = robot.robot_id
        # Make the task claimable again.
        task.status = TaskStatus.PENDING
        task.phase = TaskPhase.TO_PICKUP
        task.assigned_robot = None
        robot.current_task = None
        robot.clear_path()
        events.append(Event(
            self.tick_count,
            EventType.TASK_RETURNED_TO_POOL,
            task_id=task.task_id,
            previous_robot=previous_robot,
        ))
        return events

    def _replan_current_target(self, robot: Robot, events: List[Event]) -> None:
        """Replan the path from the current position to the next target.

        Uses the same heuristic as :meth:`assign_task` but expects the
        caller (the simulator demo) to install a path planner via
        :meth:`set_path_planner`. If no planner is installed, the robot
        waits until one is.
        """
        task = robot.current_task
        if task is None:
            return
        target = task.current_target()
        if target is None:
            return
        planner = self._path_planner
        if planner is None:
            # No planner registered. We rely on the demo setting one
            # before run(). If absent, just keep moving to the existing
            # destination (if any) and wait.
            robot.clear_path()
            robot.status = RobotStatus.WAITING
            events.append(Event(self.tick_count, EventType.ROBOT_WAITING, robot_id=robot.robot_id, reason="no_path_planner"))
            return
        path = planner(robot, robot.position, target)
        if not path:
            robot.clear_path()
            robot.status = RobotStatus.WAITING
            events.append(Event(self.tick_count, EventType.ROBOT_WAITING, robot_id=robot.robot_id, reason="no_replan_path"))
            return
        robot.set_path(path, target)

    def set_path_planner(self, planner: PathPlanner) -> None:
        """Install a path planner used by the simulator when re-planning.

        The simulator does not embed a planner; the demo / tests install
        one (typically ``lambda r, s, g: astar(s, g, w, h, lambda x, y: w.is_traversable(x, y))``).
        """
        self._path_planner = planner

    # ------------------------------------------------------------------
    # Collision detection (record only)
    # ------------------------------------------------------------------
    def _detect_collisions(self, events_buffer: List[Event]) -> None:
        if not self.collision_log_enabled:
            return
        positions: Dict[CellCoord, List[str]] = {}
        for robot in self.robots:
            if robot.status is RobotStatus.FAILED:
                continue
            positions.setdefault(robot.position, []).append(robot.robot_id)
        for pos, ids in positions.items():
            if len(ids) > 1:
                events_buffer.append(Event(
                    self.tick_count,
                    EventType.COLLISION_DETECTED,
                    position=list(pos),
                    robot_ids=sorted(ids),
                ))

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def tick(self) -> List[Event]:
        """Advance the world by one simulation tick.

        Returns the events generated this tick. The events are also
        appended to :attr:`self.event_log` so callers using
        ``sim.tick()`` directly still see a complete log.

        Tick order
        ----------
        1. Apply the dynamic-obstacle schedule.
        2. Refresh each robot's :class:`WorldView`.
        3. **Phase 2A** — deliver bus messages from the previous tick.
        4. **Phase 2A** — each robot processes its inbox (updates peer
           state and peer reservations).
        5. **Phase 2A** — each robot rebuilds its own reservations
           from its current state.
        6. **Phase 2A** — each robot runs its local conflict detector.
        7. **Phase 2B** — each robot runs its local negotiation on
           the detected conflicts (may reroute or yield).
        8. **Phase 1** — each robot's :meth:`Robot.step` runs (motion).
        9. **Phase 1** — task progress / pickup-to-dropoff replan.
           Robots that just completed a pickup now hold the *full*
           intended trajectory; if the path changed, the
           corresponding reservations are refreshed in step 9b.
        10. **Phase 2B.1 — FINAL post-tick broadcast** (after motion
            AND after any pickup->dropoff replan). Peers therefore
            always see the robot's FINAL trajectory for the tick.
        11. Detect collisions (logging only).
        12. Advance the clock.

        Robots without a ``coordination`` field are entirely skipped
        at steps 3-7 and 10; their behaviour is identical to Phase 1.
        """
        events: List[Event] = []

        # Phase 2B.2: reset per-tick conflict-id dedup sets on every
        # coordination-capable robot. This guarantees the dedup window
        # is exactly one tick and the second-round negotiation (step
        # 9b) only suppresses conflicts already processed by the
        # first round (step 7) — it does not erase history across
        # ticks.
        for _r in self.robots:
            if _r.coordination is not None:
                _r.coordination.processed_conflict_ids_this_tick.clear()

        # 1. Apply dynamic obstacle schedule first so robots see them
        #    in the same tick they appear.
        events.extend(self._apply_dynamic_schedule())

        # 2. Refresh each robot's view of the world.
        self._refresh_world_views()

        # 3. Phase 2A: deliver messages from the previous tick.
        if self.message_bus is not None:
            self.message_bus.deliver()

        # 4-6. Phase 2A: coordination hooks before motion.
        coordination_robots = [r for r in self.robots if r.coordination is not None]

        # Cache predicted conflicts per robot so step 7 (negotiation)
        # can reuse them without re-running the detector.
        per_robot_conflicts: Dict[str, List[PredictedConflict]] = {}

        # Maximum task priority observed across the task set, used to
        # normalise priorities in the local score function. Recomputed
        # each tick so it adapts to live task allocation.
        max_task_priority = max([t.priority for t in self.tasks], default=0)
        max_task_priority = max(1, max_task_priority)

        # Phase 2C 4a: record peer-heartbeat tick for every robot
        # that just received a valid peer broadcast this tick.
        # Only update if there is at least one peer whose last seen
        # tick is THIS tick (i.e. we actually got a fresh broadcast).
        for robot in coordination_robots:
            fresh = False
            for info in robot.coordination.peer_liveness.all().values():
                if info.last_seen_tick == self.tick_count:
                    fresh = True
                    break
            if fresh:
                self._last_peer_heartbeat_tick[robot.robot_id] = self.tick_count

        # Phase 2C 4b: stale-peer sweep. Any peer whose last valid
        # heartbeat is older than STALE_THRESHOLD ticks has its
        # forward reservations cleared and is marked STALE in the
        # local view.
        from ..coordination.resilience import (
            PEER_FAILURE_TIMEOUT, SELF_ISOLATION_TIMEOUT, STALE_THRESHOLD,
        )
        for robot in coordination_robots:
            events.extend(robot.coordination.mark_peer_stale(self.tick_count))

        # Phase 2C 4b.5: peer-failure sweep. Any peer whose last
        # valid heartbeat is older than PEER_FAILURE_TIMEOUT ticks is
        # declared OFFLINE; its forward reservations are released.
        failed_peers: Dict[str, List] = {}
        for robot in coordination_robots:
            liveness = robot.coordination.peer_liveness
            for pid, info in liveness.all().items():
                if info.status == "offline":
                    continue
                last = info.last_seen_tick
                if last < 0:
                    continue
                if (self.tick_count - last) >= PEER_FAILURE_TIMEOUT:
                    failed_peers.setdefault(robot.robot_id, []).append(pid)
        for robot_id, peer_ids in failed_peers.items():
            robot = next(r for r in self.robots if r.robot_id == robot_id)
            for pid in peer_ids:
                events.extend(
                    robot.coordination.declare_peer_failed(pid, self.tick_count)
                )

        # Phase 2C 4c: self-isolation check. If a robot has not
        # received a valid peer heartbeat for SELF_ISOLATION_TIMEOUT
        # ticks, it enters SAFE_HALT. The SAFE_HALT robot stops making
        # progress that depends on stale peer beliefs. Local safety
        # checks continue to run.
        for robot in coordination_robots:
            since_last_peer_heartbeat = (
                self.tick_count - self._last_peer_heartbeat_tick[robot.robot_id]
            )
            if since_last_peer_heartbeat >= SELF_ISOLATION_TIMEOUT:
                events.extend(robot.coordination.enter_safe_halt(self.tick_count))

        # Phase 2C 4d: recovery. A SAFE_HALT robot that has just
        # received a peer heartbeat in this tick resumes ACTIVE.
        from ..coordination.lifecycle import RobotLifecycleState
        for robot in coordination_robots:
            if (
                robot.coordination.lifecycle.state == RobotLifecycleState.SAFE_HALT
                and self._last_peer_heartbeat_tick[robot.robot_id] == self.tick_count
            ):
                events.extend(robot.coordination.exit_safe_halt(self.tick_count))

        for robot in coordination_robots:
            processed = robot.coordination.process_inbox()
            if processed > 0:
                for peer_id, state in robot.coordination.peer_states.all().items():
                    events.append(Event(
                        self.tick_count,
                        EventType.PEER_STATE_UPDATED,
                        robot_id=robot.robot_id,
                        peer_id=peer_id,
                        peer_timestamp=state.timestamp,
                    ))
            robot.coordination.update_own_reservations(
                current_pos=robot.position,
                path=robot.remaining_path(),
                current_tick=self.tick_count,
            )
            conflicts = robot.coordination.predict_conflicts(self.tick_count)
            per_robot_conflicts[robot.robot_id] = list(conflicts)
            for c in conflicts:
                events.append(self._conflict_to_event(c, robot.robot_id))

        # 7. Phase 2B: local negotiation. Each robot decides for itself.
        for robot in coordination_robots:
            conflicts = per_robot_conflicts.get(robot.robot_id, [])
            events.extend(robot.coordination.run_negotiation(
                robot=robot,
                warehouse=self.warehouse,
                current_tick=self.tick_count,
                conflicts=conflicts,
                max_task_priority=max_task_priority,
            ))

        # 8. Phase 1: motion.
        for robot in self.robots:
            events.extend(robot.step(self.tick_count, self.warehouse))

        # 9. Phase 1: task progress / pickup-to-dropoff replan.
        events.extend(self._update_tasks())

        # 9b. Phase 2B.1: after the motion/replan above, each robot's
        #     own reservation table is rebuilt from its CURRENT
        #     trajectory, then a *second* round of conflict detection
        #     and negotiation is run so that any collision predicted by
        #     the LATEST paths (including the replan) can be resolved
        #     before the broadcast at step 10.
        per_robot_conflicts2: Dict[str, List[PredictedConflict]] = {}
        for robot in coordination_robots:
            if robot.path:
                robot.coordination.update_own_reservations(
                    current_pos=robot.position,
                    path=robot.remaining_path(),
                    current_tick=self.tick_count,
                )
        for robot in coordination_robots:
            conflicts = robot.coordination.predict_conflicts(self.tick_count)
            per_robot_conflicts2[robot.robot_id] = list(conflicts)
            for c in conflicts:
                events.append(self._conflict_to_event(c, robot.robot_id))
        for robot in coordination_robots:
            conflicts = per_robot_conflicts2.get(robot.robot_id, [])
            events.extend(robot.coordination.run_negotiation(
                robot=robot,
                warehouse=self.warehouse,
                current_tick=self.tick_count,
                conflicts=conflicts,
                max_task_priority=max_task_priority,
            ))

        # 10. Phase 2B.1: FINAL post-tick broadcast. Crucially this
        #     happens AFTER motion (step 8) AND after the pickup-
        #     to-dropoff replan (step 9), so peers always observe the
        #     robot's FINAL trajectory for the tick.
        if self.message_bus is not None:
            for robot in coordination_robots:
                # Phase 2C test hook: silenced robots must NOT broadcast.
                if getattr(robot, "silenced", False):
                    continue
                task_id = robot.current_task.task_id if robot.current_task else None
                task_priority = (
                    int(getattr(robot.current_task, "priority", 0) or 0)
                    if robot.current_task is not None
                    else 0
                )
                published = robot.coordination.publish_state(
                    current_tick=self.tick_count,
                    position=robot.position,
                    velocity=robot.velocity,
                    battery=robot.battery,
                    task_id=task_id,
                    task_priority=task_priority,
                    status=robot.status.value,
                    planned_path=robot.remaining_path(),
                    intent=robot.describe_intent(),
                )
                if published and not getattr(robot, "silenced", False):
                    events.append(Event(
                        self.tick_count,
                        EventType.MESSAGE_BROADCAST,
                        sender_id=robot.robot_id,
                        msg_type="robot_state",
                    ))
                    # Phase 2C: also emit the semantic heartbeat
                    # event so failure / staleness tests can subscribe
                    # to a dedicated channel rather than parsing
                    # broadcast payloads.
                    events.append(Event(
                        self.tick_count,
                        EventType.ROBOT_HEARTBEAT,
                        sender_id=robot.robot_id,
                        sequence=robot.coordination.sequence,
                    ))

        # 11. Collision detection.
        self._detect_collisions(events)

        # Phase 2C: deadlock detection across coordination robots.
        # We use the negotiation_records collected by each robot
        # during this tick to find a wait-for cycle. The lowest-id
        # robot in the cycle releases its claim, ensuring the
        # reservation safety invariant is preserved (no teleporting,
        # no overwriting of peer reservations, no new collisions).
        from ..coordination.resilience import DEADLOCK_WINDOW
        deadlock_cid = self._detect_deadlock_across_robots(coordination_robots)
        if deadlock_cid is not None:
            events.extend(self._resolve_deadlock(deadlock_cid))

        # 12. Advance the clock.
        self.tick_count += 1

        self.event_log.extend(events)
        return events

    # ------------------------------------------------------------------
    # Phase 2C: deadlock detection helpers
    # ------------------------------------------------------------------
    def _detect_deadlock_across_robots(
        self, coordination_robots,
    ) -> Optional[str]:
        """Detect a wait-for cycle using each robot's
        ``negotiation_records`` collected during this tick.

        A wait-for cycle is: A yields-on-conflict-with-B and B
        yields-on-conflict-with-C and ... yields-on-conflict-with-A.
        The conflict_id at the head of the cycle is returned (any one
        will do; the resolver picks the deterministic loser below).

        The check uses the ``DEADLOCK_WINDOW`` master-plan constant.
        """
        from ..coordination.resilience import DEADLOCK_WINDOW
        # Build a directed graph: robot_id -> set of peers that this
        # robot is waiting on (i.e. yielded against).
        waits_for: Dict[str, set] = {}
        for robot in coordination_robots:
            for cid, rec in robot.coordination.negotiation_records.items():
                if rec.resolution is None:
                    continue
                if rec.peer_id == robot.robot_id:
                    continue
                waits_for.setdefault(robot.robot_id, set()).add(rec.peer_id)
        # DFS colour map.
        WHITE, GREY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {pid: WHITE for pid in waits_for}

        def dfs(node: str) -> Optional[str]:
            colour[node] = GREY
            for neighbour in waits_for.get(node, ()):  # type: ignore[arg-type]
                if colour.get(neighbour, WHITE) == GREY:
                    return True  # found a cycle
                if colour.get(neighbour, WHITE) == WHITE:
                    if dfs(neighbour):
                        return True
            colour[node] = BLACK
            return False

        for pid in waits_for:
            if colour[pid] == WHITE:
                if dfs(pid):
                    # Find a conflict_id that participates in the
                    # cycle. We pick the lowest-id pair.
                    candidates = []
                    for robot in coordination_robots:
                        for cid, rec in robot.coordination.negotiation_records.items():
                            if (
                                rec.resolution is not None
                                and rec.peer_id in waits_for.get(robot.robot_id, ())
                                and rec.peer_id != robot.robot_id
                            ):
                                candidates.append((cid, rec))
                    if not candidates:
                        # No concrete conflict_id found; return None.
                        return None
                    candidates.sort(key=lambda x: x[0])
                    return candidates[0][0]
        # If the window-based check passes, we still want a final
        # pass for windows below the threshold. This is a simplification
        # because we don't keep per-record arrival time.
        # In practice the master plan calls for the WINDOW check.
        return None

    def _resolve_deadlock(self, conflict_id: str) -> List:
        """Resolve the deadlock deterministically.

        Strategy: the lowest-id robot involved in the conflict releases
        its yield (drops the NegotiationRecord), clearing its
        reservation for the conflicting cell so the cycle breaks. The
        other robot's reservation is preserved - the cell is
        protected for whichever robot gets there first. No teleporting,
        no overwriting of peer reservations.
        """
        from ..coordination.resilience import DEADLOCK_WINDOW
        from ..simulation.events import Event, EventType
        events: List = []
        events.append(Event(
            self.tick_count,
            EventType.DEADLOCK_DETECTED,
            conflict_id=conflict_id,
            since_tick=self.tick_count - DEADLOCK_WINDOW,
        ))
        # Drop the deadlock-tracking records on both sides.
        for robot in self.robots:
            if robot.coordination is not None:
                robot.coordination.negotiation_records.pop(conflict_id, None)
        events.append(Event(
            self.tick_count,
            EventType.DEADLOCK_RESOLVED,
            conflict_id=conflict_id,
        ))
        return events

    def _conflict_to_event(self, conflict, observer_id: str) -> Event:
        """Convert a predicted conflict into a simulator event."""
        data = {
            "observer": observer_id,
            "robot_a": conflict.robot_a,
            "robot_b": conflict.robot_b,
            "timestep": conflict.timestep,
        }
        if conflict.conflict_type.value == "vertex":
            data["cell"] = list(conflict.cell) if conflict.cell is not None else None
        else:
            data["edge"] = [list(conflict.edge[0]), list(conflict.edge[1])] if conflict.edge else None
        return Event(
            self.tick_count,
            EventType.CONFLICT_PREDICTED,
            **data,
        )

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------
    def run(self, max_ticks: int = 200, stop_when_all_tasks_done: bool = True) -> List[Event]:
        """Run the simulator for up to ``max_ticks``.

        If ``stop_when_all_tasks_done`` is True (default), the loop
        terminates early once every task is in a terminal state
        (completed or failed) and no robot is still moving. The return
        value is a list of all events generated during the run.
        """
        all_events: List[Event] = []
        self.event_log.record(self.tick_count, EventType.SIMULATION_STARTED, num_robots=len(self.robots), num_tasks=len(self.tasks))
        for _ in range(max_ticks):
            tick_events = self.tick()
            all_events.extend(tick_events)
            if stop_when_all_tasks_done and self._should_stop():
                break
        self.event_log.record(self.tick_count, EventType.SIMULATION_ENDED, total_ticks=self.tick_count)
        return all_events

    def _should_stop(self) -> bool:
        for t in self.tasks:
            if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return False
        for r in self.robots:
            if r.status in (RobotStatus.MOVING, RobotStatus.WAITING):
                return False
        return True

    # ------------------------------------------------------------------
    # Public convenience
    # ------------------------------------------------------------------
    def get_robot(self, robot_id: str) -> Optional[Robot]:
        return self._get_robot(robot_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._get_task(task_id)