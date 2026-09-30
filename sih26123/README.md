# SIH26123 — Edge-AI Based Distributed Fleet Coordination for AMRs

SIH 2026 Problem Statement **SIH26123**: *Edge-AI Based Distributed Fleet
Coordination for Autonomous Mobile Robots (AMRs) in Smart Warehouses*.

This repository currently contains three layers of the prototype:

- **Phase 1 (foundation)** — deterministic simulation core, world
  clock, robot agents, perception, A\* planning, dynamic obstacles,
  collision detection, event log.
- **Phase 2A (decentralized coordination layer)** — P2P
  communication abstraction, per-robot peer-state repository,
  space-time reservations, vertex + edge conflict detection, and
  proactive conflict prediction.
- **Phase 2B / 2B.1 (decentralized conflict resolution)** —
  deterministic priority score, negotiation protocol, space-time
  rerouting, wait-vs-reroute decision, and a fixed broadcast order
  so peers always see each robot's FINAL post-tick state.

No frontend, no FastAPI, no MQTT, no ROS, no ML, no hardware. Those are
intentionally out of scope.

---

## What the simulation currently does

### Phase 1 (foundation)

1. **Grid-based warehouses** with free cells, walls, pickup cells,
   drop-off cells and charging cells.
2. **Dynamic obstacles** that can appear and disappear on a
   deterministic schedule.
3. **Three or more robots** that each own their own state,
   perception and battery. Robots move one cell per simulation tick.
4. **A\* path planning** (`planning/astar.py`) using Manhattan
   distance. The planner is decoupled from the warehouse — it depends
   only on a grid and an `is_traversable` callable.
5. **Per-robot perception** (`perception/simulated_sensor.py`) with
   a configurable sensor range.
6. **Pickup-and-delivery tasks** with phases (`TO_PICKUP`,
   `TO_DROPOFF`, `DONE`) and explicit status transitions.
7. **Collision detection** — the simulator logs when two robots
   occupy the same cell. It does **not** resolve conflicts.
8. **In-memory event log** of every important state change.
9. **Deterministic execution** — the simulator owns its own seeded
   RNG; no uncontrolled randomness anywhere.

### Phase 2A (coordination layer)

10. **P2P communication abstraction**
    (`coordination/communication.py`). A transport-agnostic
    `MessageBus` interface and an `InProcessMessageBus` reference
    implementation. The bus only transports messages — it does not
    interpret them or make decisions.
11. **Structured robot state broadcasts** via typed
    `RobotState` messages (position, velocity, battery, task,
    status, planned path, intent, task_priority).
12. **Per-robot peer-state repository**
    (`coordination/communication.py::PeerStateRepository`). Each
    robot maintains its **own** local view of peer robots. There is
    intentionally no global authoritative state.
13. **Space-time reservations**
    (`coordination/reservations.py`). Per-robot
    `ReservationTable` with `Reservation(cell, timestep, robot_id)`
    semantics. Each robot's table holds its own reservations AND the
    reservations it has received from peers.
14. **Vertex + edge conflict detection**
    (`coordination/conflicts.py`). Each robot's
    `ConflictDetector` answers the local question "is there a
    conflict involving me in the next H ticks?". It does **not**
    decide who yields.
15. **Proactive conflict prediction** with a configurable lookahead
    horizon. Conflicts are emitted as `CONFLICT_PREDICTED` events
    *before* the conflict tick.

### Phase 2B (decentralized conflict resolution)

16. **Deterministic priority score** (`coordination/priority.py`).
    Weighted sum of task urgency, deadline pressure, route progress,
    waiting time and battery pressure, all normalised to `[0, 1]`.
    Ties broken lexicographically by `robot_id`. Same inputs always
    yield the same score.
17. **Negotiation protocol**
    (`coordination/negotiation.py`). Each robot emits a typed
    `CONFLICT_PROPOSAL` message after computing its local decision.
    Decisions are made **locally** from the local priority
    comparison — no waiting for a peer's reply.
18. **Wait vs reroute cost comparison**
    (`coordination/resolver.py`). For each predicted conflict the
    yielding robot compares the cost of waiting vs rerouting using a
    pure cost function and picks the cheaper option.
19. **Space-time rerouting** (`coordination/spacetime.py`). A
    dedicated space-time A\* that reasons over `(x, y, t)` states
    (including a *wait* action). The rerouting filter blocks:
    static and dynamic obstacles, and any *peer* space-time
    reservation (own reservations are preserved).
20. **Local negotiation hooks in `RobotCoordination`** (see
    `coordination/agent.py`). Each robot owns a
    `RobotCoordination` that processes its inbox, predicts its
    conflicts, makes its own yield/reroute decision, executes it,
    and emits structured negotiation events.
21. **Decision events** (`CONFLICT_NEGOTIATION_STARTED`,
    `CONFLICT_NEGOTIATION_RESOLVED`, `RESERVATION_CONFLICT`,
    `ROBOT_YIELDED`, `ROBOT_REROUTED`) in the simulator's event log.
22. **Negotiation is bounded**: a robot that has already entered a
    WAITING period does **not** re-yield to new conflicts (which
    would extend the wait forever). This keeps negotiation
    deterministic and avoids infinite loops.
23. **Reservation-race handling**: if two robots independently
    reserve `(X, T)` before seeing each other, each robot's
    `ConflictDetector` flags it and the local priority comparison
    determines a deterministic winner. The loser invalidates its
    own reservation by rebuilding from the rerouted path.

### Phase 2B.1 — broadcast-after-replan tick order

24. **Broadcast-after-replan tick order**: in the simulator, the
    ROBOT_STATE broadcast is emitted *after* motion AND *after* the
    pickup→dropoff replan. Peers therefore always observe each
    robot's FINAL post-tick trajectory, never the stale pre-replan
    state. (See `simulation/simulator.py::Simulator.tick`.)
25. **A second round of conflict detection + negotiation** runs at
    step 9 of the simulator (after replan and reservation refresh).
    This catches conflicts that the latest paths would cause and
    resolves them with a second reroute before the broadcast at
    step 10.

### Phase 2B.2 - per-tick conflict-id dedup

26. **Per-tick conflict-id dedup**: each RobotCoordination keeps
    a per-tick set of processed conflict_ids. The simulator clears
    the set at the start of every tick. A conflict whose id has
    already been processed in the current tick is skipped by the
    negotiation. The id fully identifies the conflict (robot pair +
    cell + timestep + type), so the dedup window is exactly one
    tick and only *duplicate* processing is suppressed.
27. **Dedup is safety-preserving**: the second-round negotiation at
    step 9b still runs; it just suppresses work that the first round
    has already done. A conflict that genuinely changes between the
    two rounds (e.g. because of a different cell or timestep or
    robot pair) has a different conflict_id and is processed
    normally.

---

## Project structure

```
sih26123/
├── pyproject.toml
├── README.md
├── requirements.txt
├── main.py                    # Phase 2B.1 demo entrypoint
├── simulation/
│   ├── __init__.py
│   ├── warehouse.py           # grid world + dynamic obstacles
│   ├── obstacle.py
│   ├── robot.py               # Robot agent (local state + step())
│   ├── task.py
│   ├── events.py              # EventType + EventLog
│   └── simulator.py           # world clock (Phase 2B.1 broadcast order)
├── planning/
│   ├── __init__.py
│   └── astar.py               # single-robot A* with Manhattan heuristic
├── perception/
│   ├── __init__.py
│   └── simulated_sensor.py
├── coordination/              # Phase 2A + 2B additions
│   ├── __init__.py
│   ├── communication.py       # MessageBus, RobotState, PeerStateRepository
│   ├── reservations.py        # Reservation, ReservationTable
│   ├── conflicts.py           # ConflictType, PredictedConflict, ConflictDetector
│   ├── agent.py                # RobotCoordination (per-robot orchestrator)
│   ├── priority.py            # deterministic priority score
│   ├── negotiation.py         # CONFLICT_PROPOSAL/RESPONSE messages
│   ├── spacetime.py            # space-time A* for rerouting
│   └── resolver.py            # wait-vs-reroute cost comparison
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_astar.py
    ├── test_warehouse.py
    ├── test_robot.py
    ├── test_simulation.py
    ├── test_communication.py
    ├── test_reservations.py
    ├── test_conflicts.py
    ├── test_coordination.py
    └── test_phase2b.py
```

---

## Install

Python 3.11+ is required. From the repository root:

```bash
pip install -r sih26123/requirements.txt
```

That installs `pytest`. The rest of the code uses the standard library
only.

---

## Run the demo

From the repository root:

```bash
python -m sih26123.main --max-ticks 60 --render-every 5 --seed 7
```

The demo runs three robots with intentionally conflicting routes
through a small warehouse. With Phase 2B.1's broadcast-after-replan
tick order and the second-round negotiation, the seed-7 scenario
produces **zero collisions** while preserving the original
decentralised architecture (no central traffic controller, no +1
reservation padding, no teleporting, no collision suppression).

The demo prints:

- The initial warehouse layout (ASCII)
- Per-tick snapshots of the world
- A tick-by-tick event log including:
  - `robot_moved`, `robot_arrived`, `robot_waiting`
  - `task_assigned`, `task_picked_up`, `task_completed`
  - `obstacle_added`, `obstacle_removed`
  - `message_broadcast`, `peer_state_updated`
  - `conflict_predicted` (with `timestep >= tick` so they always fire
    *before* the conflicting tick)
  - `conflict_negotiation_started`, `reservation_conflict`,
    `robot_rerouted`, `robot_yielded`, `conflict_negotiation_resolved`
- A summary including total collisions, predicted conflicts,
  negotiations started/resolved, and yielded/rerouted counts.

Options:

```bash
python -m sih26123.main --max-ticks 200 --render-every 5 --seed 7
python -m sih26123.main --headless     # log only, no ASCII render
```

---

## Run the tests

From the repository root:

```bash
python -m pytest sih26123/tests -v
```

You should see **146 tests pass** (1 skipped):

- **A\*** (10): straight path, around-wall, maze, unreachable, edges.
- **Warehouse** (12): dimensions, traversability, static/dynamic
  obstacles, neighbours, errors.
- **Robot** (13): state, step-by-step motion, battery drain, arrival,
  waiting + recovery.
- **Simulation** (13): 3-robot coexistence, task assignment, events,
  collisions, dynamic schedule, determinism, error paths.
- **Communication** (18): typed messages, FIFO bus, deterministic
  delivery, per-peer inboxes, peer-state repository.
- **Reservations** (16): own + peer reservations, sharing cells at
  the same tick, padding, removal, in-range queries.
- **Conflicts** (12): vertex detection, edge detection (swap),
  same-cell different-tick (no false positive), non-swapping movement
  (no false positive), conflict outside horizon (ignored), multiple
  conflicts, owner-only filtering.
- **Coordination** (17): inbox processing, broadcasting, independent
  peer views, prediction before collision, period-limited
  broadcasts, no-waiting guarantee, back-compat without coordination
  module.
- **Phase 2B** (35): priority formula, space-time A\*,
  wait-vs-reroute, integration (vertex, edge, deterministic
  tie-break, reservation race, message delay, regression),
  **Phase 2B.1**: pickup-to-dropoff replan broadcast correctness,
  peer stale-trajectory handling, seed-7 zero collisions.

---

## What is intentionally NOT implemented yet

- Deadlock detection / recovery
- Robot failure recovery
- Task reassignment
- Communication failure simulation
- ML / Edge AI intent prediction
- FastAPI / React / WebSockets / MQTT
- Real hardware / ROS / real LiDAR

These belong to the next layer (Phase 2C) and beyond.

---

## Architectural commitments for this stage

These rules are enforced by code structure; future stages are
expected to preserve them:

1. **The simulator is a world clock, not a controller.** It owns
   time, obstacles, the message bus and event collection; it does
   not plan paths, assign tasks, resolve conflicts, or pick
   negotiators. Decisions live in the per-robot
   `RobotCoordination` module and in `Robot.step()`.
2. **Each robot owns its own perception, peer view, reservation
   table, conflict detector and priority calculation.** There is no
   global authoritative state object.
3. **A\* is decoupled.** It takes a grid + an `is_traversable`
   callable; it never imports `Warehouse`, `Robot` or
   coordination code. The space-time variant adds a `wait` action
   and a peer-reservation filter.
4. **Conflict detection is a pure prediction function.** It answers
   "is there a predicted conflict in my local view?". It does not
   wait, reroute, or otherwise change robot behaviour — those are
   decisions made by `RobotCoordination.run_negotiation`.
5. **Negotiation is a local, deterministic function.** Same inputs
   always produce the same decision. No central conflict manager
   exists.
6. **The simulator never teleports, never suppresses collisions,
   never rolls back movement, and never silently overwrites a
   peer's reservation.** It records events; the conflict detector
   remains the authoritative auditor.
7. **Determinism is a contract.** Re-running the demo with the same
   seed produces an identical event log. The simulator owns its
   `random.Random`; the bus uses FIFO ordering keyed on publish order;
   the priority score is a pure function.

---

## Next stage (Phase 2C) — what will be added

- **Deadlock detection.** Local cycle detection among mutual
  reservation sets across peers. Detects `A waits for B waits for
  C waits for A` patterns without a global authority.
- **Failure and recovery.** Stalled-robot detection, battery
  depletion handling, and clean re-tasking.
- **Task reassignment.** When a robot fails or is stuck for too
  long, the affected task can be re-allocated (still locally,
  with priority comparison).
- **Communication failure simulation.** Drop, delay and replay
  injections to test the negotiation's robustness under imperfect
  transport.
- **Lightweight intent-prediction edge-AI model per robot.** A
  tiny per-robot module that predicts neighbours' next moves from
  observed motion. Used to pre-emptively tighten reservations
  (optional augmentation).
- **Upgrade to a real transport.** Replace `InProcessMessageBus`
  with a UDP/WebSocket/MQTT implementation. Public contracts
  (`MessageBus`, `Message`, `RobotState`) remain identical.

The simulator's role remains **exactly** what it is now: a world
clock that records what each robot decided locally.