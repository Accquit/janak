"""Phase 2C failure and resilience constants.

Single source of truth for the master-plan timing values. All other
modules must read from here, never hard-code these numbers.
"""

from __future__ import annotations

# --- Heartbeats / peer liveness ---------------------------------------
# Robots broadcast a state/heartbeat every HEARTBEAT_INTERVAL ticks.
HEARTBEAT_INTERVAL: int = 5

# After this many ticks without a valid peer broadcast, the peer is
# considered *stale*: we stop trusting its forward reservations and
# treat its last known path cells as static obstacles.
STALE_THRESHOLD: int = 15

# After this many ticks without a valid peer broadcast, the peer is
# declared failed/offline: its task is released back to the pool and
# its last known occupied cell is held indefinitely.
PEER_FAILURE_TIMEOUT: int = 40

# --- Self isolation --------------------------------------------------
# If a robot has not received enough valid peer state for this many
# ticks, it enters SAFE_HALT: it stops making unsafe autonomous
# progress that depends on stale peer beliefs. Local safety checks
# (motion, perception, charging) continue to run. Recovery is
# automatic when communication resumes.
SELF_ISOLATION_TIMEOUT: int = 20

# --- Deadlock detection ---------------------------------------------
# A wait-for cycle (A waits for B waits for ... waits for A) is
# considered a deadlock if the cycle has been continuously present
# for at least DEADLOCK_WINDOW ticks. Cycles below this window are
# tolerated as transient lock-ups.
DEADLOCK_WINDOW: int = 25
