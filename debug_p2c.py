import sys
sys.path.insert(0, '.')
import sih26123.tests.test_phase2c as t

bus = t.InProcessMessageBus()
sim = t._make_sim(num_robots=3, bus=bus)
r0, r1, r2 = sim.robots
for _ in range(2):
    sim.tick()
print(f"After 2 ticks, R1's view of R0: {r1.coordination.peer_liveness.get('R0')}")
print(f"After 2 ticks, R2's view of R0: {r2.coordination.peer_liveness.get('R0')}")
bus.drop_messages_to("R0", drop=True)
for i in range(t.PEER_FAILURE_TIMEOUT + 1):
    sim.tick()
print(f"After 41 more ticks, R1's view of R0: {r1.coordination.peer_liveness.get('R0')}")
print(f"After 41 more ticks, R2's view of R0: {r2.coordination.peer_liveness.get('R0')}")
print(f"PEER_FAILED events: {len(sim.event_log.of_type(t.EventType.PEER_FAILED))}")
for ev in sim.event_log.of_type(t.EventType.PEER_FAILED):
    print(f"  {ev.data}")
