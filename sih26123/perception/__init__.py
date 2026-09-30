"""Perception abstractions for robot agents.

A robot's perception module is local to that robot. The current
``SimulatedPerception`` reads from a read-only world snapshot exposed by
the simulator. In the next stage this will be replaced by a real sensor
or by information shared over a P2P channel between robots.
"""

from .simulated_sensor import Perception, SimulatedPerception

__all__ = ["Perception", "SimulatedPerception"]