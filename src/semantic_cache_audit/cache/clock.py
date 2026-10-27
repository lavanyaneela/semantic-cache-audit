"""An injected clock, because a test that reads the wall clock is a test that flakes.

Nothing in this package calls `datetime.now` or `time.time`. A loaded CI runner turns
a passing time dependent test into an intermittent failure, and an intermittent
failure in a repository whose whole claim is machine checked numbers costs more than
the feature it was testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


@dataclass
class ManualClock:
    """A clock that only moves when something moves it."""

    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("time does not run backwards in this simulation")
        self.t += seconds
        return self.t


# There is deliberately no wall clock implementation here. Every path in this
# repository, the API included, runs on an injected clock, so a published number can
# never depend on how long something took. A real deployment would supply its own.
