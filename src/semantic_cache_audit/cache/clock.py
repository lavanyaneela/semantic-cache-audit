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


class SystemClock:
    """Real time. Used by the API process only, never by anything that is measured."""

    def now(self) -> float:  # pragma: no cover - not exercised by the offline suite
        import time

        return time.time()
