"""Every tunable in one place, so no threshold or budget is hardcoded in a call site.

A number that only exists inside a function is a number nobody can change without
reading the function, and a portfolio reviewer reads the config file first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .audit.statistics import CONFIDENCE
from .audit.sweep import DEFAULT_NEAR_MISS_PREVALENCE


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """The inputs that move a published number, each with the reason for its default."""

    #: Share of hits that may return a wrong answer. 5 percent is deliberately loose:
    #: it is above the tightest budget this corpus can certify, which is the point.
    budget: float = 0.05

    #: A cache serving less than this is not worth its complexity, so a threshold that
    #: meets the budget by refusing almost everything does not count as a pass.
    min_hit_rate: float = 0.10

    #: Declared, not observed. See sweep.DEFAULT_NEAR_MISS_PREVALENCE.
    near_miss_prevalence: float = DEFAULT_NEAR_MISS_PREVALENCE

    #: One sided, used for every published bound.
    confidence: float = CONFIDENCE

    #: Points on the threshold grid. 101 puts the grid spacing well below the gap
    #: between adjacent pair scores, so the chosen operating point is not an artefact
    #: of the grid.
    threshold_steps: int = 101

    #: The embedder whose numbers the README quotes.
    embedder: str = "charngram"

    #: Entries a cache holds before eviction starts, used by E3.
    capacity: int = 150

    #: Seconds before a cached answer is treated as stale, used by E3.
    ttl_seconds: float = 3600.0

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULTS = AuditConfig()
