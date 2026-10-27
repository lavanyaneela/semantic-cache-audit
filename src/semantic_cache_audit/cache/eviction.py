"""Eviction policies, as pure functions over recorded access state.

Kept separate from the cache so E3 can sweep policies without touching the matching
logic, and so a policy is testable without standing up a store.
"""

from __future__ import annotations

from dataclasses import dataclass, field

POLICIES: tuple[str, ...] = ("lru", "lfu", "fifo", "ttl_only")


@dataclass
class EntryStats:
    """What a policy is allowed to see about one cache entry."""

    key: str
    inserted_at: float
    last_seen_at: float
    hits: int = 0


@dataclass
class EvictionState:
    """Access bookkeeping, owned here so the cache does not grow a second copy."""

    policy: str = "lru"
    capacity: int | None = None
    entries: dict[str, EntryStats] = field(default_factory=dict)
    evicted_total: int = 0

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unknown eviction policy {self.policy!r}, known: {POLICIES}")
        if self.capacity is not None and self.capacity < 1:
            raise ValueError("capacity must be at least one entry")

    def record_insert(self, key: str, now: float) -> None:
        self.entries[key] = EntryStats(key=key, inserted_at=now, last_seen_at=now)

    def record_hit(self, key: str, now: float) -> None:
        stats = self.entries.get(key)
        if stats is None:
            return
        stats.last_seen_at = now
        stats.hits += 1

    def forget(self, key: str) -> None:
        self.entries.pop(key, None)

    def victims(self) -> list[str]:
        """Keys to evict so the entry count is back inside capacity."""
        if self.capacity is None or len(self.entries) <= self.capacity:
            return []
        overflow = len(self.entries) - self.capacity
        ordered = self._ranked()
        return ordered[:overflow]

    def _ranked(self) -> list[str]:
        """Worst first. The key breaks ties so eviction is deterministic."""
        if self.policy == "lru":
            return [
                s.key for s in sorted(self.entries.values(), key=lambda s: (s.last_seen_at, s.key))
            ]
        if self.policy == "lfu":
            return [
                s.key
                for s in sorted(self.entries.values(), key=lambda s: (s.hits, s.inserted_at, s.key))
            ]
        if self.policy == "fifo":
            return [
                s.key for s in sorted(self.entries.values(), key=lambda s: (s.inserted_at, s.key))
            ]
        # ttl_only never evicts on capacity. It is in the policy set precisely so E3
        # can price what unbounded growth buys in hit rate and costs in staleness.
        return []
