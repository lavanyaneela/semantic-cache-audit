"""A seeded request stream, so eviction and staleness can be measured rather than argued.

E3 asks whether an eviction policy changes correctness or only cost. Answering it needs
a stream with three properties a static corpus does not have: a popularity skew, so
eviction has something to get wrong; answers that change at known times, so staleness
is observable; and a fixed seed, so the published numbers do not move between runs.

Time is simulated through the injected clock. Nothing here sleeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..cache.clock import ManualClock
from ..cache.eviction import EvictionState
from ..cache.semantic_cache import SemanticCache, fresh_namespace
from ..corpus.schema import Pair
from ..index.faiss_store import FaissFlatStore


@dataclass(frozen=True, slots=True)
class Request:
    """One arrival: a query, the answer it needs right now, and when it arrived."""

    at: float
    query: str
    answer_key: str
    pair_id: str
    cls: str
    side: str


@dataclass
class StreamResult:
    requests: int = 0
    hits: int = 0
    false_hits: int = 0
    stale_hits: int = 0
    misses: int = 0
    evictions: int = 0
    by_class: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def false_hit_rate(self) -> float:
        return self.false_hits / self.hits if self.hits else 0.0

    @property
    def stale_hit_rate(self) -> float:
        return self.stale_hits / self.hits if self.hits else 0.0

    @property
    def wrong_or_stale_rate(self) -> float:
        """The number a caller actually experiences: served an answer it should not have."""
        return (self.false_hits + self.stale_hits) / self.hits if self.hits else 0.0

    def to_dict(self) -> dict:
        return {
            "requests": self.requests,
            "hits": self.hits,
            "false_hits": self.false_hits,
            "stale_hits": self.stale_hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "false_hit_rate": self.false_hit_rate,
            "stale_hit_rate": self.stale_hit_rate,
            "wrong_or_stale_rate": self.wrong_or_stale_rate,
            "by_class": self.by_class,
        }


def answer_key_of(pair: Pair, side: str) -> str:
    """The answer a query needs. Both sides of a positive control need the same one."""
    return f"{pair.pair_id}:shared" if pair.same_answer else f"{pair.pair_id}:{side}"


def build_stream(
    pairs: list[Pair],
    *,
    n_requests: int = 4000,
    seed: int = 20260227,
    zipf_a: float = 1.3,
    arrival_seconds: float = 2.0,
) -> list[Request]:
    """A Zipf skewed stream over both sides of every pair.

    Zipf rather than uniform because a uniform stream makes every eviction policy
    identical by construction, which would turn E3 into a tautology.
    """
    rng = np.random.default_rng(seed)
    sides = [(p, "a", p.query_a) for p in pairs] + [(p, "b", p.query_b) for p in pairs]
    order = rng.permutation(len(sides))
    ranks = rng.zipf(zipf_a, size=n_requests) - 1
    chosen = order[np.minimum(ranks, len(sides) - 1)]
    out: list[Request] = []
    now = 0.0
    for idx in chosen:
        pair, side, query = sides[int(idx)]
        now += arrival_seconds
        out.append(
            Request(
                at=now,
                query=query,
                answer_key=answer_key_of(pair, side),
                pair_id=pair.pair_id,
                cls=pair.cls,
                side=side,
            )
        )
    return out


def change_schedule(
    pairs: list[Pair], *, horizon: float, seed: int = 20260227, changes_per_answer: float = 1.5
) -> dict[str, list[float]]:
    """When each answer's content changes. Staleness is only measurable against this."""
    rng = np.random.default_rng(seed + 1)
    schedule: dict[str, list[float]] = {}
    for pair in pairs:
        for side in ("a", "b"):
            key = answer_key_of(pair, side)
            if key in schedule:
                continue
            n = rng.poisson(changes_per_answer)
            schedule[key] = sorted(float(t) for t in rng.uniform(0.0, horizon, size=n))
    return schedule


def version_at(schedule: dict[str, list[float]], key: str, when: float) -> int:
    """Version 1 before the first change, then one more after each."""
    return 1 + sum(1 for t in schedule.get(key, ()) if t <= when)


def replay(
    requests: list[Request],
    texts: list[str],
    matrix: np.ndarray,
    *,
    threshold: float,
    policy: str,
    capacity: int | None,
    ttl_seconds: float | None,
    schedule: dict[str, list[float]],
) -> StreamResult:
    """Run the stream through a real cache and judge every hit against ground truth.

    Two ways a hit can be wrong, and they are counted separately because the fixes are
    different. A false hit returned the answer to a different question, which a
    threshold change addresses. A stale hit returned the right question's answer from
    before it changed, which only a TTL addresses. Reporting one number for both would
    hide which lever to pull.

    The run takes a namespace of its own and clears it when it finishes. Without that
    the hot layer and the payload store carry over between calls, and E3 makes
    nineteen of them in a row: against a real Redis and MongoDB the second replay
    starts warm and the published figures become order dependent.
    """
    row_of = {text: i for i, text in enumerate(texts)}
    store = FaissFlatStore(matrix.shape[1])
    clock = ManualClock()
    cache = SemanticCache(
        store,
        threshold=threshold,
        clock=clock,
        eviction=EvictionState(policy=policy, capacity=capacity),
        ttl_seconds=ttl_seconds,
        namespace=fresh_namespace("replay"),
    )
    try:
        return _replay_into(cache, clock, requests, row_of, matrix, schedule)
    finally:
        cache.reset()


def _replay_into(
    cache: SemanticCache,
    clock: ManualClock,
    requests: list[Request],
    row_of: dict[str, int],
    matrix: np.ndarray,
    schedule: dict[str, list[float]],
) -> StreamResult:
    """The replay loop itself, so the caller can wrap it in setup and cleanup."""
    result = StreamResult()
    for request in requests:
        clock.t = request.at
        result.requests += 1
        bucket = result.by_class.setdefault(
            request.cls, {"requests": 0, "hits": 0, "false_hits": 0, "stale_hits": 0}
        )
        bucket["requests"] += 1
        vector = matrix[row_of[request.query]]
        got = cache.lookup(request.query, vector)
        if not got.hit:
            result.misses += 1
            cache.put(
                key=f"k:{request.query}",
                query=request.query,
                vector=vector,
                answer_id=request.answer_key,
                version=version_at(schedule, request.answer_key, request.at),
            )
            continue
        result.hits += 1
        bucket["hits"] += 1
        if got.answer_id != request.answer_key:
            result.false_hits += 1
            bucket["false_hits"] += 1
        elif got.answer_version != version_at(schedule, request.answer_key, request.at):
            result.stale_hits += 1
            bucket["stale_hits"] += 1
    result.evictions = cache.eviction.evicted_total
    return result
