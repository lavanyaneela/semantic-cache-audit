"""The cache under audit: exact lookup, then approximate match above a threshold.

The whole repository exists because of one property of this class. When `lookup`
returns a hit, the call site receives an answer that is shaped exactly like a correct
one. There is no field on the result that a caller could check to find out whether the
answer is right, because if such a field existed the cache would already know, and if
the cache knew it would not have returned it. A hit rate counts these and a correctness
metric cannot be derived from it. Everything in `audit/` is the external referee that
makes the distinction measurable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..index.vector_store import DEFAULT_NAMESPACE, VectorStore
from .backends import HotLayer, PayloadStore
from .clock import Clock, ManualClock
from .eviction import EvictionState


@dataclass(frozen=True, slots=True)
class Lookup:
    """What the call site gets back. Note what is absent: any notion of correctness."""

    hit: bool
    answer_id: str | None = None
    answer_version: int | None = None
    #: Which cached query produced the answer. Present so the audit can attribute a
    #: wrong hit; a production caller would have no reason to read it.
    matched_query: str | None = None
    matched_key: str | None = None
    score: float | None = None
    exact: bool = False
    #: Set when a candidate scored above threshold but its entry had expired.
    expired_candidate: bool = False


class SemanticCache:
    """Approximate match cache over a vector store, a hot layer and a payload store."""

    def __init__(
        self,
        store: VectorStore,
        *,
        threshold: float,
        hot: HotLayer | None = None,
        payloads: PayloadStore | None = None,
        clock: Clock | None = None,
        eviction: EvictionState | None = None,
        ttl_seconds: float | None = None,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        self.store = store
        self.threshold = float(threshold)
        # The namespace reaches the hot layer and the payload store, not only the
        # vector store. Against the in process doubles it makes no difference; against
        # a shared Redis or MongoDB it is the difference between isolated caches and
        # one cache wearing several names.
        self.hot = hot if hot is not None else HotLayer(namespace=namespace)
        self.payloads = payloads if payloads is not None else PayloadStore(namespace=namespace)
        self.clock = clock if clock is not None else ManualClock()
        self.eviction = eviction if eviction is not None else EvictionState()
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace
        self.stats = {"lookups": 0, "exact_hits": 0, "approx_hits": 0, "misses": 0, "expired": 0}

    # ---------------------------------------------------------------- writing

    def put(
        self, key: str, query: str, vector: np.ndarray, answer_id: str, version: int = 1
    ) -> None:
        now = self.clock.now()
        self.store.upsert([key], np.atleast_2d(vector), [{"query": query}], self.namespace)
        self.hot.put(key, query, now, self.ttl_seconds)
        self.payloads.put(key, query, answer_id, version, now)
        self.eviction.record_insert(key, now)
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        for key in self.eviction.victims():
            self.evict(key)

    def evict(self, key: str) -> None:
        self.store.delete([key], self.namespace)
        self.hot.drop(key)
        self.payloads.drop(key)
        self.eviction.forget(key)
        self.eviction.evicted_total += 1

    # ---------------------------------------------------------------- reading

    def lookup(self, query: str, vector: np.ndarray) -> Lookup:
        now = self.clock.now()
        self.stats["lookups"] += 1

        exact_key = self.hot.exact_lookup(query)
        if exact_key is not None and not self._retire_if_expired(exact_key, now):
            payload = self.payloads.get(exact_key)
            if payload is not None:
                self.eviction.record_hit(exact_key, now)
                self.stats["exact_hits"] += 1
                return Lookup(
                    hit=True,
                    answer_id=payload["answer_id"],
                    answer_version=payload["answer_version"],
                    matched_query=payload["query"],
                    matched_key=exact_key,
                    score=1.0,
                    exact=True,
                )

        matches = self.store.query(np.atleast_2d(vector), top_k=1, namespace=self.namespace)
        if not matches:
            self.stats["misses"] += 1
            return Lookup(hit=False)
        best = matches[0]
        if best.score < self.threshold:
            self.stats["misses"] += 1
            return Lookup(hit=False, score=best.score)
        if self._retire_if_expired(best.vector_id, now):
            self.stats["misses"] += 1
            self.stats["expired"] += 1
            return Lookup(hit=False, score=best.score, expired_candidate=True)
        payload = self.payloads.get(best.vector_id)
        if payload is None:
            # The vector outlived its payload. That is a bug rather than a miss, so it
            # is counted separately instead of being absorbed into the miss rate.
            self.stats["misses"] += 1
            return Lookup(hit=False, score=best.score)
        self.eviction.record_hit(best.vector_id, now)
        self.stats["approx_hits"] += 1
        return Lookup(
            hit=True,
            answer_id=payload["answer_id"],
            answer_version=payload["answer_version"],
            matched_query=payload["query"],
            matched_key=best.vector_id,
            score=best.score,
            exact=False,
        )

    def _retire_if_expired(self, key: str, now: float) -> bool:
        """Evict only on a real expiry.

        A missing hot layer record used to count as an expiry, which meant a read
        could destroy a live entry whenever the record was absent for any reason. It
        now reports as not expired, and the caller's payload lookup turns an entry
        with no payload into a miss rather than a deletion.
        """
        if not self.hot.is_expired(key, now):
            return False
        self.evict(key)
        return True

    def describe(self) -> dict:
        return {
            "threshold": self.threshold,
            "ttl_seconds": self.ttl_seconds,
            "eviction_policy": self.eviction.policy,
            "capacity": self.eviction.capacity,
            "namespace": self.namespace,
            "store": self.store.describe(),
            "stats": dict(self.stats),
        }
