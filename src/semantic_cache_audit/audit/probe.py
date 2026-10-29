"""Runs the corpus through a cache and records what happened to each probe.

The protocol is deliberately the simplest one that reproduces the production failure:
store the answer to one side of a pair, then ask the other side. A hit is wrong when
the stored answer does not serve the asked query, which is exactly the pair's
`same_answer` label and nothing else. ADR-004 defines a wrong hit in full, because
every headline in this repository is a count of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cache.clock import ManualClock
from ..cache.eviction import EvictionState
from ..cache.semantic_cache import SemanticCache, fresh_namespace
from ..corpus.schema import Pair
from ..index.faiss_store import FaissFlatStore


@dataclass(frozen=True, slots=True)
class Probe:
    """One stored-then-asked trial, with its outcome judged against the pair label."""

    pair_id: str
    cls: str
    difficulty: str
    score: float
    hit: bool
    correct: bool
    stored_query: str
    asked_query: str

    @property
    def false_hit(self) -> bool:
        return self.hit and not self.correct

    @property
    def missed_opportunity(self) -> bool:
        """A paraphrase that should have been served from cache and was not."""
        return not self.hit and self.correct


def pair_scores(pairs: list[Pair], texts: list[str], matrix: np.ndarray) -> np.ndarray:
    """Cosine between the two sides of every pair, in corpus order.

    Computed once and reused across the whole threshold grid. Re-embedding inside the
    sweep would make the sweep's cost the headline rather than its result.
    """
    index = {t: i for i, t in enumerate(texts)}
    missing = [p.pair_id for p in pairs if p.query_a not in index or p.query_b not in index]
    if missing:
        raise KeyError(f"{len(missing)} pairs are not in the embedding fixture: {missing[:3]}")
    left = matrix[[index[p.query_a] for p in pairs]]
    right = matrix[[index[p.query_b] for p in pairs]]
    return np.einsum("ij,ij->i", left, right)


def probe_at_threshold(pairs: list[Pair], scores: np.ndarray, threshold: float) -> list[Probe]:
    """The sweep's inner loop, expressed directly on precomputed scores.

    This bypasses the cache object on purpose: with an exact index, a single entry and
    no eviction, a store-then-ask trial is exactly a comparison of the pair score
    against the threshold. `probe_through_cache` runs the same trials through the real
    cache, and a test asserts the two agree. Keeping both is what makes the fast path
    trustworthy rather than merely fast.
    """
    out: list[Probe] = []
    for pair, score in zip(pairs, scores, strict=True):
        hit = bool(score >= threshold)
        out.append(
            Probe(
                pair_id=pair.pair_id,
                cls=pair.cls,
                difficulty=pair.difficulty,
                score=float(score),
                hit=hit,
                correct=pair.same_answer,
                stored_query=pair.query_a,
                asked_query=pair.query_b,
            )
        )
    return out


def probe_through_cache(
    pairs: list[Pair], texts: list[str], matrix: np.ndarray, threshold: float
) -> list[Probe]:
    """The same trials, routed through the real store, hot layer and payload store.

    Slower, and it is the referee for the fast path above. Each pair gets its own
    namespace so one pair's stored entry cannot be returned for another pair's query,
    which would turn the measurement into a retrieval benchmark over the whole corpus
    rather than a cache correctness trial.

    The namespaces are scoped to the call, not just to the pair. Numbering them
    `pair-0` upward meant a second call reused the first call's namespaces, so
    against a shared Redis every pair started with the previous run's entry already
    stored and the referee silently stopped being a referee.
    """
    index = {t: i for i, t in enumerate(texts)}
    store = FaissFlatStore(matrix.shape[1])
    clock = ManualClock()
    run = fresh_namespace("probe")
    caches: list[SemanticCache] = []
    out: list[Probe] = []
    for i, pair in enumerate(pairs):
        cache = SemanticCache(
            store,
            threshold=threshold,
            clock=clock,
            eviction=EvictionState(),
            namespace=f"{run}-pair-{i}",
        )
        caches.append(cache)
        cache.put(
            key=f"{pair.pair_id}:a",
            query=pair.query_a,
            vector=matrix[index[pair.query_a]],
            answer_id=f"{pair.pair_id}:answer-a",
        )
        result = cache.lookup(pair.query_b, matrix[index[pair.query_b]])
        out.append(
            Probe(
                pair_id=pair.pair_id,
                cls=pair.cls,
                difficulty=pair.difficulty,
                score=float(result.score if result.score is not None else -1.0),
                hit=result.hit,
                correct=pair.same_answer,
                stored_query=pair.query_a,
                asked_query=pair.query_b,
            )
        )
    for cache in caches:
        cache.reset()
    return out
