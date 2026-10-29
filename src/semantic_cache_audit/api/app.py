"""The FastAPI application. Built by a factory so tests get a fresh cache each time.

A module level app would share one cache across every test in the file, and the first
test that stored an entry would change the result of the next one.
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException

from ..cache.clock import ManualClock
from ..cache.eviction import EvictionState
from ..cache.semantic_cache import SemanticCache, fresh_namespace
from ..corpus.loader import load_pairs
from ..embed.fixtures import load_fixture
from ..index.faiss_store import FaissFlatStore
from .models import (
    BatchLookupRequest,
    BatchLookupResponse,
    LookupRequest,
    LookupResponse,
    StatsResponse,
    StoreRequest,
)

#: The API answers from the committed fixture rather than by running an embedder, for
#: the same reason everything else here does: a reviewer with no model reproduces the
#: behaviour exactly. A query outside the fixture is refused with 422 rather than
#: silently embedded by a different route, which would make two endpoints disagree.
UNKNOWN_QUERY = "query is not in the committed embedding fixture for this embedder"


def build_app(embedder: str = "charngram", *, threshold: float = 0.9) -> FastAPI:
    pairs = load_pairs()
    texts, matrix = load_fixture(embedder)
    lookup_row = {text: i for i, text in enumerate(texts)}
    store = FaissFlatStore(matrix.shape[1])
    clock = ManualClock()
    # Each application gets its own namespace. Against the in process doubles two
    # applications are isolated by accident; against a real server they would share
    # every exact match pointer and every payload.
    cache = SemanticCache(
        store,
        threshold=threshold,
        clock=clock,
        eviction=EvictionState(policy="lru"),
        namespace=fresh_namespace("app"),
    )
    app = FastAPI(
        title="semantic-cache-audit",
        version="1.0.0",
        description=(
            "A semantic cache behind the interface a service would call. The lookup "
            "response deliberately carries no indication of whether the answer is "
            "right, because a real one cannot."
        ),
    )
    app.state.cache = cache
    app.state.clock = clock
    app.state.embedder = embedder
    app.state.pairs = pairs

    def vector_for(query: str) -> np.ndarray:
        row = lookup_row.get(query)
        if row is None:
            raise HTTPException(status_code=422, detail=UNKNOWN_QUERY)
        return matrix[row]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "embedder": embedder, "threshold": cache.threshold}

    @app.post("/v1/entries", status_code=201)
    def store_entry(body: StoreRequest) -> dict:
        vector = vector_for(body.query)
        key = f"entry:{body.query}"
        cache.put(key=key, query=body.query, vector=vector, answer_id=body.answer_id,
                  version=body.version)
        return {"key": key, "stored": True}

    @app.delete("/v1/entries/{key:path}")
    def delete_entry(key: str) -> dict:
        cache.evict(key)
        return {"key": key, "deleted": True}

    @app.post("/v1/lookup")
    def lookup(body: LookupRequest) -> LookupResponse:
        result = cache.lookup(body.query, vector_for(body.query))
        return LookupResponse(
            hit=result.hit,
            answer_id=result.answer_id,
            answer_version=result.answer_version,
            score=result.score,
            exact=result.exact,
        )

    @app.post("/v1/lookup/batch")
    def lookup_batch(body: BatchLookupRequest) -> BatchLookupResponse:
        """One encode pass and one index search, with the same accounting as the
        single endpoint.

        The first version skipped the hot layer entirely, so an exact repeat came
        back as an approximate hit and `/v1/stats` reported zero exact hits for any
        batch traffic. The two "they agree" tests compared only the hit flag and the
        answer, so neither could see it. The batch path is still a genuinely
        different code path, which is the point of having it, but it now goes through
        the same exact match shortcut and the same counters.
        """
        vectors = np.stack([vector_for(q) for q in body.queries])
        batched = store.query_many(vectors, top_k=1, namespace=cache.namespace)
        results: list[LookupResponse] = []
        now = clock.now()
        evicted_at_start = cache.eviction.evicted_total
        for query, matches in zip(body.queries, batched, strict=True):
            cache.stats["lookups"] += 1

            exact_key = cache.hot.exact_lookup(query)
            if exact_key is not None and cache.hot.is_known(exact_key):
                if cache.hot.is_expired(exact_key, now):
                    # Retire it, exactly as the single path does. Checking expiry
                    # without retiring left the two endpoints disagreeing on the
                    # expired counter and on the score they report for a miss.
                    cache.evict(exact_key)
                    cache.stats["misses"] += 1
                    cache.stats["expired"] += 1
                    results.append(LookupResponse(hit=False, score=None))
                    continue
                payload = cache.payloads.get(exact_key)
                if payload is not None:
                    cache.eviction.record_hit(exact_key, now)
                    cache.stats["exact_hits"] += 1
                    results.append(
                        LookupResponse(
                            hit=True,
                            answer_id=payload["answer_id"],
                            answer_version=payload["answer_version"],
                            score=1.0,
                            exact=True,
                        )
                    )
                    continue

            # Re-query rather than trust the pre-loop candidates: an eviction
            # performed earlier in this batch has to be visible to later queries in
            # it, or the batch answers from a cache that no longer exists.
            if cache.eviction.evicted_total != evicted_at_start:
                matches = store.query_many(
                    vectors[len(results) : len(results) + 1], top_k=1,
                    namespace=cache.namespace,
                )[0]
            best = matches[0] if matches else None
            if best is None or best.score < cache.threshold:
                cache.stats["misses"] += 1
                results.append(LookupResponse(hit=False, score=best.score if best else None))
                continue
            if not cache.hot.is_known(best.vector_id):
                cache.stats["misses"] += 1
                results.append(LookupResponse(hit=False, score=best.score))
                continue
            if cache.hot.is_expired(best.vector_id, now):
                cache.evict(best.vector_id)
                cache.stats["misses"] += 1
                cache.stats["expired"] += 1
                results.append(LookupResponse(hit=False, score=best.score))
                continue
            payload = cache.payloads.get(best.vector_id)
            if payload is None:
                cache.stats["misses"] += 1
                results.append(LookupResponse(hit=False, score=best.score))
                continue
            cache.eviction.record_hit(best.vector_id, now)
            cache.stats["approx_hits"] += 1
            results.append(
                LookupResponse(
                    hit=True,
                    answer_id=payload["answer_id"],
                    answer_version=payload["answer_version"],
                    score=best.score,
                    exact=False,
                )
            )
        hits = sum(1 for r in results if r.hit)
        return BatchLookupResponse(results=results, hits=hits, misses=len(results) - hits)

    @app.get("/v1/stats")
    def stats() -> StatsResponse:
        return StatsResponse(
            threshold=cache.threshold,
            entries=store.count(cache.namespace),
            lookups=cache.stats["lookups"],
            exact_hits=cache.stats["exact_hits"],
            approx_hits=cache.stats["approx_hits"],
            misses=cache.stats["misses"],
            note=(
                "hit counts say nothing about whether the answers were right. Run "
                "semantic-cache-audit to find that out."
            ),
        )

    return app
