"""Redis and MongoDB, each with an in-process double as the default.

The pattern, which ADR-003 records: the default path uses `fakeredis` and `mongomock`
so a reviewer reproduces every number offline with one command, and an optional CI job
sets `REDIS_URL` and `MONGO_URL` to real service containers and runs the same tests
against them. The claim the repository makes is therefore "these code paths run
against real servers in CI", not "this is built on Redis", which would be an overclaim.
"""

from __future__ import annotations

import json
import os
from typing import Any


def redis_client(url: str | None = None) -> Any:
    """A real client when a URL is given or `REDIS_URL` is set, otherwise fakeredis."""
    url = url or os.environ.get("REDIS_URL")
    if url:
        import redis

        return redis.Redis.from_url(url, decode_responses=True)
    import fakeredis

    return fakeredis.FakeStrictRedis(decode_responses=True)


def mongo_collection(
    url: str | None = None, database: str = "cache_audit", name: str = "payloads"
):
    """A real collection when a URL is given or `MONGO_URL` is set, otherwise mongomock."""
    url = url or os.environ.get("MONGO_URL")
    if url:
        import pymongo

        client = pymongo.MongoClient(url, serverSelectionTimeoutMS=5000)
    else:
        import mongomock

        client = mongomock.MongoClient()
    return client[database][name]


class HotLayer:
    """Redis side: exact match shortcut, entry bookkeeping, and the TTL.

    An exact repeat of a query never needs the vector path, and in production it is the
    majority of traffic. Keeping it here means the false hit rate this repository
    publishes is a rate over the approximate matches only, which is the number a team
    actually needs. Mixing exact repeats into the denominator would dilute it towards
    zero and flatter the cache.
    """

    def __init__(self, client: Any | None = None, prefix: str = "sca") -> None:
        self._r = client if client is not None else redis_client()
        self._prefix = prefix

    def _exact(self, query: str) -> str:
        return f"{self._prefix}:exact:{query}"

    def _entry(self, key: str) -> str:
        return f"{self._prefix}:entry:{key}"

    def put(self, key: str, query: str, inserted_at: float, ttl_seconds: float | None) -> None:
        self._r.set(self._exact(query), key)
        self._r.set(
            self._entry(key),
            json.dumps({"inserted_at": inserted_at, "ttl_seconds": ttl_seconds, "query": query}),
        )

    def exact_lookup(self, query: str) -> str | None:
        return self._r.get(self._exact(query))

    def entry(self, key: str) -> dict | None:
        raw = self._r.get(self._entry(key))
        return json.loads(raw) if raw else None

    def drop(self, key: str) -> None:
        record = self.entry(key)
        if record:
            self._r.delete(self._exact(record["query"]))
        self._r.delete(self._entry(key))

    def is_expired(self, key: str, now: float) -> bool:
        """TTL is evaluated against the injected clock, not against Redis expiry.

        Redis would expire on wall time, which would make a simulated day take a day.
        The real Redis path is still exercised: the record round trips through the
        server, only the comparison happens here.
        """
        record = self.entry(key)
        if record is None:
            return True
        ttl = record.get("ttl_seconds")
        if ttl is None:
            return False
        return now - float(record["inserted_at"]) >= float(ttl)

    def flush(self) -> None:
        for key in self._r.keys(f"{self._prefix}:*"):
            self._r.delete(key)


class PayloadStore:
    """MongoDB side: the cached answer and the provenance of the entry that served it.

    Provenance is not decoration. When the audit reports a wrong hit it names the
    cached query that produced it, and that attribution is only possible because the
    payload records which query the answer was stored against.
    """

    def __init__(self, collection: Any | None = None) -> None:
        self._c = collection if collection is not None else mongo_collection()

    def put(
        self, key: str, query: str, answer_id: str, answer_version: int, stored_at: float
    ) -> None:
        self._c.replace_one(
            {"_id": key},
            {
                "_id": key,
                "query": query,
                "answer_id": answer_id,
                "answer_version": answer_version,
                "stored_at": stored_at,
            },
            upsert=True,
        )

    def get(self, key: str) -> dict | None:
        return self._c.find_one({"_id": key})

    def drop(self, key: str) -> None:
        self._c.delete_one({"_id": key})

    def count(self) -> int:
        return int(self._c.count_documents({}))

    def flush(self) -> None:
        self._c.delete_many({})
