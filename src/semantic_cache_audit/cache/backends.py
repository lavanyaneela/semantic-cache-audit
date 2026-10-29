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


def _glob_escape(value: str) -> str:
    """Escape the characters Redis treats as pattern syntax in a KEYS glob.

    A namespace is caller supplied. Interpolated raw, a namespace containing `[`,
    `*` or `?` made `flush` silently match nothing, which is worse than raising.
    """
    return "".join("\\" + c if c in "*?[]\\" else c for c in value)


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

    def __init__(
        self, client: Any | None = None, prefix: str = "sca", namespace: str = "default"
    ) -> None:
        self._r = client if client is not None else redis_client()
        # The namespace is part of the key rather than a property of the caller. It
        # was not, and the omission was invisible against the in process doubles and
        # wrong against a real server: the vector store was namespaced while these
        # two were global, so two caches sharing a Redis would answer each other's
        # exact match lookups and the per pair isolation the audit relies on held
        # only by accident of using a fresh double each time.
        self._prefix = f"{prefix}:{namespace}"

    def _exact(self, query: str) -> str:
        return f"{self._prefix}:exact:{query}"

    def _entry(self, key: str) -> str:
        return f"{self._prefix}:entry:{key}"

    def _reverse(self, key: str) -> str:
        """Key to query. Lets `drop` find the exact match pointer in one read.

        Without it `drop` had to scan the keyspace for the pointer whenever the entry
        record was missing, which is a `KEYS` call in a hot path: linear in the cache
        on every eviction, and against a real Redis a blocking full keyspace scan.
        """
        return f"{self._prefix}:rev:{key}"

    def put(self, key: str, query: str, inserted_at: float, ttl_seconds: float | None) -> None:
        """Write both keys in one round trip.

        Written separately once, and a read landing between the two saw the pointer,
        found no record, concluded the entry had expired, and deleted a live entry.
        A pipeline makes the pair atomic; `is_expired` no longer treats a missing
        record as an expiry either, so neither half of that failure survives.
        """
        record = json.dumps(
            {"inserted_at": inserted_at, "ttl_seconds": ttl_seconds, "query": query}
        )
        pipeline = self._r.pipeline()
        pipeline.set(self._exact(query), key)
        pipeline.set(self._entry(key), record)
        pipeline.set(self._reverse(key), query)
        pipeline.execute()

    def exact_lookup(self, query: str) -> str | None:
        return self._r.get(self._exact(query))

    def entry(self, key: str) -> dict | None:
        raw = self._r.get(self._entry(key))
        return json.loads(raw) if raw else None

    def drop(self, key: str) -> None:
        """Remove every key this entry owns, in one round trip.

        The reverse pointer is what makes this possible without a scan. Reading the
        query out of the entry record instead failed in exactly the case that matters,
        when the record was the thing that had gone missing, and orphaned the exact
        match pointer permanently.
        """
        query = self._r.get(self._reverse(key))
        pipeline = self._r.pipeline()
        pipeline.delete(self._entry(key))
        pipeline.delete(self._reverse(key))
        if query is not None:
            pipeline.delete(self._exact(query))
        pipeline.execute()

    def is_expired(self, key: str, now: float) -> bool:
        """TTL is evaluated against the injected clock, not against Redis expiry.

        Redis would expire on wall time, which would make a simulated day take a day.
        The real Redis path is still exercised: the record round trips through the
        server, only the comparison happens here.

        A missing record is neither expired nor live: see `is_known`. Treating it as
        expired destroyed live entries, because the caller evicts on expiry;
        treating it as live served an answer that could never expire. It is a third
        state, and the caller turns it into a miss.
        """
        record = self.entry(key)
        if record is None:
            return False
        ttl = record.get("ttl_seconds")
        if ttl is None:
            return False
        return now - float(record["inserted_at"]) >= float(ttl)

    def is_known(self, key: str) -> bool:
        """Whether this entry has bookkeeping at all.

        An entry with no record cannot be reasoned about: its age is unknown, so it
        can neither be expired nor served. The caller treats it as a miss and lets
        the next write repopulate it.
        """
        return self.entry(key) is not None

    def flush(self) -> None:
        for key in self._r.keys(f"{_glob_escape(self._prefix)}:*"):
            self._r.delete(key)


class PayloadStore:
    """MongoDB side: the cached answer and the provenance of the entry that served it.

    Provenance is not decoration. When the audit reports a wrong hit it names the
    cached query that produced it, and that attribution is only possible because the
    payload records which query the answer was stored against.

    Documents are namespaced for the same reason the hot layer is: against a shared
    server an unnamespaced `_id` makes two caches one cache.
    """

    def __init__(self, collection: Any | None = None, namespace: str = "default") -> None:
        self._c = collection if collection is not None else mongo_collection()
        self._namespace = namespace

    def _id(self, key: str) -> str:
        return f"{self._namespace}\u0000{key}"

    def put(
        self, key: str, query: str, answer_id: str, answer_version: int, stored_at: float
    ) -> None:
        self._c.replace_one(
            {"_id": self._id(key)},
            {
                "_id": self._id(key),
                "namespace": self._namespace,
                "query": query,
                "answer_id": answer_id,
                "answer_version": answer_version,
                "stored_at": stored_at,
            },
            upsert=True,
        )

    def get(self, key: str) -> dict | None:
        return self._c.find_one({"_id": self._id(key)})

    def drop(self, key: str) -> None:
        self._c.delete_one({"_id": self._id(key)})

    def count(self) -> int:
        # Matched on a stored field, not on a regex over the id. A namespace is
        # caller supplied, and interpolating one containing `(`, `[` or `+` into a
        # pattern either raised or silently matched nothing.
        return int(self._c.count_documents({"namespace": self._namespace}))

    def flush(self) -> None:
        self._c.delete_many({"namespace": self._namespace})
