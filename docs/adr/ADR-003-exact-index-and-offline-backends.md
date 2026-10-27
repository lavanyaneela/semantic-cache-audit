# ADR-003: An exact index, and backends with in process doubles

**Status:** accepted
**Context:** the serving stack the cache is measured through

## Decision, part one: a flat index rather than an approximate one

Use a FAISS flat inner product index. The vectors arrive L2 normalised, so an inner
product search computes cosine exactly.

The reason is methodological rather than about speed. An approximate index returns a
neighbour that is usually but not always the nearest. A false hit measured over one is
attributable either to the threshold or to the approximation, and the entire point of
this repository is that it is attributable to the threshold. Removing the second
explanation is worth more than the latency, and at this corpus size the latency is not
worth measuring anyway.

**What it costs.** Production caches run on approximate indexes, and an approximate
index changes the false hit rate in both directions: it can miss the dangerous
neighbour, which looks like an improvement, and it can return a worse neighbour above
the threshold, which is a new failure mode this repository does not measure. The
honest statement is that these numbers are a lower bound on what an approximate index
would do, not an estimate of it.

**The boring choice, written up as one.** A reranker in front of the cache would cut
the false hit rate and remove the reason the cache exists, which is to avoid paying
for a second model. This repository measures the cache as teams actually build it.

## Decision, part two: the vector store interface

The audit never calls FAISS. It calls upsert, query, delete and a batch query over
namespaces with metadata filters, which is the shape of a hosted vector database. The
indirection means the question "would this change on a managed index" is a question
about approximation rather than about rewriting the caller.

The filter language implements equality and `$in` and nothing else. A filter language
the audit does not exercise would be untested surface area pretending to be a
capability.

One detail that is not cosmetic: the query path over fetches before filtering. Soft
deleted rows, rows superseded by an overwrite, and rows excluded by a filter would
otherwise starve the result, and a short result reads as a cache miss. That
understates the hit rate in exactly the experiment that evicts. The bug was real and
is pinned in `tests/test_regression.py`.

## Decision, part three: Redis and MongoDB, doubled by default

Redis holds the exact match shortcut and the entry bookkeeping. MongoDB holds the
cached answers and the provenance that lets the audit name which stored query produced
a wrong hit. By default both run as in process doubles, `fakeredis` and `mongomock`,
so `make verify` passes offline. An optional, non blocking CI job sets `REDIS_URL` and
`MONGO_URL` to real service containers and runs the same tests against them.

**What this entitles the repository to claim.** That there is a CI job which runs
these code paths against real servers. The job is non blocking, so its failure would
not turn the badge red and its result has to be read rather than assumed; that is a
deliberate trade, because `make verify` must never depend on a service container.
Not that the project is built on Redis, and not that anything here says something
about Redis under load. A service container can never be on the critical
path, because `make verify` has to work on a laptop with the network off.

One deliberate deviation. TTL is evaluated against an injected clock rather than
handed to Redis expiry. Redis would expire on wall time, which would make a simulated
day take a day, and a test that waits is a test that flakes on a loaded runner. The
record still round trips through the server; only the comparison happens in Python.

## Why the exact match shortcut is separate from the approximate path

In production most repeat traffic is byte identical, and counting those in the
denominator would dilute the false hit rate towards zero and flatter the cache. The
rate this repository publishes is a rate over approximate matches, which is the number
a team actually needs.
