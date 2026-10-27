"""A FAISS backed store with an exact flat index.

Flat, not IVF or HNSW, and the reason is methodological rather than performance. An
approximate index returns a neighbour that is usually but not always the nearest, so a
measured false hit could be the cache's fault or the index's. At this corpus size the
exact search costs nothing worth measuring, and removing the confound means every
false hit this repository reports is attributable to the threshold alone. ADR-003
records what that costs and what an approximate index would change.

Vectors arrive L2 normalised from the embedder, so an inner product index computes
cosine similarity exactly.
"""

from __future__ import annotations

import numpy as np

from .vector_store import DEFAULT_NAMESPACE, Match, VectorStore, matches_filter


class FaissFlatStore(VectorStore):
    """Exact inner product search over unit vectors, with soft deletes per namespace."""

    def __init__(self, dim: int) -> None:
        import faiss

        self._faiss = faiss
        self._dim = dim
        self._indexes: dict[str, object] = {}
        self._ids: dict[str, list[str]] = {}
        self._meta: dict[str, dict[str, dict]] = {}
        self._live: dict[str, set[str]] = {}
        self._row_of: dict[str, dict[str, int]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def _index_for(self, namespace: str):
        if namespace not in self._indexes:
            self._indexes[namespace] = self._faiss.IndexFlatIP(self._dim)
            self._ids[namespace] = []
            self._meta[namespace] = {}
            self._live[namespace] = set()
            self._row_of[namespace] = {}
        return self._indexes[namespace]

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict] | None = None,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> int:
        vectors = np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)
        if vectors.shape[0] != len(ids):
            raise ValueError(f"{len(ids)} ids but {vectors.shape[0]} vectors")
        if vectors.shape[1] != self._dim:
            raise ValueError(f"store has dim {self._dim}, got {vectors.shape[1]}")
        metadata = metadata or [{} for _ in ids]
        if len(metadata) != len(ids):
            raise ValueError(f"{len(ids)} ids but {len(metadata)} metadata entries")
        index = self._index_for(namespace)
        # A repeated id is appended as a new row and the old row is retired from the
        # live set. FAISS flat has no in place update, and rebuilding the index on
        # every overwrite would make eviction experiments quadratic.
        for vector_id, meta in zip(ids, metadata, strict=True):
            self._row_of[namespace][vector_id] = len(self._ids[namespace])
            self._ids[namespace].append(vector_id)
            self._meta[namespace][vector_id] = dict(meta)
            self._live[namespace].add(vector_id)
        index.add(vectors)  # type: ignore[attr-defined]
        return len(ids)

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 1,
        namespace: str = DEFAULT_NAMESPACE,
        metadata_filter: dict | None = None,
    ) -> list[Match]:
        if namespace not in self._indexes or not self._live[namespace]:
            return []
        probe = np.ascontiguousarray(np.atleast_2d(vector), dtype=np.float32)
        return self.query_many(probe, top_k, namespace, metadata_filter)[0]

    def query_many(
        self,
        vectors: np.ndarray,
        top_k: int = 1,
        namespace: str = DEFAULT_NAMESPACE,
        metadata_filter: dict | None = None,
    ) -> list[list[Match]]:
        """One FAISS search over the whole batch, widened until the live neighbours
        are actually found.

        The single query path delegates here, so the batch and single endpoints
        cannot drift: they are the same search with a different number of rows.

        The widening is the part that matters. A fixed over-fetch was the first fix
        for soft deleted rows starving the result, and it only moved the threshold:
        with a budget of thirty three rows, thirty three dead rows ahead of the live
        one still returned nothing, which reads as a cache miss. That regime is
        reached in the eviction experiment, where it happened on 702 of 4000 requests
        in the tightest arm and moved a published figure. The search now doubles its
        fetch until it has `top_k` live matches or has exhausted the index, so the
        answer is the true nearest live neighbour regardless of how much is dead.
        """
        probes = np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)
        if namespace not in self._indexes or not self._live[namespace]:
            return [[] for _ in range(probes.shape[0])]
        index = self._index_for(namespace)
        total = index.ntotal  # type: ignore[attr-defined]
        out: list[list[Match]] = [[] for _ in range(probes.shape[0])]
        pending = list(range(probes.shape[0]))
        want = min(total, max(top_k * 8, top_k + 32))
        while pending:
            scores, rows = index.search(probes[pending], want)  # type: ignore[attr-defined]
            still_short: list[int] = []
            for position, row_scores, row_ids in zip(pending, scores, rows, strict=True):
                matches = self._collect(row_scores, row_ids, top_k, namespace, metadata_filter)
                out[position] = matches
                if len(matches) < top_k and want < total:
                    still_short.append(position)
            if want >= total:
                break
            pending = still_short
            want = min(total, want * 2)
        return out

    def _collect(
        self,
        row_scores,
        row_ids,
        top_k: int,
        namespace: str,
        metadata_filter: dict | None,
    ) -> list[Match]:
        matches: list[Match] = []
        seen: set[str] = set()
        for score, row in zip(row_scores, row_ids, strict=True):
            if row < 0:
                continue
            vector_id = self._ids[namespace][row]
            if vector_id in seen or vector_id not in self._live[namespace]:
                continue
            if self._row_of[namespace][vector_id] != row:
                continue  # a superseded row from an overwrite
            meta = self._meta[namespace][vector_id]
            if not matches_filter(meta, metadata_filter):
                continue
            seen.add(vector_id)
            matches.append(Match(vector_id=vector_id, score=float(score), metadata=dict(meta)))
            if len(matches) == top_k:
                break
        return matches

    def delete(self, ids: list[str], namespace: str = DEFAULT_NAMESPACE) -> int:
        if namespace not in self._live:
            return 0
        removed = 0
        for vector_id in ids:
            if vector_id in self._live[namespace]:
                self._live[namespace].discard(vector_id)
                removed += 1
        return removed

    def count(self, namespace: str = DEFAULT_NAMESPACE) -> int:
        return len(self._live.get(namespace, ()))

    def namespaces(self) -> list[str]:
        return sorted(self._indexes)

    def describe(self) -> dict:
        base = super().describe()
        base.update(
            {
                "index": "IndexFlatIP",
                "exact": True,
                "dim": self._dim,
                "rows_physical": {ns: len(v) for ns, v in self._ids.items()},
                "rows_live": {ns: len(v) for ns, v in self._live.items()},
            }
        )
        return base
