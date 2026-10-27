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
        index = self._index_for(namespace)
        probe = np.ascontiguousarray(np.atleast_2d(vector), dtype=np.float32)
        # Over-fetch so that soft deleted rows, stale rows from an overwrite, and
        # filtered out rows cannot starve the result. Without this the store silently
        # returns fewer than top_k live neighbours once anything has been evicted,
        # which reads as a cache miss and would understate the hit rate.
        total = index.ntotal  # type: ignore[attr-defined]
        want = min(total, max(top_k * 8, top_k + 32))
        scores, rows = index.search(probe, want)  # type: ignore[attr-defined]
        out: list[Match] = []
        seen: set[str] = set()
        for score, row in zip(scores[0], rows[0], strict=True):
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
            out.append(Match(vector_id=vector_id, score=float(score), metadata=dict(meta)))
            if len(out) == top_k:
                break
        return out

    def query_many(
        self,
        vectors: np.ndarray,
        top_k: int = 1,
        namespace: str = DEFAULT_NAMESPACE,
        metadata_filter: dict | None = None,
    ) -> list[list[Match]]:
        """One FAISS search over the whole batch, then filter each row.

        This is the batch inference path the REST surface exposes. It exists because a
        per row loop pays the index setup cost once per query, and because the batch
        endpoint should be a different code path from the single endpoint rather than
        a loop over it wearing a different name. A test asserts the two agree row for
        row, which is the only thing that makes the claim worth making.
        """
        probes = np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)
        if namespace not in self._indexes or not self._live[namespace]:
            return [[] for _ in range(probes.shape[0])]
        index = self._index_for(namespace)
        total = index.ntotal  # type: ignore[attr-defined]
        want = min(total, max(top_k * 8, top_k + 32))
        scores, rows = index.search(probes, want)  # type: ignore[attr-defined]
        out: list[list[Match]] = []
        for row_scores, row_ids in zip(scores, rows, strict=True):
            matches: list[Match] = []
            seen: set[str] = set()
            for score, row in zip(row_scores, row_ids, strict=True):
                if row < 0:
                    continue
                vector_id = self._ids[namespace][row]
                if vector_id in seen or vector_id not in self._live[namespace]:
                    continue
                if self._row_of[namespace][vector_id] != row:
                    continue
                meta = self._meta[namespace][vector_id]
                if not matches_filter(meta, metadata_filter):
                    continue
                seen.add(vector_id)
                matches.append(Match(vector_id=vector_id, score=float(score), metadata=dict(meta)))
                if len(matches) == top_k:
                    break
            out.append(matches)
        return out

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
