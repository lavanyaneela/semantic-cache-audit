"""The vector store interface, shaped after a hosted service rather than after FAISS.

Why the indirection exists: a semantic cache in production usually sits on a managed
vector database, and the audit's conclusions should survive that substitution. Writing
the caller against upsert, query and delete over namespaces with metadata filters
means the only thing a hosted index changes is the approximation, which is exactly the
variable ADR-003 is about.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

DEFAULT_NAMESPACE = "default"


@dataclass(frozen=True, slots=True)
class Match:
    """One neighbour returned by a query."""

    vector_id: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    """Upsert, query, delete, over namespaces, with metadata carried alongside."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict] | None = None,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> int: ...

    @abstractmethod
    def query(
        self,
        vector: np.ndarray,
        top_k: int = 1,
        namespace: str = DEFAULT_NAMESPACE,
        metadata_filter: dict | None = None,
    ) -> list[Match]: ...

    def query_many(
        self,
        vectors: np.ndarray,
        top_k: int = 1,
        namespace: str = DEFAULT_NAMESPACE,
        metadata_filter: dict | None = None,
    ) -> list[list[Match]]:
        """Default: one query per row. Backends that can batch should override it."""
        return [
            self.query(row, top_k=top_k, namespace=namespace, metadata_filter=metadata_filter)
            for row in np.atleast_2d(vectors)
        ]

    @abstractmethod
    def delete(self, ids: list[str], namespace: str = DEFAULT_NAMESPACE) -> int: ...

    @abstractmethod
    def count(self, namespace: str = DEFAULT_NAMESPACE) -> int: ...

    @abstractmethod
    def namespaces(self) -> list[str]: ...

    def describe(self) -> dict:
        return {"kind": type(self).__name__, "namespaces": self.namespaces()}


def matches_filter(metadata: dict, metadata_filter: dict | None) -> bool:
    """Equality and `$in` matching, the subset of a hosted filter language used here.

    Deliberately small. A filter language this audit does not exercise would be
    untested surface area pretending to be a capability.
    """
    if not metadata_filter:
        return True
    for key, want in metadata_filter.items():
        got = metadata.get(key)
        if isinstance(want, dict) and "$in" in want:
            if got not in want["$in"]:
                return False
        elif got != want:
            return False
    return True
