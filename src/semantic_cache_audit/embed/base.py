"""The embedder contract, and the normalisation rule every implementation shares.

Vectors are L2 normalised at the boundary so that an inner product index computes
cosine similarity exactly. Doing it here rather than in the index means a similarity
threshold means the same thing whichever store it is evaluated against.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalise_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Deliberately shared by every embedder. If one embedder normalised differently its
    similarity numbers would not be comparable with the others, which would quietly
    invalidate E5, the one experiment whose whole subject is the comparison.
    """
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction. Leaving the norm at zero would produce NaN and
    # silently poison every downstream similarity, so it is mapped to a unit basis
    # vector instead and the condition is observable through `zero_rows`.
    norms[norms == 0.0] = 1.0
    return matrix / norms


class Embedder(ABC):
    """Turns text into unit vectors. Deterministic for a fixed `name`."""

    #: Stable identifier. It names the committed fixture file and appears in every
    #: published number that depends on the embedder, so it must not drift.
    name: str = "base"

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2 normalised rows."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def describe(self) -> dict:
        return {"name": self.name, "dim": self.dim, "kind": type(self).__name__}


def zero_rows(matrix: np.ndarray) -> int:
    """How many rows had no direction before normalisation. Reported, never hidden."""
    return int((np.linalg.norm(matrix, axis=1) == 0.0).sum())
