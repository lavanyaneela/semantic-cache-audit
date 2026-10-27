"""The adapter for real pretrained weights, behind a flag and never on the default path.

Named `st:<model>` on the command line. It is the one route by which this repository
can say something about a real encoder, and it is deliberately the only part of the
codebase that can touch the network, so `make verify` stays offline by construction.

ADR-002 names what this adds: with real weights, the published operating point becomes
a fact about that model rather than a requirement on any model. Nothing else changes,
because every other component reads a matrix.
"""

from __future__ import annotations

import numpy as np

from .base import Embedder, l2_normalise


class LiveEmbedderUnavailable(RuntimeError):
    """sentence-transformers is not installed, or the weights are not present locally."""


class SentenceTransformerEmbedder(Embedder):
    """Thin wrapper over a sentence-transformers model. Not exercised by `make verify`."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - requires the optional extra
            raise LiveEmbedderUnavailable(
                "sentence-transformers is not installed. Install the 'live' extra: "
                "pip install -e '.[live]'"
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - requires weights on disk
            raise LiveEmbedderUnavailable(
                f"could not load {model_name!r}. The default path of this repository is "
                "offline by design; this adapter needs the weights already on disk."
            ) from exc
        self._name = model_name

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"st:{self._name}"

    @property
    def dim(self) -> int:  # pragma: no cover - requires weights on disk
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs weights
        raw = np.asarray(self._model.encode(texts, show_progress_bar=False), dtype=np.float32)
        return l2_normalise(raw)

    def describe(self) -> dict:  # pragma: no cover - requires weights on disk
        return {"name": self.name, "dim": self.dim, "kind": type(self).__name__, "live": True}
