"""A seeded vector family with a controllable amount of answer separation.

This is not an embedder and it is never presented as one. It is an instrument, and
ADR-002 is about the distinction.

The audit this repository performs needs exactly one property of an embedder: how well
it separates "one cached answer serves both queries" from "it does not". Everything
downstream, the threshold sweep, the error budget decision, the resolution floor, is a
function of that separation and of the corpus. So this class produces vectors with a
controllable separation and nothing else, which lets the repository do two things it
otherwise could not:

1. Validate the requirement curve. The audit computes, from the corpus and a stated
   error budget, the separation an embedder would need. Running the same audit over
   this family at a range of separations checks that the predicted requirement matches
   the measured outcome. That is a test of the method, not a result about any model.
2. Generate property test inputs at a known separation, so a test can assert that the
   false hit rate falls as separation rises without depending on a fixture.

Construction, with every constant declared rather than observed:

    vector(q) = normalise( background + answer_vector(q) + noise * query_vector(q) )

`answer_vector` is shared by every query with the same answer, so at noise zero the
family is an oracle. `query_vector` is unique per query, so raising noise degrades
separation smoothly towards chance. `background` is a single shared direction that
lifts every cosine into the band real encoders occupy, because an audit tool that only
ever sees similarities near zero would not exercise its own threshold grid.

No number produced from this class is ever published without its noise level and its
measured separation beside it.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import Embedder, l2_normalise

#: Declared, not observed. Chosen so that the cosines this family produces fall in the
#: 0.6 to 1.0 band that published sentence encoder similarities typically occupy, which
#: keeps the swept threshold grid meaningful. It does not affect separation.
BACKGROUND_WEIGHT = 2.0

#: Declared. Spans measured separation from chance to perfect over this corpus, which
#: is the range the requirement curve has to be validated across.
NOISE_GRID: tuple[float, ...] = (0.0, 0.6, 1.2, 2.0, 2.6, 3.4, 4.5, 6.0, 9.0)

DEFAULT_DIM = 256
DEFAULT_SEED = 20260227


def _unit(label: str, dim: int, seed: int) -> np.ndarray:
    """A deterministic unit vector for a label, identical on every machine.

    Seeded from sha256 rather than from the built in `hash`, which is salted per
    process. A committed fixture built on a salted hash is reproducible exactly once,
    and the failure is silent. Pinned by `test_vectors_are_identical_across_processes`.
    """
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class SeparationFamily(Embedder):
    """Vectors whose answer separation is set by a single declared noise level."""

    def __init__(
        self,
        answer_of: dict[str, str],
        *,
        noise: float,
        dim: int = DEFAULT_DIM,
        seed: int = DEFAULT_SEED,
        background_weight: float = BACKGROUND_WEIGHT,
    ) -> None:
        if noise < 0.0:
            raise ValueError(f"noise must be non negative, got {noise}")
        self._noise = float(noise)
        self._dim = dim
        self._seed = seed
        self._background = _unit("background", dim, seed) * background_weight
        self._background_weight = background_weight
        self._answer_of = dict(answer_of)
        self._answer_cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"synthetic@{self._noise:g}"

    @property
    def noise(self) -> float:
        return self._noise

    @property
    def dim(self) -> int:
        return self._dim

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dim": self._dim,
            "kind": type(self).__name__,
            "noise": self._noise,
            "seed": self._seed,
            "background_weight": self._background_weight,
            "declared": True,
            "is_a_model_of_an_embedder": True,
        }

    def _answer_vector(self, text: str) -> np.ndarray:
        # A query with no declared answer gets its own, so it can never be matched to
        # anything. The instrument must not invent a hit it was not told about.
        label = self._answer_of.get(text, f"__unlabelled__{text}")
        if label not in self._answer_cache:
            self._answer_cache[label] = _unit(f"answer:{label}", self._dim, self._seed)
        return self._answer_cache[label]

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows = np.stack(
            [
                self._background
                + self._answer_vector(t)
                + self._noise * _unit(f"query:{t}", self._dim, self._seed)
                for t in texts
            ]
        )
        return l2_normalise(rows.astype(np.float32))
