"""Three lexical embedders, implemented here as the comparison the headline needs.

The embedder every published figure is about is the pretrained encoder in
`sentence_encoder.py`. These three exist beside it so that "a threshold is a fact
about one embedding space" can be measured rather than asserted: E5 carries a tuned
threshold between real embedders and reports what the cache does afterwards, and that
experiment needs more than one real embedding space to carry it between.

The three are chosen to differ in exactly the property that matters here, which is
how much word order survives into the vector:

``bow``       unigram TF-IDF. Order blind by construction, which makes it the clearest
              demonstration of the mechanism behind the headline: a class an embedder
              scores identically cannot be separated by any threshold at all.
``lsa``       unigram and bigram TF-IDF reduced by truncated SVD. Locally order
              sensitive, which is the behaviour most lexical retrievers have.
``charngram`` hashed character 3 to 5 grams reduced by a seeded signed random
              projection. Order sensitive across word boundaries, and robust to the
              spelling variation that defeats word level models.

All three fit their vocabulary and inverse document frequencies on the corpus they are
then measured over. ADR-002 records what that costs: their numbers are an upper bound
on what a cold vocabulary would manage, not a lower bound.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np

from .base import Embedder, l2_normalise, normalise_text


def _tokens(text: str) -> list[str]:
    return normalise_text(text).split()


def _unigrams(text: str) -> list[str]:
    return _tokens(text)


def _uni_and_bigrams(text: str) -> list[str]:
    toks = _tokens(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]


def _char_ngrams(text: str, low: int = 3, high: int = 5) -> list[str]:
    padded = f" {normalise_text(text)} "
    out: list[str] = []
    for n in range(low, high + 1):
        if len(padded) < n:
            continue
        out.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return out


class _TfidfBase(Embedder):
    """Shared fitting machinery: build a vocabulary and inverse document frequencies.

    Fitting is on the corpus the audit runs over. That is a real limitation and it is
    stated in ADR-002: the IDF weights know the corpus, so these numbers are an upper
    bound on how well a cold embedder would do, not a lower bound.
    """

    featuriser = staticmethod(_unigrams)
    min_df = 1

    def __init__(self, documents: list[str]) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._fit(documents)

    def _fit(self, documents: list[str]) -> None:
        df: Counter[str] = Counter()
        featurised = [set(self.featuriser(doc)) for doc in documents]
        for feats in featurised:
            df.update(feats)
        kept = sorted(term for term, n in df.items() if n >= self.min_df)
        self._vocab = {term: i for i, term in enumerate(kept)}
        n_docs = max(len(documents), 1)
        idf = np.zeros(len(kept), dtype=np.float32)
        for term, i in self._vocab.items():
            idf[i] = math.log((1.0 + n_docs) / (1.0 + df[term])) + 1.0
        self._idf = idf

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def _sparse_rows(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), max(len(self._vocab), 1)), dtype=np.float32)
        for r, text in enumerate(texts):
            counts = Counter(self.featuriser(text))
            for term, tf in counts.items():
                col = self._vocab.get(term)
                if col is None:
                    continue
                # Sublinear tf: a term repeated five times is not five times the signal.
                out[r, col] = (1.0 + math.log(tf)) * self._idf[col]
        return out


class BagOfWordsEmbedder(_TfidfBase):
    """Unigram TF-IDF. Order blind, and that blindness is the point."""

    name = "bow"
    featuriser = staticmethod(_unigrams)

    @property
    def dim(self) -> int:
        return max(len(self._vocab), 1)

    def encode(self, texts: list[str]) -> np.ndarray:
        return l2_normalise(self._sparse_rows(texts))


class LsaEmbedder(_TfidfBase):
    """Unigram and bigram TF-IDF reduced to a dense space by truncated SVD.

    SVD is computed once at fit time from the corpus matrix. The components are stored,
    so encoding an unseen string is a projection and not a refit, which keeps the
    embedder a function of its fitted state rather than of the batch it is handed.
    """

    name = "lsa"
    featuriser = staticmethod(_uni_and_bigrams)

    def __init__(self, documents: list[str], n_components: int = 128) -> None:
        super().__init__(documents)
        rows = self._sparse_rows(documents)
        rank = min(n_components, min(rows.shape) - 1)
        self._n_components = max(rank, 1)
        # Economy SVD on the fitted matrix. Deterministic: LAPACK gives the same
        # factorisation for the same input, and sign is pinned below so the vectors do
        # not flip between platforms.
        _, _, vt = np.linalg.svd(rows, full_matrices=False)
        components = vt[: self._n_components]
        signs = np.where(components.sum(axis=1) < 0, -1.0, 1.0).astype(np.float32)
        self._components = (components * signs[:, None]).astype(np.float32)

    @property
    def dim(self) -> int:
        return self._n_components

    def encode(self, texts: list[str]) -> np.ndarray:
        projected = self._sparse_rows(texts) @ self._components.T
        return l2_normalise(projected.astype(np.float32))


class CharNgramEmbedder(Embedder):
    """Hashed character 3 to 5 grams, reduced by a seeded signed random projection.

    No fitting. The hash is sha1 of the gram, which makes the feature index a pure
    function of the string and therefore identical on every machine and every Python
    version. Python's built in `hash` is salted per process and would have made the
    committed fixture unreproducible, which is a real trap and is pinned by a test.
    """

    name = "charngram"

    def __init__(self, n_features: int = 2**18, n_components: int = 256, seed: int = 20260227):
        self._n_features = n_features
        self._n_components = n_components
        self._seed = seed
        rng = np.random.default_rng(seed)
        # Signed random projection: +1 or -1 scaled, which preserves inner products in
        # expectation and costs one multiply per feature.
        self._projection = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32), size=(n_features, n_components)
        ) / math.sqrt(n_components)

    @property
    def dim(self) -> int:
        return self._n_components

    def _bucket(self, gram: str) -> int:
        return int.from_bytes(hashlib.sha1(gram.encode()).digest()[:8], "big") % self._n_features

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._n_components), dtype=np.float32)
        for r, text in enumerate(texts):
            counts = Counter(self._char_buckets(text))
            if not counts:
                continue
            idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
            vals = np.fromiter(
                (1.0 + math.log(v) for v in counts.values()),
                dtype=np.float32,
                count=len(counts),
            )
            out[r] = vals @ self._projection[idx]
        return l2_normalise(out)

    def _char_buckets(self, text: str) -> list[int]:
        return [self._bucket(g) for g in _char_ngrams(text)]

    def describe(self) -> dict:
        base = super().describe()
        base.update({"n_features": self._n_features, "seed": self._seed})
        return base
