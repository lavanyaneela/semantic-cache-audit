"""Builds an embedder from its name, and derives the intent labels the model needs.

One place that knows every embedder name, so the CLI, the experiments and the fixture
writer cannot drift apart about what `lsa` means.
"""

from __future__ import annotations

from ..corpus.schema import Pair
from .base import Embedder
from .lexical import BagOfWordsEmbedder, CharNgramEmbedder, LsaEmbedder
from .synthetic import NOISE_GRID, SeparationFamily

#: Embedders that are real functions of the text alone.
LEXICAL_NAMES: tuple[str, ...] = ("bow", "lsa", "charngram")

#: Noise levels of the separation family, re-exported so callers have one import site.
SYNTHETIC_GRID: tuple[float, ...] = NOISE_GRID

#: Every embedder name `make embeddings` writes a fixture for.
def fixture_names() -> list[str]:
    return [*LEXICAL_NAMES, *(f"synthetic@{n:g}" for n in SYNTHETIC_GRID)]


class UnknownEmbedder(ValueError):
    """The name does not match any embedder this repository knows how to build."""


def corpus_documents(pairs: list[Pair]) -> list[str]:
    """Every query string in the corpus, in a stable order, deduplicated.

    The lexical embedders fit their vocabulary and IDF weights on this. ADR-002 states
    the cost: the weights know the corpus, so lexical numbers here are an upper bound
    on what a cold vocabulary would achieve, not a lower bound.
    """
    seen: dict[str, None] = {}
    for pair in pairs:
        seen.setdefault(pair.query_a, None)
        seen.setdefault(pair.query_b, None)
    return list(seen)


class CorpusInconsistent(ValueError):
    """A near-miss pair's two sides are joined by a chain of paraphrase pairs.

    If A paraphrases B and B paraphrases C, then A and C must have the same answer. A
    near-miss pair claiming A and C differ contradicts that, and the contradiction has
    to be an authoring error rather than something to average over.
    """


def _components(pairs: list[Pair]) -> dict[str, str]:
    """Union-find over the paraphrase relation, returning query -> component root.

    "One answer serves both" is an equivalence relation, so intent cannot be assigned
    pair by pair. It was, in the first version of this module, and because 14 of the
    626 distinct queries appear in two different pairs the second assignment silently
    overwrote the first. The bug is pinned by
    `test_a_query_in_two_pairs_keeps_one_intent_per_answer_class`.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for pair in pairs:
        find(pair.query_a)
        find(pair.query_b)
    for pair in pairs:
        if pair.same_answer:
            union(pair.query_a, pair.query_b)
    return {q: find(q) for q in parent}


def intent_labels(pairs: list[Pair]) -> dict[str, str]:
    """Map each query to the answer it needs, as an equivalence class over the corpus.

    Two queries share a label exactly when the paraphrase relation connects them. Every
    near-miss pair is then checked for consistency: its two sides must land in
    different classes, or the corpus contradicts itself.
    """
    roots = _components(pairs)
    contradictions = [
        p.pair_id for p in pairs if p.is_near_miss and roots[p.query_a] == roots[p.query_b]
    ]
    if contradictions:
        raise CorpusInconsistent(
            f"{len(contradictions)} near-miss pairs have both sides in one answer class: "
            f"{contradictions[:5]}"
        )
    return dict(roots)


def shared_queries(pairs: list[Pair]) -> int:
    """How many queries appear in more than one pair. Reported in the fixture manifest."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for pair in pairs:
        counts[pair.query_a] += 1
        counts[pair.query_b] += 1
    return sum(1 for n in counts.values() if n > 1)


def parse_noise(name: str) -> float:
    """`synthetic@2.6` -> 2.6. A bare `synthetic` is refused, because an unnamed noise
    level would let a published number lose the parameter it depends on."""
    if name == "synthetic":
        raise UnknownEmbedder(
            "write synthetic@<noise>, for example synthetic@2.6. A bare 'synthetic' is "
            "refused because every number from this family must carry its noise level."
        )
    try:
        return float(name.split("@", 1)[1])
    except (IndexError, ValueError) as exc:
        raise UnknownEmbedder(f"{name!r}: noise must be a non negative number") from exc


def build(name: str, pairs: list[Pair]) -> Embedder:
    """Construct the named embedder, fitting it on the corpus where it needs fitting."""
    docs = corpus_documents(pairs)
    if name == "bow":
        return BagOfWordsEmbedder(docs)
    if name == "lsa":
        return LsaEmbedder(docs)
    if name == "charngram":
        return CharNgramEmbedder()
    if name.startswith("synthetic"):
        return SeparationFamily(intent_labels(pairs), noise=parse_noise(name))
    if name.startswith("st:"):
        from .live import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(name[3:])
    raise UnknownEmbedder(
        f"unknown embedder {name!r}. Known: {', '.join(LEXICAL_NAMES)}, "
        "synthetic@<noise>, st:<model-name>"
    )
