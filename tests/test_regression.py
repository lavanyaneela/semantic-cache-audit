"""Tests that pin bugs actually hit during the build, each with a counter test.

A regression test that passes against the bug it was written for is worse than no
test, because it is believed. Every pinning test here is followed by a counter test
that reconstructs the broken implementation and asserts the pinning test rejects it.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_cache_audit.audit.budget import Exit, decide
from semantic_cache_audit.audit.statistics import trials_needed_for
from semantic_cache_audit.audit.sweep import run_sweep
from semantic_cache_audit.corpus.schema import Pair
from semantic_cache_audit.embed.fixtures import load_fixture
from semantic_cache_audit.embed.registry import (
    PRETRAINED_NAME,
    CorpusInconsistent,
    intent_labels,
)


@pytest.fixture(scope="module")
def headline(pairs):
    texts, matrix = load_fixture(PRETRAINED_NAME)
    return run_sweep(pairs, texts, matrix, PRETRAINED_NAME)


@pytest.fixture(scope="module")
def oracle(pairs):
    texts, matrix = load_fixture("synthetic@0")
    return run_sweep(pairs, texts, matrix, "synthetic@0")


# ---------------------------------------------------------------------------
# Bug one: intent labelling could not express the invariant it depended on.
#
# The first implementation walked the pairs and wrote `labels[query] = ...` for each
# side. Fourteen of the 626 distinct queries appear in two different pairs, so the
# second write silently replaced the first. Investigating that turned out to be more
# interesting than the overwrite itself: on this corpus the result stayed pairwise
# consistent, so nothing measured was wrong. What was wrong is that the per pair form
# cannot represent the property the labels are supposed to carry. "One answer serves
# both" is an equivalence relation, so if A paraphrases B and B paraphrases C, then a
# near miss pair claiming A and C differ is a contradiction. The per pair loop cannot
# see that, has nothing to check, and would have carried the contradiction into every
# number built on it. The replacement is a union find that computes answer classes and
# then checks every near miss against them.
#
# The honest summary: this was not a wrong number, it was a missing invariant, and the
# test below is the one that would have caught the wrong number if the corpus had ever
# grown one.
# ---------------------------------------------------------------------------


def _broken_intent_labels(pairs) -> dict[str, str]:
    """The original per pair implementation, reconstructed for the counter test."""
    labels: dict[str, str] = {}
    for pair in pairs:
        if pair.same_answer:
            labels[pair.query_a] = f"{pair.pair_id}:shared"
            labels[pair.query_b] = f"{pair.pair_id}:shared"
        else:
            labels[pair.query_a] = f"{pair.pair_id}:a"
            labels[pair.query_b] = f"{pair.pair_id}:b"
    return labels


def _contradicting_corpus(pairs) -> list[Pair]:
    """A near miss whose two sides are joined by a paraphrase edge."""
    near = next(p for p in pairs if p.is_near_miss)
    bridge = Pair(
        pair_id="bridge", cls="true_paraphrase", domain=near.domain,
        query_a=near.query_a, query_b=near.query_b, same_answer=True,
        difficulty="easy", template_id="bridge", provenance="regression test",
    )
    return [*pairs, bridge]


def test_a_corpus_that_contradicts_itself_is_detected(pairs):
    """The pinning test."""
    with pytest.raises(CorpusInconsistent):
        intent_labels(_contradicting_corpus(pairs))


def test_the_contradiction_test_fails_against_the_implementation_that_had_the_bug(pairs):
    """The counter test: the per pair loop accepts the contradiction in silence."""
    labels = _broken_intent_labels(_contradicting_corpus(pairs))
    assert isinstance(labels, dict) and labels


def test_a_query_in_two_pairs_resolves_to_a_single_answer_class(pairs):
    labels = intent_labels(pairs)
    assert len(set(labels.values())) < len(labels)


def test_answer_classes_are_transitive_across_paraphrase_chains(pairs):
    """What the per pair form could not represent, stated directly."""
    labels = intent_labels(pairs)
    for pair in pairs:
        if pair.same_answer:
            assert labels[pair.query_a] == labels[pair.query_b]


# ---------------------------------------------------------------------------
# Bug two: the refusal gave the right verdict for the wrong reason, and so gave the
# wrong verdict.
#
# Exit 2 refuses when the corpus cannot resolve the requested budget. The first
# version compared the required hit count against the most hits any threshold
# produced, which is every probe in the corpus. That count includes near misses, and a
# near miss that hits is a wrong answer, so it can never be part of a clean run. At a
# one percent budget the ceiling test therefore passed, no threshold qualified, and
# the tool returned exit 1: "no threshold meets your budget."
#
# That is the wrong answer even though it is not a wrong number. Exit 1 tells a team
# their cache is not good enough, which sends them to tune the embedder. Exit 2 tells
# them their evidence cannot settle the question at that budget, which sends them to
# build a bigger corpus. The correct ceiling is the positive control count: 120 here,
# against the 299 clean hits a one percent budget needs.
# ---------------------------------------------------------------------------


def _broken_ceiling_refuses(sweep, budget: float) -> bool:
    """The original ceiling: most hits at any threshold, near misses included."""
    return sweep.max_hits < trials_needed_for(budget)


def test_a_budget_needing_more_clean_hits_than_there_are_controls_is_refused(oracle, pairs):
    """The pinning test, run on a perfectly separating family so nothing else can refuse."""
    positive_controls = sum(1 for p in pairs if p.same_answer)
    assert trials_needed_for(0.01) > positive_controls
    assert decide(oracle, budget=0.01).exit_code is Exit.REFUSED


def test_the_refusal_test_fails_against_the_ceiling_that_had_the_bug(oracle):
    """The counter test: the old ceiling does not refuse, so exit 1 would be returned."""
    assert not _broken_ceiling_refuses(oracle, 0.01)


def test_the_old_ceiling_counted_hits_that_could_never_have_been_clean(oracle, pairs):
    """Records the gap that caused it: 320 against 120."""
    assert oracle.max_hits == len(pairs)
    assert sum(1 for p in pairs if p.same_answer) < oracle.max_hits


def test_refusal_is_a_property_of_the_corpus_not_of_the_embedder(headline, oracle):
    """The point of the fix: an inverted encoder and a perfect one refuse alike."""
    assert decide(headline, budget=0.01).exit_code is Exit.REFUSED
    assert decide(oracle, budget=0.01).exit_code is Exit.REFUSED


# ---------------------------------------------------------------------------
# Bug three: a soft deleted row silently shortened every result set.
#
# The store returns the top k live neighbours. The first version asked FAISS for
# exactly k rows and then filtered out deleted, superseded and metadata excluded ones,
# so after any eviction a query came back short. In the eviction sweep a short result
# reads as a cache miss, which understates the hit rate in precisely the configuration
# the experiment was there to measure. The fix over fetches before filtering.
# ---------------------------------------------------------------------------


def test_a_result_set_keeps_its_length_after_an_eviction(unit_vectors):
    """The pinning test."""
    from semantic_cache_audit.index.faiss_store import FaissFlatStore

    store = FaissFlatStore(unit_vectors.shape[1])
    store.upsert([f"v{i}" for i in range(len(unit_vectors))], unit_vectors)
    before = len(store.query(unit_vectors[0], top_k=4))
    store.delete(["v0", "v1"])
    assert len(store.query(unit_vectors[0], top_k=4)) == before


def test_the_result_length_test_fails_against_the_fetch_size_that_had_the_bug(unit_vectors):
    """The counter test: fetch exactly k, then filter, and rows go missing."""
    import faiss

    index = faiss.IndexFlatIP(unit_vectors.shape[1])
    index.add(np.ascontiguousarray(unit_vectors))
    deleted = {0, 1}
    _, rows = index.search(np.ascontiguousarray(unit_vectors[:1]), 4)
    assert len([r for r in rows[0] if r not in deleted]) < 4


def test_a_batch_query_keeps_its_length_after_an_eviction_too(unit_vectors):
    """The same invariant on the second implementation, which could have drifted."""
    from semantic_cache_audit.index.faiss_store import FaissFlatStore

    store = FaissFlatStore(unit_vectors.shape[1])
    store.upsert([f"v{i}" for i in range(len(unit_vectors))], unit_vectors)
    before = [len(r) for r in store.query_many(unit_vectors[:3], top_k=4)]
    store.delete(["v0", "v1"])
    assert [len(r) for r in store.query_many(unit_vectors[:3], top_k=4)] == before
