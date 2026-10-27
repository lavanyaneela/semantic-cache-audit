"""Properties of the vector store, including the ones eviction depends on."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_cache_audit.index.faiss_store import FaissFlatStore
from semantic_cache_audit.index.vector_store import Match, matches_filter


@pytest.fixture
def store(unit_vectors):
    s = FaissFlatStore(unit_vectors.shape[1])
    s.upsert(
        [f"v{i}" for i in range(len(unit_vectors))],
        unit_vectors,
        [{"group": "a" if i % 2 == 0 else "b", "n": i} for i in range(len(unit_vectors))],
    )
    return s


def test_upserting_returns_the_number_of_vectors_written(store, unit_vectors):
    assert store.upsert(["extra"], unit_vectors[:1]) == 1


def test_a_query_for_a_stored_vector_returns_that_vector_first(store, unit_vectors):
    matches = store.query(unit_vectors[3], top_k=1)
    assert matches[0].vector_id == "v3"


def test_a_query_for_a_stored_vector_scores_it_at_one(store, unit_vectors):
    assert store.query(unit_vectors[3], top_k=1)[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_cosine_because_the_vectors_are_unit_length(store, unit_vectors):
    expected = float(unit_vectors[0] @ unit_vectors[1])
    found = [m for m in store.query(unit_vectors[0], top_k=12) if m.vector_id == "v1"]
    assert found[0].score == pytest.approx(expected, abs=1e-5)


def test_results_come_back_in_descending_score_order(store, unit_vectors):
    scores = [m.score for m in store.query(unit_vectors[0], top_k=6)]
    assert scores == sorted(scores, reverse=True)


def test_a_query_never_returns_the_same_identifier_twice(store, unit_vectors):
    ids = [m.vector_id for m in store.query(unit_vectors[0], top_k=12)]
    assert len(ids) == len(set(ids))


def test_a_query_against_an_empty_namespace_returns_nothing(store, unit_vectors):
    assert store.query(unit_vectors[0], namespace="never-written") == []


def test_a_query_against_a_namespace_whose_entries_are_all_deleted_returns_nothing(
    store, unit_vectors
):
    store.delete([f"v{i}" for i in range(len(unit_vectors))])
    assert store.query(unit_vectors[0]) == []


def test_deleting_reports_how_many_identifiers_it_removed(store):
    assert store.delete(["v0", "v1", "not-there"]) == 2


def test_deleting_from_an_unknown_namespace_removes_nothing(store):
    assert store.delete(["v0"], namespace="nope") == 0


def test_a_deleted_vector_is_never_returned_again(store, unit_vectors):
    store.delete(["v3"])
    ids = [m.vector_id for m in store.query(unit_vectors[3], top_k=12)]
    assert "v3" not in ids


def test_deleting_one_vector_still_returns_a_full_result_set(store, unit_vectors):
    """Pins the over-fetch. Without it a soft delete silently shortens every result."""
    before = len(store.query(unit_vectors[0], top_k=5))
    store.delete(["v0"])
    assert len(store.query(unit_vectors[0], top_k=5)) == before


def test_the_live_count_falls_when_a_vector_is_deleted(store):
    before = store.count()
    store.delete(["v0"])
    assert store.count() == before - 1


def test_overwriting_an_identifier_returns_the_new_vector(store, unit_vectors):
    replacement = unit_vectors[7]
    store.upsert(["v0"], replacement[None, :], [{"group": "c"}])
    match = next(m for m in store.query(replacement, top_k=12) if m.vector_id == "v0")
    assert match.score == pytest.approx(1.0, abs=1e-5)


def test_overwriting_an_identifier_replaces_its_metadata(store, unit_vectors):
    store.upsert(["v0"], unit_vectors[7][None, :], [{"group": "c"}])
    match = next(m for m in store.query(unit_vectors[7], top_k=12) if m.vector_id == "v0")
    assert match.metadata["group"] == "c"


def test_overwriting_an_identifier_does_not_change_the_live_count(store):
    before = store.count()
    store.upsert(["v0"], np.zeros((1, store.dim), dtype=np.float32))
    assert store.count() == before


def test_namespaces_are_isolated_from_one_another(store, unit_vectors):
    store.upsert(["other"], unit_vectors[0][None, :], namespace="second")
    assert [m.vector_id for m in store.query(unit_vectors[0], namespace="second")] == ["other"]


def test_the_store_lists_the_namespaces_it_holds(store, unit_vectors):
    store.upsert(["other"], unit_vectors[0][None, :], namespace="second")
    assert store.namespaces() == ["default", "second"]


def test_a_metadata_filter_restricts_the_result(store, unit_vectors):
    matches = store.query(unit_vectors[0], top_k=12, metadata_filter={"group": "b"})
    assert all(m.metadata["group"] == "b" for m in matches)


def test_an_in_filter_accepts_any_listed_value(store, unit_vectors):
    matches = store.query(unit_vectors[0], top_k=12, metadata_filter={"n": {"$in": [1, 3]}})
    assert {m.metadata["n"] for m in matches} <= {1, 3}


def test_an_empty_filter_matches_everything():
    assert matches_filter({"a": 1}, None) and matches_filter({"a": 1}, {})


def test_a_filter_on_an_absent_key_matches_nothing():
    assert not matches_filter({"a": 1}, {"b": 2})


def test_upserting_with_the_wrong_number_of_vectors_is_rejected(store, unit_vectors):
    with pytest.raises(ValueError, match="ids but"):
        store.upsert(["a", "b"], unit_vectors[:1])


def test_upserting_with_the_wrong_dimension_is_rejected(store):
    with pytest.raises(ValueError, match="dim"):
        store.upsert(["a"], np.zeros((1, 99), dtype=np.float32))


def test_upserting_with_mismatched_metadata_is_rejected(store, unit_vectors):
    with pytest.raises(ValueError, match="metadata entries"):
        store.upsert(["a", "b"], unit_vectors[:2], [{}])


def test_a_batch_query_agrees_with_the_single_query_row_for_row(store, unit_vectors):
    """The batch path is a different implementation, so the claim needs this test."""
    batched = store.query_many(unit_vectors[:5], top_k=3)
    singles = [store.query(v, top_k=3) for v in unit_vectors[:5]]
    assert [[m.vector_id for m in row] for row in batched] == [
        [m.vector_id for m in row] for row in singles
    ]


def test_a_batch_query_agrees_on_scores_as_well(store, unit_vectors):
    batched = store.query_many(unit_vectors[:5], top_k=3)
    singles = [store.query(v, top_k=3) for v in unit_vectors[:5]]
    for row_b, row_s in zip(batched, singles, strict=True):
        for mb, ms in zip(row_b, row_s, strict=True):
            assert mb.score == pytest.approx(ms.score, abs=1e-6)


def test_a_batch_query_respects_deletions(store, unit_vectors):
    store.delete(["v0"])
    assert all("v0" not in [m.vector_id for m in row]
               for row in store.query_many(unit_vectors[:4], top_k=5))


def test_a_batch_query_respects_metadata_filters(store, unit_vectors):
    rows = store.query_many(unit_vectors[:3], top_k=5, metadata_filter={"group": "a"})
    assert all(m.metadata["group"] == "a" for row in rows for m in row)


def test_a_batch_query_on_an_empty_namespace_returns_one_empty_row_per_input(store, unit_vectors):
    rows = store.query_many(unit_vectors[:4], namespace="nope")
    assert rows == [[], [], [], []]


def test_the_store_describes_itself_as_exact(store):
    described = store.describe()
    assert described["exact"] is True and described["index"] == "IndexFlatIP"


def test_the_store_reports_physical_and_live_row_counts_separately(store, unit_vectors):
    store.delete(["v0"])
    described = store.describe()
    assert described["rows_physical"]["default"] > described["rows_live"]["default"]


def test_a_match_carries_a_copy_of_its_metadata_not_a_reference(store, unit_vectors):
    match = store.query(unit_vectors[0], top_k=1)[0]
    match.metadata["group"] = "mutated"
    assert store.query(unit_vectors[0], top_k=1)[0].metadata["group"] != "mutated"


def test_a_match_is_immutable():
    with pytest.raises((AttributeError, TypeError)):
        Match(vector_id="a", score=1.0).vector_id = "b"
