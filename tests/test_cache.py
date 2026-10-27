"""Properties of the cache, including the absence the whole repository is about."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from semantic_cache_audit.cache.backends import HotLayer, PayloadStore
from semantic_cache_audit.cache.clock import ManualClock
from semantic_cache_audit.cache.eviction import EvictionState
from semantic_cache_audit.cache.semantic_cache import Lookup, SemanticCache
from semantic_cache_audit.index.faiss_store import FaissFlatStore


@pytest.fixture
def cache(unit_vectors):
    return SemanticCache(
        FaissFlatStore(unit_vectors.shape[1]),
        threshold=0.5,
        hot=HotLayer(),
        payloads=PayloadStore(),
        clock=ManualClock(),
        eviction=EvictionState(policy="lru"),
    )


def test_a_lookup_against_an_empty_cache_misses(cache, unit_vectors):
    assert cache.lookup("anything", unit_vectors[0]).hit is False


def test_storing_then_asking_the_same_query_is_an_exact_hit(cache, unit_vectors):
    cache.put("k", "q", unit_vectors[0], "answer-1")
    result = cache.lookup("q", unit_vectors[0])
    assert result.hit and result.exact and result.answer_id == "answer-1"


def test_an_exact_hit_scores_one(cache, unit_vectors):
    cache.put("k", "q", unit_vectors[0], "answer-1")
    assert cache.lookup("q", unit_vectors[0]).score == 1.0


def test_a_different_query_above_the_threshold_is_an_approximate_hit(cache, unit_vectors):
    cache.put("k", "stored", unit_vectors[0], "answer-1")
    result = cache.lookup("asked", unit_vectors[0])
    assert result.hit and not result.exact


def test_an_approximate_hit_names_the_query_whose_answer_it_returned(cache, unit_vectors):
    cache.put("k", "stored", unit_vectors[0], "answer-1")
    assert cache.lookup("asked", unit_vectors[0]).matched_query == "stored"


def test_a_query_below_the_threshold_misses(cache, unit_vectors):
    cache.put("k", "stored", unit_vectors[0], "answer-1")
    far = -unit_vectors[0]
    assert cache.lookup("asked", far).hit is False


def test_a_miss_below_the_threshold_still_reports_the_score_it_saw(cache, unit_vectors):
    cache.put("k", "stored", unit_vectors[0], "answer-1")
    assert cache.lookup("asked", -unit_vectors[0]).score is not None


def test_a_score_exactly_at_the_threshold_is_a_hit(unit_vectors):
    cache = SemanticCache(FaissFlatStore(unit_vectors.shape[1]), threshold=1.0,
                          hot=HotLayer(), payloads=PayloadStore(), clock=ManualClock())
    cache.put("k", "stored", unit_vectors[0], "answer-1")
    assert cache.lookup("asked", unit_vectors[0]).hit is True


def test_a_lookup_result_carries_no_indication_of_whether_the_answer_is_right():
    """The absence that makes this repository necessary, pinned so it cannot regress.

    If a field like `correct` ever appeared here, the cache would be able to answer
    the question the audit exists to answer, and the premise would be gone.
    """
    fields = {f.name for f in dataclasses.fields(Lookup)}
    assert not fields & {"correct", "is_correct", "wrong", "false_hit", "valid"}


def test_the_lookup_result_is_immutable(cache, unit_vectors):
    cache.put("k", "q", unit_vectors[0], "a")
    with pytest.raises((AttributeError, TypeError)):
        cache.lookup("q", unit_vectors[0]).hit = False


def test_an_expired_entry_is_not_returned(unit_vectors):
    clock = ManualClock()
    cache = SemanticCache(FaissFlatStore(unit_vectors.shape[1]), threshold=0.5,
                          hot=HotLayer(), payloads=PayloadStore(), clock=clock,
                          ttl_seconds=10.0)
    cache.put("k", "q", unit_vectors[0], "a")
    clock.advance(11.0)
    assert cache.lookup("q", unit_vectors[0]).hit is False


def test_an_entry_inside_its_ttl_is_still_returned(unit_vectors):
    clock = ManualClock()
    cache = SemanticCache(FaissFlatStore(unit_vectors.shape[1]), threshold=0.5,
                          hot=HotLayer(), payloads=PayloadStore(), clock=clock,
                          ttl_seconds=10.0)
    cache.put("k", "q", unit_vectors[0], "a")
    clock.advance(9.0)
    assert cache.lookup("q", unit_vectors[0]).hit is True


def test_an_expired_entry_is_removed_rather_than_left_to_be_found_again(unit_vectors):
    clock = ManualClock()
    cache = SemanticCache(FaissFlatStore(unit_vectors.shape[1]), threshold=0.5,
                          hot=HotLayer(), payloads=PayloadStore(), clock=clock,
                          ttl_seconds=10.0)
    cache.put("k", "q", unit_vectors[0], "a")
    clock.advance(11.0)
    cache.lookup("q", unit_vectors[0])
    assert cache.store.count(cache.namespace) == 0


def test_an_entry_with_no_ttl_never_expires(cache, unit_vectors):
    cache.put("k", "q", unit_vectors[0], "a")
    cache.clock.advance(10_000_000.0)
    assert cache.lookup("q", unit_vectors[0]).hit is True


def test_evicting_removes_the_entry_from_every_layer(cache, unit_vectors):
    cache.put("k", "q", unit_vectors[0], "a")
    cache.evict("k")
    assert cache.store.count(cache.namespace) == 0
    assert cache.payloads.get("k") is None
    assert cache.hot.exact_lookup("q") is None


def test_capacity_is_enforced_as_entries_are_added(unit_vectors):
    cache = SemanticCache(FaissFlatStore(unit_vectors.shape[1]), threshold=0.99,
                          hot=HotLayer(), payloads=PayloadStore(), clock=ManualClock(),
                          eviction=EvictionState(policy="lru", capacity=3))
    for i in range(6):
        cache.clock.advance(1.0)
        cache.put(f"k{i}", f"q{i}", unit_vectors[i], f"a{i}")
    assert cache.store.count(cache.namespace) == 3


def test_eviction_is_counted(unit_vectors):
    cache = SemanticCache(FaissFlatStore(unit_vectors.shape[1]), threshold=0.99,
                          hot=HotLayer(), payloads=PayloadStore(), clock=ManualClock(),
                          eviction=EvictionState(policy="lru", capacity=2))
    for i in range(5):
        cache.clock.advance(1.0)
        cache.put(f"k{i}", f"q{i}", unit_vectors[i], f"a{i}")
    assert cache.eviction.evicted_total == 3


def test_a_vector_whose_payload_vanished_is_a_miss_rather_than_a_crash(cache, unit_vectors):
    cache.put("k", "q", unit_vectors[0], "a")
    cache.payloads.drop("k")
    cache.hot.drop("k")
    assert cache.lookup("other", unit_vectors[0]).hit is False


def test_the_cache_counts_exact_and_approximate_hits_separately(cache, unit_vectors):
    cache.put("k", "q", unit_vectors[0], "a")
    cache.lookup("q", unit_vectors[0])
    cache.lookup("other", unit_vectors[0])
    assert cache.stats["exact_hits"] == 1 and cache.stats["approx_hits"] == 1


def test_the_cache_counts_every_lookup(cache, unit_vectors):
    for _ in range(4):
        cache.lookup("q", unit_vectors[0])
    assert cache.stats["lookups"] == 4


def test_the_cache_describes_its_own_configuration(cache):
    described = cache.describe()
    assert described["threshold"] == 0.5 and described["eviction_policy"] == "lru"


def test_a_hit_refreshes_the_entry_for_a_recency_policy(cache, unit_vectors):
    cache.put("k", "q", unit_vectors[0], "a")
    cache.clock.advance(50.0)
    cache.lookup("q", unit_vectors[0])
    assert cache.eviction.entries["k"].last_seen_at == 50.0


def test_the_manual_clock_refuses_to_run_backwards():
    with pytest.raises(ValueError, match="backwards"):
        ManualClock().advance(-1.0)


def test_the_hot_layer_treats_an_unknown_key_as_expired():
    assert HotLayer().is_expired("never-written", 0.0) is True


def test_the_hot_layer_can_be_flushed():
    hot = HotLayer()
    hot.put("k", "q", 0.0, None)
    hot.flush()
    assert hot.exact_lookup("q") is None


def test_the_payload_store_counts_what_it_holds():
    store = PayloadStore()
    store.put("k", "q", "a", 1, 0.0)
    assert store.count() == 1


def test_the_payload_store_replaces_rather_than_duplicates():
    store = PayloadStore()
    store.put("k", "q", "a", 1, 0.0)
    store.put("k", "q", "b", 2, 1.0)
    assert store.count() == 1 and store.get("k")["answer_id"] == "b"


def test_the_payload_store_can_be_flushed():
    store = PayloadStore()
    store.put("k", "q", "a", 1, 0.0)
    store.flush()
    assert store.count() == 0


def test_the_payload_records_which_query_the_answer_was_stored_against():
    store = PayloadStore()
    store.put("k", "the stored question", "a", 1, 0.0)
    assert store.get("k")["query"] == "the stored question"


def test_the_cache_puts_an_entry_into_the_namespace_it_was_configured_with(unit_vectors):
    cache = SemanticCache(FaissFlatStore(unit_vectors.shape[1]), threshold=0.5,
                          hot=HotLayer(), payloads=PayloadStore(), clock=ManualClock(),
                          namespace="tenant-7")
    cache.put("k", "q", unit_vectors[0], "a")
    assert cache.store.count("tenant-7") == 1


def test_vectors_are_accepted_as_one_dimensional_arrays(cache, unit_vectors):
    cache.put("k", "q", np.asarray(unit_vectors[0]), "a")
    assert cache.store.count(cache.namespace) == 1
