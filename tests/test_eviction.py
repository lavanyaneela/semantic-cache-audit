"""Properties of the eviction policies, as pure functions over access state."""

from __future__ import annotations

import pytest

from semantic_cache_audit.cache.eviction import POLICIES, EvictionState


def build(policy: str, capacity: int | None = 2) -> EvictionState:
    state = EvictionState(policy=policy, capacity=capacity)
    state.record_insert("old", 0.0)
    state.record_insert("middle", 10.0)
    state.record_insert("new", 20.0)
    return state


def test_an_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown eviction policy"):
        EvictionState(policy="random")


def test_a_capacity_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        EvictionState(policy="lru", capacity=0)


def test_an_unbounded_cache_never_names_a_victim():
    assert build("lru", capacity=None).victims() == []


def test_a_cache_inside_its_capacity_never_names_a_victim():
    assert build("lru", capacity=10).victims() == []


def test_a_cache_over_capacity_names_exactly_the_overflow():
    assert len(build("lru", capacity=1).victims()) == 2


def test_least_recently_used_evicts_the_oldest_touch():
    assert build("lru").victims() == ["old"]


def test_a_hit_protects_an_entry_from_a_recency_policy():
    state = build("lru")
    state.record_hit("old", 99.0)
    assert state.victims() == ["middle"]


def test_least_frequently_used_evicts_the_least_hit():
    state = build("lfu")
    state.record_hit("old", 1.0)
    state.record_hit("middle", 2.0)
    assert state.victims() == ["new"]


def test_least_frequently_used_breaks_a_tie_by_insertion_order():
    assert build("lfu").victims() == ["old"]


def test_first_in_first_out_evicts_by_insertion_regardless_of_hits():
    state = build("fifo")
    state.record_hit("old", 99.0)
    assert state.victims() == ["old"]


def test_a_ttl_only_policy_never_evicts_on_capacity():
    assert build("ttl_only", capacity=1).victims() == []


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_is_deterministic(policy):
    assert build(policy).victims() == build(policy).victims()


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_names_only_keys_it_holds(policy):
    state = build(policy, capacity=1)
    assert set(state.victims()) <= set(state.entries)


def test_forgetting_a_key_removes_it_from_the_state():
    state = build("lru")
    state.forget("old")
    assert "old" not in state.entries


def test_forgetting_a_key_that_is_not_held_is_harmless():
    build("lru").forget("never-there")


def test_recording_a_hit_on_an_unknown_key_is_harmless():
    build("lru").record_hit("never-there", 1.0)


def test_recording_a_hit_increments_the_count():
    state = build("lfu")
    state.record_hit("old", 1.0)
    state.record_hit("old", 2.0)
    assert state.entries["old"].hits == 2
