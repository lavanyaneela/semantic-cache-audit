"""Properties of the seeded request stream used by E3."""

from __future__ import annotations

import pytest

from semantic_cache_audit.audit.stream import (
    answer_key_of,
    build_stream,
    change_schedule,
    replay,
    version_at,
)


@pytest.fixture(scope="module")
def stream(pairs):
    return build_stream(pairs, n_requests=600, seed=11)


def test_the_stream_has_the_requested_length(stream):
    assert len(stream) == 600


def test_the_stream_is_reproducible_from_its_seed(pairs):
    assert build_stream(pairs, n_requests=200, seed=5) == build_stream(
        pairs, n_requests=200, seed=5
    )


def test_a_different_seed_gives_a_different_stream(pairs):
    assert build_stream(pairs, n_requests=200, seed=5) != build_stream(
        pairs, n_requests=200, seed=6
    )


def test_arrival_times_increase(stream):
    times = [r.at for r in stream]
    assert times == sorted(times)


def test_the_stream_is_skewed_rather_than_uniform(stream):
    from collections import Counter

    counts = Counter(r.query for r in stream)
    top = counts.most_common(1)[0][1]
    assert top > 3 * (len(stream) / len(counts))


def test_both_sides_of_a_paraphrase_need_the_same_answer(pairs):
    positive = next(p for p in pairs if p.same_answer)
    assert answer_key_of(positive, "a") == answer_key_of(positive, "b")


def test_the_sides_of_a_near_miss_need_different_answers(pairs):
    near = next(p for p in pairs if p.is_near_miss)
    assert answer_key_of(near, "a") != answer_key_of(near, "b")


def test_an_answer_starts_at_version_one():
    assert version_at({"k": [5.0]}, "k", 1.0) == 1


def test_an_answer_advances_a_version_at_each_change():
    assert version_at({"k": [5.0, 9.0]}, "k", 10.0) == 3


def test_an_answer_with_no_scheduled_change_stays_at_version_one():
    assert version_at({}, "k", 1e9) == 1


def test_the_change_schedule_is_reproducible_from_its_seed(pairs):
    assert change_schedule(pairs, horizon=100.0, seed=3) == change_schedule(
        pairs, horizon=100.0, seed=3
    )


def test_every_change_falls_inside_the_horizon(pairs):
    schedule = change_schedule(pairs, horizon=100.0)
    assert all(0.0 <= t <= 100.0 for times in schedule.values() for t in times)


def test_a_replay_accounts_for_every_request(pairs, shaped_fixture, stream):
    texts, matrix = shaped_fixture
    schedule = change_schedule(pairs, horizon=stream[-1].at)
    result = replay(stream, texts, matrix, threshold=0.5, policy="lru", capacity=50,
                    ttl_seconds=3600.0, schedule=schedule)
    assert result.hits + result.misses == result.requests == len(stream)


def test_a_replay_is_reproducible(pairs, shaped_fixture, stream):
    texts, matrix = shaped_fixture
    schedule = change_schedule(pairs, horizon=stream[-1].at)
    kwargs = {"threshold": 0.5, "policy": "lru", "capacity": 50, "ttl_seconds": 3600.0,
              "schedule": schedule}
    assert replay(stream, texts, matrix, **kwargs).to_dict() == replay(
        stream, texts, matrix, **kwargs
    ).to_dict()


def test_a_wrong_hit_and_a_stale_hit_are_counted_separately(pairs, shaped_fixture, stream):
    texts, matrix = shaped_fixture
    schedule = change_schedule(pairs, horizon=stream[-1].at)
    result = replay(stream, texts, matrix, threshold=0.5, policy="lru", capacity=50,
                    ttl_seconds=3600.0, schedule=schedule)
    assert result.false_hits + result.stale_hits <= result.hits


def test_a_shorter_ttl_never_increases_staleness(pairs, shaped_fixture, stream):
    texts, matrix = shaped_fixture
    schedule = change_schedule(pairs, horizon=stream[-1].at)
    short = replay(stream, texts, matrix, threshold=0.5, policy="lru", capacity=None,
                   ttl_seconds=200.0, schedule=schedule)
    long = replay(stream, texts, matrix, threshold=0.5, policy="lru", capacity=None,
                  ttl_seconds=None, schedule=schedule)
    assert short.stale_hit_rate <= long.stale_hit_rate


def test_a_smaller_capacity_never_increases_the_hit_rate(pairs, shaped_fixture, stream):
    texts, matrix = shaped_fixture
    schedule = change_schedule(pairs, horizon=stream[-1].at)
    small = replay(stream, texts, matrix, threshold=0.5, policy="lru", capacity=10,
                   ttl_seconds=None, schedule=schedule)
    big = replay(stream, texts, matrix, threshold=0.5, policy="lru", capacity=None,
                 ttl_seconds=None, schedule=schedule)
    assert small.hit_rate <= big.hit_rate


def test_a_replay_breaks_its_counts_down_by_class(pairs, shaped_fixture, stream):
    texts, matrix = shaped_fixture
    schedule = change_schedule(pairs, horizon=stream[-1].at)
    result = replay(stream, texts, matrix, threshold=0.5, policy="lru", capacity=50,
                    ttl_seconds=3600.0, schedule=schedule)
    assert sum(b["requests"] for b in result.by_class.values()) == result.requests


def test_an_empty_result_does_not_divide_by_zero():
    from semantic_cache_audit.audit.stream import StreamResult

    empty = StreamResult()
    assert (empty.hit_rate, empty.false_hit_rate, empty.stale_hit_rate) == (0.0, 0.0, 0.0)


def test_the_caller_experienced_rate_combines_wrong_and_stale():
    from semantic_cache_audit.audit.stream import StreamResult

    result = StreamResult(requests=10, hits=10, false_hits=2, stale_hits=3)
    assert result.wrong_or_stale_rate == pytest.approx(0.5)
