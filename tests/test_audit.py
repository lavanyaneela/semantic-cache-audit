"""Properties of the sweep, and the agreement between its fast path and the real cache."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_cache_audit.audit.probe import pair_scores, probe_at_threshold, probe_through_cache
from semantic_cache_audit.audit.sweep import (
    DEFAULT_NEAR_MISS_PREVALENCE,
    run_sweep,
    summarise,
    threshold_grid,
)
from semantic_cache_audit.corpus.schema import NEAR_MISS_CLASSES
from semantic_cache_audit.embed.registry import PRETRAINED_NAME


@pytest.fixture(scope="module")
def headline(pairs):
    from semantic_cache_audit.embed.fixtures import load_fixture

    texts, matrix = load_fixture(PRETRAINED_NAME)
    return run_sweep(pairs, texts, matrix, PRETRAINED_NAME)


def test_pair_scores_returns_one_score_per_pair(pairs, charngram_fixture):
    texts, matrix = charngram_fixture
    assert pair_scores(pairs, texts, matrix).shape == (len(pairs),)


def test_pair_scores_are_cosines_within_range(pairs, charngram_fixture):
    texts, matrix = charngram_fixture
    scores = pair_scores(pairs, texts, matrix)
    assert scores.min() >= -1.0001 and scores.max() <= 1.0001


def test_a_pair_missing_from_the_fixture_is_reported_rather_than_skipped(pairs, charngram_fixture):
    texts, matrix = charngram_fixture
    trimmed = [t for t in texts if t != pairs[0].query_a]
    with pytest.raises(KeyError, match="not in the embedding fixture"):
        pair_scores(pairs, trimmed, matrix[: len(trimmed)])


def test_the_threshold_grid_spans_the_observed_score_range():
    scores = np.array([0.2, 0.9])
    grid = threshold_grid(scores, steps=11)
    assert grid[0] < 0.2 and grid[-1] > 0.9


def test_the_threshold_grid_has_the_requested_number_of_steps():
    assert len(threshold_grid(np.array([0.0, 1.0]), steps=51)) == 51


def test_a_degenerate_score_range_gives_a_single_grid_point():
    assert len(threshold_grid(np.array([0.5, 0.5]))) == 1


def test_the_lowest_threshold_in_the_grid_makes_everything_hit(headline):
    assert headline.points[0].hit_rate == pytest.approx(1.0)


def test_the_highest_threshold_in_the_grid_makes_nothing_hit(headline):
    assert headline.points[-1].hits == 0


def test_the_hit_rate_falls_monotonically_as_the_threshold_rises(headline):
    rates = [p.hit_rate for p in headline.points]
    assert all(a >= b for a, b in zip(rates, rates[1:], strict=False))


def test_every_point_accounts_for_every_probe(headline, pairs):
    assert all(p.probes == len(pairs) for p in headline.points)


def test_false_hits_never_exceed_hits(headline):
    assert all(p.false_hits <= p.hits for p in headline.points)


def test_the_upper_bound_is_never_below_the_measured_rate(headline):
    assert all(p.false_hit_rate_upper >= p.false_hit_rate - 1e-9 for p in headline.points)


def test_every_point_breaks_down_by_class(headline):
    assert all(set(p.by_class) >= set(NEAR_MISS_CLASSES) for p in headline.points)


def test_class_hit_counts_sum_to_the_total(headline):
    for point in headline.points:
        assert sum(s["hits"] for s in point.by_class.values()) == point.hits


def test_class_false_hit_counts_sum_to_the_total(headline):
    for point in headline.points:
        assert sum(s["false_hits"] for s in point.by_class.values()) == point.false_hits


def test_a_positive_control_class_never_produces_a_false_hit(headline):
    for point in headline.points:
        assert point.by_class["true_paraphrase"]["false_hits"] == 0


def test_a_missed_opportunity_is_a_paraphrase_that_did_not_hit(headline, pairs):
    point = headline.points[-1]
    assert point.missed_opportunities == sum(1 for p in pairs if p.same_answer)


def test_the_prevalence_weighted_rate_differs_from_the_raw_one(headline):
    point = headline.points[0]
    assert point.weighted_false_hit_rate != pytest.approx(point.false_hit_rate)


def test_the_prevalence_weighted_rate_tracks_the_declared_prevalence(pairs, charngram_fixture):
    texts, matrix = charngram_fixture
    low = run_sweep(pairs, texts, matrix, "charngram", steps=5, prevalence=0.01)
    high = run_sweep(pairs, texts, matrix, "charngram", steps=5, prevalence=0.50)
    assert low.points[0].weighted_false_hit_rate < high.points[0].weighted_false_hit_rate


def test_the_default_prevalence_is_declared_in_one_place():
    assert 0.0 < DEFAULT_NEAR_MISS_PREVALENCE < 1.0


def test_the_resolution_floor_at_a_point_is_one_over_its_hits(headline):
    point = headline.points[0]
    assert point.resolution_floor == pytest.approx(1.0 / point.hits)


def test_the_fast_probe_path_agrees_with_the_real_cache(pairs, charngram_fixture):
    """The sweep bypasses the cache for speed, so the bypass needs a referee."""
    texts, matrix = charngram_fixture
    scores = pair_scores(pairs, texts, matrix)
    subset = pairs[:60]
    fast = probe_at_threshold(subset, scores[:60], 0.5)
    slow = probe_through_cache(subset, texts, matrix, 0.5)
    assert [p.hit for p in fast] == [p.hit for p in slow]


def test_the_fast_probe_path_agrees_with_the_real_cache_on_scores(pairs, charngram_fixture):
    texts, matrix = charngram_fixture
    scores = pair_scores(pairs, texts, matrix)
    subset = pairs[:40]
    fast = probe_at_threshold(subset, scores[:40], 0.5)
    slow = probe_through_cache(subset, texts, matrix, 0.5)
    for f, s in zip(fast, slow, strict=True):
        if s.hit:
            assert f.score == pytest.approx(s.score, abs=1e-5)


def test_a_probe_knows_whether_it_was_a_false_hit(pairs, charngram_fixture):
    texts, matrix = charngram_fixture
    probes = probe_at_threshold(pairs, pair_scores(pairs, texts, matrix), -1.0)
    assert sum(1 for p in probes if p.false_hit) == sum(1 for p in pairs if p.is_near_miss)


def test_the_wrong_prefix_counts_leading_wrong_hits(headline):
    assert headline.wrong_prefix_hits > 0


def test_the_wrong_prefix_is_zero_when_a_paraphrase_ranks_first(pairs, oracle_fixture):
    texts, matrix = oracle_fixture
    assert run_sweep(pairs, texts, matrix, "synthetic@0").wrong_prefix_hits == 0


def test_the_clean_ceiling_is_zero_when_a_near_miss_outscores_every_paraphrase(headline):
    assert headline.clean_ceiling_hits == 0


def test_the_clean_ceiling_is_every_paraphrase_for_a_perfectly_separating_family(
    pairs, oracle_fixture
):
    texts, matrix = oracle_fixture
    sweep = run_sweep(pairs, texts, matrix, "synthetic@0")
    assert sweep.clean_ceiling_hits == sum(1 for p in pairs if p.same_answer)


def test_the_clean_ceiling_rate_is_a_share_of_all_probes(pairs, oracle_fixture):
    texts, matrix = oracle_fixture
    sweep = run_sweep(pairs, texts, matrix, "synthetic@0")
    assert sweep.clean_ceiling_hit_rate == pytest.approx(sweep.clean_ceiling_hits / len(pairs))


def test_separation_below_half_means_the_embedder_is_inverted(headline):
    assert headline.separation_auc < 0.5


def test_a_sweep_serialises_without_losing_its_points(headline):
    assert len(headline.to_dict()["points"]) == len(headline.points)


def test_the_worst_class_is_a_near_miss_class(headline):
    name, _ = headline.worst_class(headline.points[0])
    assert name in NEAR_MISS_CLASSES


def test_the_worst_class_of_a_point_with_no_hits_is_reported_as_none(headline):
    assert headline.worst_class(headline.points[-1])[0] == "none"


def test_summarising_an_empty_probe_list_does_not_divide_by_zero(pairs):
    point = summarise([], 0.5, pairs)
    assert point.hit_rate == 0.0 and point.false_hit_rate == 0.0
