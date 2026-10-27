"""Properties of the arithmetic that keeps a measured zero from reading as zero."""

from __future__ import annotations

import math

import pytest

from semantic_cache_audit.audit.statistics import (
    CONFIDENCE,
    resolution_floor,
    separation_auc,
    trials_needed_for,
    upper_bound,
)


def test_an_upper_bound_over_no_trials_is_total_uncertainty():
    assert upper_bound(0, 0) == 1.0


def test_a_clean_run_never_bounds_the_rate_at_zero():
    assert upper_bound(0, 100) > 0.0


def test_a_clean_run_of_n_trials_bounds_the_rate_near_three_over_n():
    assert upper_bound(0, 100) == pytest.approx(0.03, abs=0.005)


def test_the_bound_tightens_as_the_sample_grows():
    assert upper_bound(0, 30) > upper_bound(0, 300) > upper_bound(0, 3000)


def test_the_bound_is_never_below_the_observed_rate():
    assert upper_bound(5, 100) >= 0.05


def test_a_run_where_everything_failed_bounds_at_one():
    assert upper_bound(10, 10) == 1.0


def test_more_failures_give_a_looser_bound():
    assert upper_bound(1, 100) < upper_bound(10, 100)


def test_more_successes_than_trials_is_rejected():
    with pytest.raises(ValueError, match="not a proportion"):
        upper_bound(11, 10)


def test_a_negative_count_is_rejected():
    with pytest.raises(ValueError, match="not a proportion"):
        upper_bound(-1, 10)


def test_the_resolution_floor_is_one_over_the_sample():
    assert resolution_floor(40) == pytest.approx(0.025)


def test_the_resolution_floor_of_an_empty_sample_is_total_uncertainty():
    assert resolution_floor(0) == 1.0


def test_certifying_a_tighter_budget_needs_more_trials():
    assert trials_needed_for(0.001) > trials_needed_for(0.01) > trials_needed_for(0.1)


def test_the_trials_needed_really_do_push_the_bound_under_the_budget():
    for budget in (0.001, 0.01, 0.05, 0.2):
        assert upper_bound(0, trials_needed_for(budget)) <= budget


def test_one_fewer_trial_than_needed_leaves_the_bound_above_the_budget():
    for budget in (0.01, 0.05):
        assert upper_bound(0, trials_needed_for(budget) - 1) > budget


def test_certifying_one_percent_needs_about_three_hundred_trials():
    assert trials_needed_for(0.01) == 299


def test_a_budget_outside_zero_to_one_is_rejected():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="strictly between"):
            trials_needed_for(bad)


def test_the_confidence_used_everywhere_is_declared_once():
    assert CONFIDENCE == 0.95


def test_separation_of_one_means_every_positive_outscores_every_negative():
    assert separation_auc([0.9, 0.8], [0.1, 0.2]) == pytest.approx(1.0)


def test_separation_of_zero_means_the_ranking_is_inverted():
    assert separation_auc([0.1, 0.2], [0.9, 0.8]) == pytest.approx(0.0)


def test_identical_distributions_separate_at_chance():
    assert separation_auc([0.5] * 10, [0.5] * 10) == pytest.approx(0.5)


def test_separation_without_positives_is_undefined_rather_than_zero():
    assert math.isnan(separation_auc([], [0.1]))


def test_separation_without_negatives_is_undefined_rather_than_one():
    assert math.isnan(separation_auc([0.1], []))


def test_separation_is_invariant_to_a_monotone_rescaling():
    pos, neg = [0.9, 0.7, 0.6], [0.5, 0.2]
    scaled_pos = [p * 10 + 3 for p in pos]
    scaled_neg = [n * 10 + 3 for n in neg]
    assert separation_auc(pos, neg) == pytest.approx(separation_auc(scaled_pos, scaled_neg))
