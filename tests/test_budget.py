"""Properties of the decision, and of the three answers it can give."""

from __future__ import annotations

import pytest

from semantic_cache_audit.audit.budget import Exit, decide
from semantic_cache_audit.audit.statistics import trials_needed_for
from semantic_cache_audit.audit.sweep import run_sweep
from semantic_cache_audit.embed.fixtures import load_fixture
from semantic_cache_audit.embed.registry import PRETRAINED_NAME


@pytest.fixture(scope="module")
def headline(pairs):
    texts, matrix = load_fixture(PRETRAINED_NAME)
    return run_sweep(pairs, texts, matrix, PRETRAINED_NAME)


@pytest.fixture(scope="module")
def oracle(pairs):
    texts, matrix = load_fixture("synthetic@0")
    return run_sweep(pairs, texts, matrix, "synthetic@0")


def test_a_budget_outside_zero_to_one_is_rejected(headline):
    with pytest.raises(ValueError, match="strictly between"):
        decide(headline, budget=0.0)


def test_a_budget_the_corpus_cannot_resolve_is_refused(headline):
    assert decide(headline, budget=0.001).exit_code is Exit.REFUSED


def test_a_refusal_is_a_statement_about_the_corpus_not_the_embedder(headline, oracle):
    assert decide(headline, budget=0.001).exit_code == decide(oracle, budget=0.001).exit_code


def test_a_refusal_names_the_number_of_clean_hits_the_budget_would_need(headline):
    decision = decide(headline, budget=0.01)
    assert decision.hits_required == trials_needed_for(0.01)


def test_a_refusal_names_the_corpus_size_the_budget_would_need(headline, pairs):
    decision = decide(headline, budget=0.01)
    assert decision.corpus_pairs_required > len(pairs)


def test_a_refusal_explains_that_positive_controls_are_the_ceiling(headline):
    assert "positive control" in decide(headline, budget=0.001).reason


def test_a_refusal_reports_the_tightest_budget_the_corpus_can_certify(headline):
    decision = decide(headline, budget=0.001)
    assert 0.0 < decision.tightest_certifiable_budget < 0.1


def test_the_tightest_certifiable_budget_does_not_depend_on_the_budget_asked_for(headline):
    assert (
        decide(headline, budget=0.001).tightest_certifiable_budget
        == decide(headline, budget=0.2).tightest_certifiable_budget
    )


def test_a_budget_just_above_the_tightest_certifiable_one_is_not_refused(headline):
    tightest = decide(headline, budget=0.2).tightest_certifiable_budget
    assert decide(headline, budget=tightest * 1.05).exit_code is not Exit.REFUSED


def test_an_inverted_embedder_finds_no_threshold_at_any_resolvable_budget(headline):
    for budget in (0.025, 0.05, 0.1, 0.2):
        assert decide(headline, budget=budget).exit_code is Exit.NO_THRESHOLD


def test_a_perfectly_separating_family_finds_a_threshold(oracle):
    assert decide(oracle, budget=0.05).exit_code is Exit.OK


def test_a_found_threshold_meets_the_budget_on_the_bound_not_the_point_estimate(oracle):
    decision = decide(oracle, budget=0.05)
    assert decision.operating_point.false_hit_rate_upper <= 0.05


def test_a_found_threshold_serves_at_least_the_minimum_hit_rate(oracle):
    decision = decide(oracle, budget=0.05, min_hit_rate=0.3)
    assert decision.operating_point.hit_rate >= 0.3


def test_an_unreachable_minimum_hit_rate_turns_a_pass_into_a_no(oracle):
    assert decide(oracle, budget=0.05, min_hit_rate=0.99).exit_code is Exit.NO_THRESHOLD


def test_the_chosen_threshold_serves_the_most_traffic_among_those_that_qualify(oracle):
    decision = decide(oracle, budget=0.05)
    qualifying = [
        p for p in oracle.points
        if p.hits >= decision.hits_required
        and p.hit_rate >= decision.min_hit_rate
        and p.false_hit_rate_upper <= decision.budget
    ]
    assert decision.operating_point.hit_rate == max(p.hit_rate for p in qualifying)


def test_a_looser_budget_never_serves_less_traffic(oracle):
    tight = decide(oracle, budget=0.05).operating_point.hit_rate
    loose = decide(oracle, budget=0.20).operating_point.hit_rate
    assert loose >= tight


def test_a_no_threshold_decision_reports_the_closest_it_got(headline):
    decision = decide(headline, budget=0.05)
    assert decision.operating_point is not None


def test_a_no_threshold_decision_names_the_worst_class(headline):
    assert decide(headline, budget=0.05).worst_class


def test_every_decision_carries_the_separation_of_the_embedder(headline):
    assert decide(headline, budget=0.05).separation_auc == pytest.approx(headline.separation_auc)


def test_every_decision_carries_the_clean_ceiling(headline):
    assert decide(headline, budget=0.05).clean_ceiling_hits == headline.clean_ceiling_hits


def test_every_decision_serialises_to_a_dictionary(headline):
    payload = decide(headline, budget=0.05).to_dict()
    assert payload["exit_name"] == "NO_THRESHOLD" and payload["exit_code"] == 1


def test_every_decision_explains_itself_in_a_sentence(headline):
    assert len(decide(headline, budget=0.05).reason) > 40


def test_every_decision_carries_at_least_one_note(headline):
    assert decide(headline, budget=0.05).notes


def test_the_exit_codes_are_the_documented_integers():
    assert (int(Exit.OK), int(Exit.NO_THRESHOLD), int(Exit.REFUSED), int(Exit.ERROR)) == (
        0, 1, 2, 3
    )


def test_each_exit_code_is_documented_in_the_readme(repo_root):
    """The previous version of this asserted `Exit.__doc__ is not None` inside a
    loop over the members, which re-checked the class docstring four times and
    could not fail. The contract that matters is the one a reader sees."""
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    table = readme[readme.index("## Exit codes") :]
    for code in Exit:
        assert f"| {int(code)} |" in table, code.name


def test_the_exit_codes_are_distinct():
    assert len({int(c) for c in Exit}) == len(list(Exit))
