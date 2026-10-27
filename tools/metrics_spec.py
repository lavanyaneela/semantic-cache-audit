"""The single declaration of every published figure: where it comes from and how it reads.

One file so that the collector and the checker cannot drift apart about what a metric
means. Each entry names the experiment file it is read from, the path into that file's
JSON, how the value is rendered into prose, and the anchor phrases a document must use
when it quotes it.

Anchor rules, each learned by getting it wrong on an earlier build:

* An anchor names the number and two or three words, never a whole clause. A longer
  phrase straddles a line break in wrapped markdown and then matches nothing while the
  prose it guards is perfectly correct.
* Anchors are alternatives. Any one matching is enough, because a figure reads
  differently in a table cell and in a sentence, and requiring all of them fails the
  build for correct prose.
* Small numbers are checked only through anchors. A bare 3 or 6 appears by coincidence
  in a version string or a heading, so matching its digits proves nothing.
* No two metrics may share an anchor, including by containment. `"{} pairs"` and
  `"{} pairs required"` would let one metric satisfy the other's guard.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def as_int(value: Any) -> str:
    return str(int(round(float(value))))


def as_pct1(value: Any) -> str:
    return f"{float(value) * 100:.1f}"


def as_pct2(value: Any) -> str:
    return f"{float(value) * 100:.2f}"


def as_points1(value: Any) -> str:
    """A value already expressed in percentage points."""
    return f"{float(value):.1f}"


def as_points2(value: Any) -> str:
    return f"{float(value):.2f}"


def as_ratio1(value: Any) -> str:
    return f"{float(value):.1f}"


def as_auc(value: Any) -> str:
    return f"{float(value):.3f}"


def as_text(value: Any) -> str:
    return str(value)


@dataclass(frozen=True)
class Metric:
    """One published figure."""

    name: str
    source: str
    path: tuple[str, ...]
    render: Callable[[Any], str]
    anchors: tuple[str, ...] = field(default_factory=tuple)
    #: Set for values that are words rather than numbers. They are checked by
    #: presence of the anchor phrase alone, since there is no digit to compare.
    textual: bool = False
    note: str = ""


#: `source` is a file stem under docs/experiments, or one of the pseudo sources
#: "corpus" (the fixture manifest) and "suite" (the test run).
METRICS: tuple[Metric, ...] = (
    # ---------------------------------------------------------------- the corpus
    Metric("corpus_pairs_total", "corpus", ("corpus_pairs_total",), as_int,
           ("{} labelled pairs", "{} pairs in")),
    Metric("near_miss_pairs", "corpus", ("near_miss_pairs",), as_int,
           ("{} near miss pairs",)),
    Metric("positive_controls", "corpus", ("positive_control_pairs",), as_int,
           ("{} positive controls", "{} true paraphrases", "{} of them share")),
    Metric("templates_total", "corpus", ("templates_total",), as_int,
           ("{} template families",)),
    Metric("corpus_domains", "corpus", ("__domains__",), as_int,
           ("{} domains",)),
    Metric("near_miss_classes", "corpus", ("__classes__",), as_int,
           ("{} near miss classes",)),
    # ------------------------------------------------- E1, the headline encoder
    Metric("separation_auc", "e1_threshold_curve", ("headline_separation_auc",), as_auc,
           ("separation of {}", "separation is {}", "scores {} where")),
    Metric("wrong_prefix_hits", "e1_threshold_curve", ("headline_wrong_prefix_hits",), as_int,
           ("{} most confident", "first {} matches", "{} consecutive wrong")),
    Metric("clean_ceiling_hits", "e1_threshold_curve", ("headline_clean_ceiling_hits",), as_int,
           ("{} of the 120 true", "exactly {} paraphrases", "serves {} paraphrases")),
    # ---------------------------------------------------- E2, where it breaks
    Metric("worst_class_name", "e2_class_breakdown", ("worst_class_name",), as_text,
           ("worst class is {}", "worst is {}"), textual=True),
    Metric("worst_class_false_hit_pct", "e2_class_breakdown", ("worst_class_false_hit_pct",),
           as_pct1, ("{}% of that class", "{}% on that class")),
    Metric("safest_class_name", "e2_class_breakdown", ("safest_class_name",), as_text,
           ("only {} stays", "safest is {}"), textual=True),
    Metric("safest_class_false_hit_pct", "e2_class_breakdown", ("safest_class_false_hit_pct",),
           as_pct1, ("{}% on the safest",)),
    Metric("classes_below_chance", "e2_class_breakdown", ("classes_below_chance_separation",),
           as_int, ("{} of the five classes", "{} classes separate below")),
    Metric("probes_per_class", "e2_class_breakdown", ("probes_per_class",), as_int,
           ("{} probes per class",)),
    Metric("negation_false_hit_pct", "e2_class_breakdown", ("rate_by_class", "negation"),
           as_pct1, ("{}% of negation",)),
    Metric("numeric_change_false_hit_pct", "e2_class_breakdown",
           ("rate_by_class", "numeric_change"), as_pct1, ("{}% of numeric",)),
    Metric("scope_change_false_hit_pct", "e2_class_breakdown",
           ("rate_by_class", "scope_change"), as_pct1, ("{}% of scope",)),
    Metric("safest_class_upper_pct", "e2_class_breakdown", ("safest_class_upper",), as_pct1,
           ("bounded at {}%",)),
    Metric("negation_upper_pct", "e2_class_breakdown", ("upper_by_class", "negation"),
           as_pct1, ("{}% on negation",)),
    Metric("numeric_change_upper_pct", "e2_class_breakdown",
           ("upper_by_class", "numeric_change"), as_pct1, ("{}% on numeric",)),
    Metric("scope_change_upper_pct", "e2_class_breakdown", ("upper_by_class", "scope_change"),
           as_pct1, ("{}% on scope",)),
    Metric("worst_class_min_similarity", "e2_class_breakdown", ("worst_class_min_similarity",),
           as_auc, ("above {} on this", "never below {}")),
    # ------------------------------------------------------- E3, the two levers
    Metric("policy_false_hit_spread_pp", "e3_eviction_sweep",
           ("false_hit_spread_across_policies_pp",), as_points2,
           ("{} points of wrong", "by {} points of false")),
    Metric("policy_stale_spread_pp", "e3_eviction_sweep",
           ("stale_hit_spread_across_policies_pp",), as_points1,
           ("{} points of staleness",)),
    Metric("bounded_false_hit_pct", "e3_eviction_sweep", ("bounded_false_hit_rate_lru",), as_pct1,
           ("{}% when bounded",)),
    Metric("unbounded_false_hit_pct", "e3_eviction_sweep", ("unbounded_false_hit_rate",), as_pct1,
           ("{}% when unbounded",)),
    Metric("unbounded_false_hit_ratio", "e3_eviction_sweep",
           ("false_hit_ratio_unbounded_over_bounded",), as_ratio1,
           ("{} times the wrong",)),
    Metric("lowest_stale_policy", "e3_eviction_sweep", ("lowest_stale_policy",), as_text,
           ("{} cuts staleness",), textual=True),
    Metric("stale_reduction_ratio", "e3_eviction_sweep",
           ("stale_reduction_ratio_fifo_over_lru",), as_ratio1,
           ("{} fold reduction", "{} times less stale")),
    # --------------------------------------------- E4, the budget decision table
    Metric("tightest_budget_pct", "e4_budget_decisions", ("tightest_certifiable_budget",),
           as_pct2, ("tighter than {}%", "budget of {}%")),
    Metric("hits_required_one_pct", "e4_budget_decisions", ("hits_required_at_one_percent",),
           as_int, ("{} clean hits",)),
    Metric("pairs_required_one_pct", "e4_budget_decisions",
           ("corpus_pairs_required_at_one_percent",), as_int, ("{} pairs would be",)),
    Metric("real_embedders_ever_ok", "e4_budget_decisions", ("real_embedders_ever_ok",), as_int,
           ("{} of the four real",)),
    # ---------------------------------------------------- E5, threshold transfer
    #
    # Real embedders only. The sweep also covers the separation family, which
    # inflates every count roughly thirteen fold, and quoting a pooled figure as a
    # result about real models is the misreading ADR-005 forbids. The instrument
    # count is published beside them so the split is visible rather than implied.
    Metric("transfers_real_only", "e5_threshold_transfer", ("transfers_real_only",), as_int,
           ("{} transfers between",)),
    Metric("real_threshold_shift_points", "e5_threshold_transfer",
           ("real_threshold_shift_median_points",), as_points1,
           ("shifts by {} cosine", "median shift of {}")),
    Metric("real_transfers_saturated", "e5_threshold_transfer",
           ("real_transfers_that_saturate_the_cache",), as_int, ("{} stop rejecting",)),
    Metric("real_transfers_all_wrong", "e5_threshold_transfer",
           ("real_transfers_where_every_hit_is_wrong",), as_int,
           ("{} of them every hit",)),
    Metric("instrument_transfers", "e5_threshold_transfer",
           ("transfers_involving_instrument",), as_int, ("{} further transfers",)),
    # --------------------------------------------------------------- the suite
    Metric("tests_total", "suite", ("tests_total",), as_int, ("{} tests",)),
    Metric("coverage_line_pct", "suite", ("coverage_line_pct",), as_points1,
           ("{}% line coverage",)),
)


def by_name() -> dict[str, Metric]:
    return {m.name: m for m in METRICS}
