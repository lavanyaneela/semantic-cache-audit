"""Turns the sweep into a decision, and knows when it is not entitled to take one.

Three outcomes rather than two. A threshold meets the budget, no threshold meets it,
or the corpus is too small to tell the difference. The third is the one most tools
lack, and it is the reason exit code 2 exists: a measured zero over thirty hits
certifies nothing, so the tool says what sample it would take instead of approving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .statistics import CONFIDENCE, trials_needed_for, upper_bound
from .sweep import Point, Sweep


class Exit(IntEnum):
    """CLI exit codes, documented in the README and asserted by tests."""

    OK = 0
    """A threshold exists that meets the stated error budget. It is printed."""

    NO_THRESHOLD = 1
    """No threshold meets the budget. The tool worked; the answer is that this cache
    should not ship at this budget."""

    REFUSED = 2
    """The corpus is too small to resolve the requested error budget. The tool prints
    the number of hits, and hence the corpus size, the budget would require."""

    ERROR = 3
    """Tool error."""


@dataclass
class Decision:
    """The decision, the evidence behind it, and what it would take to change it."""

    exit_code: Exit
    budget: float
    min_hit_rate: float
    confidence: float = CONFIDENCE
    operating_point: Point | None = None
    reason: str = ""
    hits_required: int | None = None
    corpus_pairs_required: int | None = None
    best_achievable_hits: int = 0
    feasible_thresholds: int = 0
    worst_class: str = ""
    worst_class_false_hit_rate: float = 0.0
    wrong_prefix_hits: int = 0
    clean_ceiling_hits: int = 0
    clean_ceiling_hit_rate: float = 0.0
    tightest_certifiable_budget: float = 1.0
    positive_controls: int = 0
    separation_auc: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "exit_code": int(self.exit_code),
            "exit_name": self.exit_code.name,
            "budget": self.budget,
            "min_hit_rate": self.min_hit_rate,
            "confidence": self.confidence,
            "reason": self.reason,
            "hits_required": self.hits_required,
            "corpus_pairs_required": self.corpus_pairs_required,
            "best_achievable_hits": self.best_achievable_hits,
            "feasible_thresholds": self.feasible_thresholds,
            "worst_class": self.worst_class,
            "worst_class_false_hit_rate": self.worst_class_false_hit_rate,
            "wrong_prefix_hits": self.wrong_prefix_hits,
            "clean_ceiling_hits": self.clean_ceiling_hits,
            "clean_ceiling_hit_rate": self.clean_ceiling_hit_rate,
            "tightest_certifiable_budget": self.tightest_certifiable_budget,
            "positive_controls": self.positive_controls,
            "separation_auc": self.separation_auc,
            "operating_point": self.operating_point.to_dict() if self.operating_point else None,
            "notes": list(self.notes),
        }


def decide(sweep: Sweep, *, budget: float, min_hit_rate: float = 0.10) -> Decision:
    """Pick the threshold that serves the most traffic while provably meeting `budget`.

    "Provably" is doing real work. A threshold qualifies when the upper confidence
    bound on its false hit rate is under the budget, not when its point estimate is.
    The difference is the entire gap between a cache that has been shown to be safe
    and one that has merely not yet been caught.
    """
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be strictly between 0 and 1, got {budget}")

    hits_required = trials_needed_for(budget)
    positive_controls = sum(1 for p in sweep.pairs if p.same_answer)
    common = {
        "wrong_prefix_hits": sweep.wrong_prefix_hits,
        "clean_ceiling_hits": sweep.clean_ceiling_hits,
        "clean_ceiling_hit_rate": sweep.clean_ceiling_hit_rate,
        "tightest_certifiable_budget": upper_bound(0, positive_controls),
        "positive_controls": positive_controls,
        "separation_auc": sweep.separation_auc,
    }
    best_hits = sweep.max_hits
    near_miss_share = 1.0 - positive_controls / max(len(sweep.pairs), 1)

    # Refusal comes first, because it is a statement about the evidence and does not
    # depend on which way the measurement happened to fall.
    #
    # The ceiling is the positive control count, not the probe count. A hit that is
    # clean is a hit on a pair whose two sides really do share an answer, so the best
    # case any embedder could ever produce is every positive control hitting and no
    # near miss hitting. If certifying the budget needs more clean hits than there are
    # positive controls, no embedder and no threshold can reach it on this corpus, and
    # saying so is more useful than reporting a flattering rate over too few trials.
    if positive_controls < hits_required:
        pairs_required = _pairs_required(hits_required, positive_controls, len(sweep.pairs))
        return Decision(
            exit_code=Exit.REFUSED,
            **common,
            budget=budget,
            min_hit_rate=min_hit_rate,
            best_achievable_hits=positive_controls,
            hits_required=hits_required,
            corpus_pairs_required=pairs_required,
            reason=(
                f"the corpus cannot resolve a {budget:.2%} budget. Certifying it at "
                f"{CONFIDENCE:.0%} confidence needs at least {hits_required} clean hits, "
                f"and this corpus has only {positive_controls} positive control pairs, "
                "which is the most clean hits any embedder could produce."
            ),
            notes=[
                f"a flawless run over all {positive_controls} positive controls only "
                f"bounds the false hit rate at {upper_bound(0, positive_controls):.2%}",
                f"the tightest budget this corpus can certify is "
                f"{upper_bound(0, positive_controls):.2%}",
                f"reaching {budget:.2%} needs about {pairs_required} pairs, up from "
                f"{len(sweep.pairs)}",
                f"near miss share of this corpus is {near_miss_share:.1%}",
            ],
        )

    feasible = [
        p
        for p in sweep.points
        if p.hits >= hits_required
        and p.hit_rate >= min_hit_rate
        and p.false_hit_rate_upper <= budget
    ]
    if not feasible:
        usable = [p for p in sweep.points if p.hit_rate >= min_hit_rate]
        # Report the threshold that comes closest to the budget rather than the most
        # permissive one. The most permissive threshold always hits everything, which
        # tells a reader nothing about how far from shippable the cache actually is.
        best = (
            min(usable, key=lambda p: (p.false_hit_rate_upper, -p.hit_rate))
            if usable
            else max(sweep.points, key=lambda p: p.hit_rate)
        )
        worst_cls, worst_rate = sweep.worst_class(best)
        return Decision(
            exit_code=Exit.NO_THRESHOLD,
            **common,
            budget=budget,
            min_hit_rate=min_hit_rate,
            operating_point=best,
            best_achievable_hits=best_hits,
            hits_required=hits_required,
            worst_class=worst_cls,
            worst_class_false_hit_rate=worst_rate,
            reason=(
                f"no threshold meets a {budget:.2%} budget at a {min_hit_rate:.0%} hit "
                "rate. The tool worked; the answer is that this cache should not ship."
            ),
            notes=[
                f"the closest any usable threshold gets is {best.false_hit_rate:.2%} "
                f"measured over {best.hits} hits, bounded at "
                f"{best.false_hit_rate_upper:.2%}",
                f"separation of this embedder is {sweep.separation_auc:.3f}, where 0.5 "
                "is chance and below 0.5 means near misses outrank true paraphrases",
                f"worst class at that threshold is {worst_cls}",
            ],
        )

    # Among thresholds that provably meet the budget, serve the most traffic.
    chosen = max(feasible, key=lambda p: (p.hit_rate, -p.threshold))
    worst_cls, worst_rate = sweep.worst_class(chosen)
    return Decision(
        exit_code=Exit.OK,
        **common,
        budget=budget,
        min_hit_rate=min_hit_rate,
        operating_point=chosen,
        best_achievable_hits=best_hits,
        hits_required=hits_required,
        feasible_thresholds=len(feasible),
        worst_class=worst_cls,
        worst_class_false_hit_rate=worst_rate,
        reason=(
            f"threshold {chosen.threshold:.4f} serves {chosen.hit_rate:.1%} of probes "
            f"with a false hit rate bounded at {chosen.false_hit_rate_upper:.2%}."
        ),
        notes=[
            f"measured false hit rate at this threshold is {chosen.false_hit_rate:.2%} "
            f"over {chosen.hits} hits, bounded at {chosen.false_hit_rate_upper:.2%}",
            "the prevalence reweighted rate is reported alongside, because the corpus "
            "is deliberately near miss heavy and a rate read straight off it is a rate "
            "on an adversarial stream",
        ],
    )


def _pairs_required(hits_required: int, positive_controls: int, corpus_pairs: int) -> int:
    """Scale the whole corpus until its positive controls reach `hits_required`.

    Keeping the near miss to positive ratio fixed, because changing it to reach the
    number faster would change what the corpus measures.
    """
    if positive_controls <= 0:
        return hits_required
    return int(-(-corpus_pairs * hits_required // positive_controls))
