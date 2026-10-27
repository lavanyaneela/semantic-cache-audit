"""Sweeps the similarity threshold and reports what each setting buys and costs.

The two columns that matter sit side by side at every threshold: the share of traffic
served from cache, and the share of those served answers that are wrong. A hit rate
alone is the metric this repository exists to argue against.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from ..corpus.schema import NEAR_MISS_CLASSES, Pair
from .probe import Probe, pair_scores, probe_at_threshold
from .statistics import resolution_floor, separation_auc, upper_bound

#: Declared, not observed. The share of cache probes that are near misses rather than
#: true paraphrases. The corpus is deliberately near-miss heavy so the dangerous region
#: is measurable at all, which means a rate read straight off it is a rate on an
#: adversarial stream. Every published rate is therefore reported twice: on the corpus,
#: and reweighted to this prevalence. ADR-005 states that the prevalence is an input.
DEFAULT_NEAR_MISS_PREVALENCE = 0.05


def threshold_grid(scores: np.ndarray, steps: int = 101) -> np.ndarray:
    """A grid that spans the observed score range, with the extremes included.

    Spanning the observed range rather than a fixed 0 to 1 matters: an embedder whose
    scores all sit above 0.8 would otherwise be swept almost entirely in a region where
    everything hits, and the curve would be a flat line through the interesting part.
    """
    low = float(np.min(scores))
    high = float(np.max(scores))
    if high <= low:
        return np.array([low], dtype=np.float64)
    pad = (high - low) * 1e-6
    return np.linspace(low - pad, high + pad, steps)


@dataclass
class Point:
    """One threshold, fully priced."""

    threshold: float
    hits: int
    false_hits: int
    probes: int
    hit_rate: float
    false_hit_rate: float
    false_hit_rate_upper: float
    missed_opportunities: int
    weighted_hit_rate: float
    weighted_false_hit_rate: float
    resolution_floor: float
    by_class: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _class_weights(pairs: list[Pair], prevalence: float) -> dict[str, float]:
    """Per-probe weights that reweight the corpus to a declared near-miss prevalence."""
    n_near = sum(1 for p in pairs if p.is_near_miss)
    n_pos = len(pairs) - n_near
    near_w = (prevalence / n_near) if n_near else 0.0
    pos_w = ((1.0 - prevalence) / n_pos) if n_pos else 0.0
    return {"near": near_w, "pos": pos_w}


def summarise(
    probes: list[Probe],
    threshold: float,
    pairs: list[Pair],
    prevalence: float = DEFAULT_NEAR_MISS_PREVALENCE,
) -> Point:
    hits = [p for p in probes if p.hit]
    false_hits = [p for p in hits if not p.correct]
    weights = _class_weights(pairs, prevalence)
    w_hit = sum(weights["near"] if not p.correct else weights["pos"] for p in hits)
    w_false = sum(weights["near"] for p in false_hits)

    by_class: dict[str, dict] = {}
    grouped: dict[str, list[Probe]] = defaultdict(list)
    for probe in probes:
        grouped[probe.cls].append(probe)
    for cls, members in sorted(grouped.items()):
        cls_hits = [m for m in members if m.hit]
        cls_false = [m for m in cls_hits if not m.correct]
        by_class[cls] = {
            "probes": len(members),
            "hits": len(cls_hits),
            "false_hits": len(cls_false),
            "hit_rate": len(cls_hits) / len(members) if members else 0.0,
            "false_hit_rate": len(cls_false) / len(cls_hits) if cls_hits else 0.0,
            "false_hit_rate_upper": upper_bound(len(cls_false), len(cls_hits)),
        }

    return Point(
        threshold=float(threshold),
        hits=len(hits),
        false_hits=len(false_hits),
        probes=len(probes),
        hit_rate=len(hits) / len(probes) if probes else 0.0,
        false_hit_rate=len(false_hits) / len(hits) if hits else 0.0,
        false_hit_rate_upper=upper_bound(len(false_hits), len(hits)),
        missed_opportunities=sum(1 for p in probes if p.missed_opportunity),
        weighted_hit_rate=w_hit,
        weighted_false_hit_rate=(w_false / w_hit) if w_hit > 0 else 0.0,
        resolution_floor=resolution_floor(len(hits)),
        by_class=by_class,
    )


@dataclass
class Sweep:
    """The full curve for one embedder, plus the separation that produced it."""

    embedder: str
    points: list[Point]
    separation_auc: float
    scores: np.ndarray
    pairs: list[Pair]

    @property
    def wrong_prefix_hits(self) -> int:
        """How many hits the cache makes, from its most confident downward, before the
        first correct one.

        The one statistic here that cannot be read off the corpus composition. A false
        hit rate at a fully permissive threshold is just the near miss share of the
        corpus and says nothing about the embedder. This counts the leading run of
        wrong answers in confidence order, so it is zero for an embedder that ranks any
        true paraphrase first and large only for one that is actively inverted.
        """
        order = sorted(
            zip(self.scores, [p.same_answer for p in self.pairs], strict=True),
            key=lambda kv: -kv[0],
        )
        run = 0
        for _, correct in order:
            if correct:
                break
            run += 1
        return run

    @property
    def max_hits(self) -> int:
        """The most hits any threshold produces. The ceiling on what the corpus resolves."""
        return max((p.hits for p in self.points), default=0)

    def worst_class(self, point: Point) -> tuple[str, float]:
        """The near-miss class with the highest false hit rate at this threshold."""
        candidates = [
            (cls, stats["false_hit_rate"])
            for cls, stats in point.by_class.items()
            if cls in NEAR_MISS_CLASSES and stats["hits"] > 0
        ]
        if not candidates:
            return ("none", 0.0)
        return max(candidates, key=lambda kv: (kv[1], kv[0]))

    def to_dict(self) -> dict:
        return {
            "embedder": self.embedder,
            "separation_auc": self.separation_auc,
            "points": [p.to_dict() for p in self.points],
        }


def run_sweep(
    pairs: list[Pair],
    texts: list[str],
    matrix: np.ndarray,
    embedder: str,
    *,
    steps: int = 101,
    prevalence: float = DEFAULT_NEAR_MISS_PREVALENCE,
) -> Sweep:
    scores = pair_scores(pairs, texts, matrix)
    positives = [s for p, s in zip(pairs, scores, strict=True) if p.same_answer]
    negatives = [s for p, s in zip(pairs, scores, strict=True) if p.is_near_miss]
    points = [
        summarise(probe_at_threshold(pairs, scores, t), t, pairs, prevalence)
        for t in threshold_grid(scores, steps)
    ]
    return Sweep(
        embedder=embedder,
        points=points,
        separation_auc=separation_auc(positives, negatives),
        scores=scores,
        pairs=pairs,
    )
