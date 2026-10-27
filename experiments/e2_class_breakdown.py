"""E2: which near miss class is most dangerous?

A single false hit rate tells a team the cache is unsafe. A ranked breakdown tells
them which guard to write first, which is the difference between a finding and an
action.

The measurement is taken at a **reference threshold**, defined as the median positive
control score: the setting at which half of all true paraphrases are served from
cache. Two reasons it is not the budget operating point. At a conservative operating
point each class contributes only a handful of hits, and a hundred percent over two
hits is receipts rule four's warning rather than a result. And a reference point tied
to paraphrase coverage rather than to a budget makes the classes comparable across
embedders, which the budget operating point does not, because it sits at a different
place on every curve.

Each class also gets a separation statistic of its own: how often a true paraphrase
outscores a member of that class. Below one half means that class is harder to
reject than a real paraphrase is to accept, and it is the only figure here that does
not depend on where the threshold was put.
"""

from __future__ import annotations

import sys

import numpy as np

from semantic_cache_audit import report
from semantic_cache_audit.audit.probe import pair_scores
from semantic_cache_audit.audit.statistics import resolution_floor, separation_auc, upper_bound
from semantic_cache_audit.corpus.schema import NEAR_MISS_CLASSES

from _harness import HEADLINE_EMBEDDER, SHAPED_EMBEDDER, corpus, fixture, write


def analyse(pairs, scores: np.ndarray) -> dict:
    positives = np.array([s for p, s in zip(pairs, scores, strict=True) if p.same_answer])
    reference = float(np.median(positives))
    rows = []
    for cls in NEAR_MISS_CLASSES:
        member = np.array([s for p, s in zip(pairs, scores, strict=True) if p.cls == cls])
        hits = int((member >= reference).sum())
        rows.append(
            {
                "class": cls,
                "probes": int(member.size),
                "hits": hits,
                "false_hits": hits,
                "false_hit_rate": hits / member.size if member.size else 0.0,
                "false_hit_rate_upper": upper_bound(hits, int(member.size)),
                "resolution_floor": resolution_floor(int(member.size)),
                "median_score": float(np.median(member)),
                "min_score": float(np.min(member)) if member.size else 0.0,
                "separation_auc": separation_auc(positives, member),
            }
        )
    rows.sort(key=lambda r: (-r["false_hit_rate"], r["separation_auc"], r["class"]))
    worst_class_min = min(
        (r for r in rows), key=lambda r: r["min_score"]
    )
    return {
        "reference_threshold": reference,
        "probes_per_class": rows[0]["probes"] if rows else 0,
        "lowest_min_score_class": worst_class_min["class"],
        "reference_definition": "median positive control score, so half of true paraphrases hit",
        "positive_controls": int(positives.size),
        "ranked": rows,
    }


def main() -> int:
    pairs = corpus()
    out = {}
    for name in (HEADLINE_EMBEDDER, SHAPED_EMBEDDER):
        texts, matrix = fixture(name)
        out[name] = analyse(pairs, pair_scores(pairs, texts, matrix))

    headline = out[HEADLINE_EMBEDDER]
    worst = headline["ranked"][0]
    safest = headline["ranked"][-1]
    payload = {
        "question": "Which near miss class is most dangerous?",
        "headline_embedder": HEADLINE_EMBEDDER,
        "by_embedder": out,
        "worst_class_name": worst["class"],
        "worst_class_false_hit_pct": worst["false_hit_rate"],
        "safest_class_name": safest["class"],
        "safest_class_false_hit_pct": safest["false_hit_rate"],
        "classes_at_total_failure": sum(
            1 for r in headline["ranked"] if r["false_hit_rate"] >= 1.0
        ),
        "classes_below_chance_separation": sum(
            1 for r in headline["ranked"] if r["separation_auc"] < 0.5
        ),
        "probes_per_class": headline["probes_per_class"],
        "rate_by_class": {r["class"]: r["false_hit_rate"] for r in headline["ranked"]},
        "upper_by_class": {r["class"]: r["false_hit_rate_upper"] for r in headline["ranked"]},
        "safest_class_upper": next(
            r["false_hit_rate_upper"] for r in headline["ranked"] if r["class"] == safest["class"]
        ),
        # The floor of the worst class: not its average, its weakest member. A mean
        # could hide an easy pair dragging it down, and the claim being made is that
        # the whole class sits above the threshold.
        "worst_class_min_similarity": next(
            r["min_score"] for r in headline["ranked"] if r["class"] == worst["class"]
        ),
    }
    path = write("e2_class_breakdown", payload)

    for name, data in out.items():
        print(f"\n{name}, reference threshold {data['reference_threshold']:.4f} "
              f"(half of {data['positive_controls']} positive controls hit)")
        print(report.table(
            ["class", "probes", "wrong hits", "wrong share", "upper bound", "separation"],
            [
                [r["class"], str(r["probes"]), str(r["hits"]),
                 report.pct(r["false_hit_rate"]), report.pct(r["false_hit_rate_upper"]),
                 f"{r['separation_auc']:.3f}"]
                for r in data["ranked"]
            ],
        ))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
