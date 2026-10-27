"""E1: what does the hit rate buy, and what does it cost?

The question the README asks first. A hit rate is published by every cache; the second
column, the share of those hits that returned the wrong answer, is the one that
decides whether the first column is good news.
"""

from __future__ import annotations

import sys

from semantic_cache_audit import report
from semantic_cache_audit.audit.budget import decide
from semantic_cache_audit.audit.sweep import run_sweep
from semantic_cache_audit.config import DEFAULTS

from _harness import HEADLINE_EMBEDDER, REAL_EMBEDDERS, SHAPED_EMBEDDER, corpus, fixture, write


def main() -> int:
    pairs = corpus()
    curves = {}
    for name in (*REAL_EMBEDDERS, SHAPED_EMBEDDER):
        texts, matrix = fixture(name)
        sweep = run_sweep(pairs, texts, matrix, name, steps=DEFAULTS.threshold_steps,
                          prevalence=DEFAULTS.near_miss_prevalence)
        decision = decide(sweep, budget=DEFAULTS.budget, min_hit_rate=DEFAULTS.min_hit_rate)
        point = decision.operating_point
        curves[name] = {
            "separation_auc": sweep.separation_auc,
            "wrong_prefix_hits": sweep.wrong_prefix_hits,
            "clean_ceiling_hits": sweep.clean_ceiling_hits,
            "exit_code": int(decision.exit_code),
            "exit_name": decision.exit_code.name,
            "operating_point": point.to_dict() if point else None,
            "curve": [
                {
                    "threshold": p.threshold,
                    "hit_rate": p.hit_rate,
                    "false_hit_rate": p.false_hit_rate,
                    "false_hit_rate_upper": p.false_hit_rate_upper,
                    "weighted_hit_rate": p.weighted_hit_rate,
                    "weighted_false_hit_rate": p.weighted_false_hit_rate,
                    "hits": p.hits,
                    "false_hits": p.false_hits,
                    "missed_opportunities": p.missed_opportunities,
                }
                for p in sweep.points
            ],
        }

    headline = curves[HEADLINE_EMBEDDER]
    shaped = curves[SHAPED_EMBEDDER]
    payload = {
        "question": "What does the hit rate buy and what does it cost?",
        "budget": DEFAULTS.budget,
        "min_hit_rate": DEFAULTS.min_hit_rate,
        "near_miss_prevalence": DEFAULTS.near_miss_prevalence,
        "headline_embedder": HEADLINE_EMBEDDER,
        "shaped_embedder": SHAPED_EMBEDDER,
        "curves": curves,
        "headline_separation_auc": headline["separation_auc"],
        "headline_wrong_prefix_hits": headline["wrong_prefix_hits"],
        "headline_clean_ceiling_hits": headline["clean_ceiling_hits"],
        "positive_controls": sum(1 for p in pairs if p.same_answer),
        "shaped_operating_point": shaped["operating_point"],
    }
    path = write("e1_threshold_curve", payload)

    rows = []
    for name, data in curves.items():
        point = data["operating_point"]
        rows.append([
            name,
            f"{data['separation_auc']:.3f}",
            str(data["wrong_prefix_hits"]),
            str(data["clean_ceiling_hits"]),
            data["exit_name"],
            f"{point['threshold']:.4f}" if point else "-",
            report.pct(point["hit_rate"], 1) if point else "-",
            report.pct(point["false_hit_rate"], 1) if point else "-",
            report.pct(point["weighted_false_hit_rate"], 1) if point else "-",
        ])
    print(report.table(
        ["embedder", "separation", "wrong run", "safe hits", "verdict", "threshold",
         "hit rate", "wrong share", "at prevalence"],
        rows,
    ))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
