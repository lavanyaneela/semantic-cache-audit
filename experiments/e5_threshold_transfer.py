"""E5: does a threshold transfer to another embedder?

The expected answer is no, and publishing the size of the failure is the point. A
threshold is the one number a team copies between systems, from a blog post, from a
previous project, from a vendor default, and it is a fact about an embedding space
rather than about caching.

Transfer is measured the way a team would actually do it: take the threshold tuned on
embedder A, apply it unchanged to embedder B, and report what the cache does then.
"""

from __future__ import annotations

import sys

import numpy as np

from semantic_cache_audit import report
from semantic_cache_audit.audit.probe import pair_scores, probe_at_threshold
from semantic_cache_audit.audit.sweep import summarise
from semantic_cache_audit.config import DEFAULTS
from semantic_cache_audit.embed.registry import fixture_names

from _harness import REAL_EMBEDDERS, corpus, fixture, write


def main() -> int:
    pairs = corpus()
    names = list(fixture_names())
    scores = {}
    tuned = {}
    for name in names:
        texts, matrix = fixture(name)
        scores[name] = pair_scores(pairs, texts, matrix)
        positives = [s for p, s in zip(pairs, scores[name], strict=True) if p.same_answer]
        # The threshold a team would pick: serve half the true paraphrases.
        tuned[name] = float(np.median(positives))

    records = []
    for source in names:
        threshold = tuned[source]
        home = summarise(probe_at_threshold(pairs, scores[source], threshold), threshold, pairs,
                         DEFAULTS.near_miss_prevalence)
        for target in names:
            if target == source:
                continue
            away = summarise(probe_at_threshold(pairs, scores[target], threshold), threshold,
                             pairs, DEFAULTS.near_miss_prevalence)
            records.append({
                "source": source,
                "target": target,
                "real_only": source in REAL_EMBEDDERS and target in REAL_EMBEDDERS,
                "threshold": threshold,
                "native_threshold_on_target": tuned[target],
                "threshold_shift_points": abs(threshold - tuned[target]) * 100,
                "home_hit_rate": home.hit_rate,
                "home_false_hit_rate": home.false_hit_rate,
                "away_hit_rate": away.hit_rate,
                "away_false_hit_rate": away.false_hit_rate,
                "hit_rate_change_pp": (away.hit_rate - home.hit_rate) * 100,
                "false_hit_change_pp": (away.false_hit_rate - home.false_hit_rate) * 100,
            })

    # Split the real embedders from the instrument before reporting anything.
    # Pooling them produces a larger and more impressive set of transfers whose
    # members are mostly draws from a seeded family, and quoting that as a result
    # about real models is exactly the misreading ADR-005 forbids. The headline is
    # the real only figure; the instrument sweep is reported beside it as what it is.
    real = [r for r in records if r["real_only"]]
    instrument = [r for r in records if not r["real_only"]]
    shifts = [r["threshold_shift_points"] for r in records]
    # One magnitude per embedder pair. The shift is symmetric, so taking the median
    # over the directed transfers counted every magnitude twice and landed the
    # median in a gap between two clusters that no pair occupies.
    real_shifts = sorted(
        {
            (
                min(r["source"], r["target"]),
                max(r["source"], r["target"]),
                round(r["threshold_shift_points"], 6),
            )
            for r in real
        }
    )
    real_shifts = [shift for _, _, shift in real_shifts]
    worst = max(real, key=lambda r: abs(r["false_hit_change_pp"]))
    real_collapsed = [r for r in real if r["away_hit_rate"] < 0.01]
    real_saturated = [r for r in real if r["away_hit_rate"] > 0.99]
    collapsed = [r for r in records if r["away_hit_rate"] < 0.01]
    saturated = [r for r in records if r["away_hit_rate"] > 0.99]
    payload = {
        "question": "Does the threshold transfer to another embedder?",
        "records": records,
        "native_thresholds": tuned,
        "real_embedders": list(REAL_EMBEDDERS),
        # The published figures. Real embedders only.
        "transfers_real_only": len(real),
        "real_embedder_pairs": len(real_shifts),
        "real_threshold_shift_median_points": float(np.median(real_shifts)),
        "real_threshold_shift_max_points": float(np.max(real_shifts)),
        "real_transfers_that_collapse_the_cache": len(real_collapsed),
        "real_transfers_that_saturate_the_cache": len(real_saturated),
        "real_worst_false_hit_change_pp": worst["false_hit_change_pp"],
        "real_worst_transfer": f"{worst['source']} to {worst['target']}",
        "real_transfers_within_five_points": sum(1 for s in real_shifts if s <= 5.0),
        "real_transfers_where_every_hit_is_wrong": sum(
            1 for r in real if r["away_false_hit_rate"] >= 1.0
        ),
        # The instrument sweep, reported separately and never quoted as a model result.
        "transfers_involving_instrument": len(instrument),
        "pooled_transfers_measured": len(records),
        "pooled_threshold_shift_median_points": float(np.median(shifts)),
        "pooled_transfers_that_collapse_the_cache": len(collapsed),
        "pooled_transfers_that_saturate_the_cache": len(saturated),
        "answer": (
            "a threshold is a fact about one embedding space. Carried across, it "
            "either stops serving traffic or stops rejecting anything."
        ),
    }
    path = write("e5_threshold_transfer", payload)

    headline = real
    print(report.table(
        ["from", "to", "threshold", "shift pts", "hit rate there", "wrong share there"],
        [[r["source"], r["target"], f"{r['threshold']:.4f}",
          f"{r['threshold_shift_points']:.1f}", report.pct(r["away_hit_rate"], 1),
          report.pct(r["away_false_hit_rate"], 1)] for r in headline],
    ))
    print(f"\nreal embedders only: {len(real)} transfers over {len(real_shifts)} "
          f"embedder pairs, median shift {np.median(real_shifts):.1f} cosine points, "
          f"max {np.max(real_shifts):.1f}")
    print(f"  {len(real_collapsed)} drop the hit rate below 1 percent, "
          f"{len(real_saturated)} push it above 99 percent")
    print(f"instrument sweep, reported separately: {len(instrument)} further transfers, "
          f"median shift {np.median(shifts):.1f} pooled")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
