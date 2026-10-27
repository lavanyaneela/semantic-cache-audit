"""E3: does eviction policy change correctness, or only cost?

The question matters because teams reach for an eviction change when a cache starts
returning bad answers, and an eviction change is cheap while a threshold change is
not. Two failure modes are counted separately here, because they have different
fixes: a **false hit** returned the answer to a different question, which only the
threshold addresses, and a **stale hit** returned the right question's answer from
before it changed, which only the TTL addresses.

The stream is Zipf skewed rather than uniform. Under a uniform stream every policy
evicts the same thing and the experiment answers itself.
"""

from __future__ import annotations

import sys

from semantic_cache_audit import report
from semantic_cache_audit.audit.stream import build_stream, change_schedule, replay
from semantic_cache_audit.cache.eviction import POLICIES
from semantic_cache_audit.config import DEFAULTS

from _harness import SHAPED_EMBEDDER, corpus, fixture, write

CAPACITIES: tuple[int | None, ...] = (20, 50, 100, None)
TTLS: tuple[float | None, ...] = (600.0, 3600.0, None)
REFERENCE_THRESHOLD = 0.4176  # the E2 reference point for SHAPED_EMBEDDER
N_REQUESTS = 4000


def main() -> int:
    pairs = corpus()
    texts, matrix = fixture(SHAPED_EMBEDDER)
    requests = build_stream(pairs, n_requests=N_REQUESTS, zipf_a=1.15)
    schedule = change_schedule(pairs, horizon=requests[-1].at)

    rows = []
    records = []
    for capacity in CAPACITIES:
        for policy in POLICIES:
            result = replay(
                requests, texts, matrix,
                threshold=REFERENCE_THRESHOLD, policy=policy, capacity=capacity,
                ttl_seconds=DEFAULTS.ttl_seconds, schedule=schedule,
            )
            record = {"capacity": capacity, "policy": policy, "ttl_seconds": DEFAULTS.ttl_seconds,
                      **result.to_dict()}
            records.append(record)
            rows.append([
                str(capacity), policy, report.pct(result.hit_rate, 1),
                report.pct(result.false_hit_rate, 1), report.pct(result.stale_hit_rate, 1),
                str(result.evictions),
            ])

    ttl_rows = []
    for ttl in TTLS:
        result = replay(
            requests, texts, matrix,
            threshold=REFERENCE_THRESHOLD, policy="lru", capacity=DEFAULTS.capacity,
            ttl_seconds=ttl, schedule=schedule,
        )
        records.append({"capacity": DEFAULTS.capacity, "policy": "lru", "ttl_seconds": ttl,
                        **result.to_dict()})
        ttl_rows.append([
            str(ttl), report.pct(result.hit_rate, 1), report.pct(result.false_hit_rate, 1),
            report.pct(result.stale_hit_rate, 1),
        ])

    bound = [r for r in records if r["capacity"] == 20 and r["ttl_seconds"] == DEFAULTS.ttl_seconds]
    false_spread = max(r["false_hit_rate"] for r in bound) - min(
        r["false_hit_rate"] for r in bound)
    stale_spread = max(r["stale_hit_rate"] for r in bound) - min(
        r["stale_hit_rate"] for r in bound)
    best_stale = min(bound, key=lambda r: (r["stale_hit_rate"], r["policy"]))
    best_hit = max(bound, key=lambda r: (r["hit_rate"], r["policy"]))

    payload = {
        "question": "Does eviction policy change correctness or only cost?",
        "embedder": SHAPED_EMBEDDER,
        "threshold": REFERENCE_THRESHOLD,
        "requests": N_REQUESTS,
        "records": records,
        "false_hit_spread_across_policies_pp": false_spread * 100,
        "stale_hit_spread_across_policies_pp": stale_spread * 100,
        "stale_spread_over_false_spread": (
            stale_spread / false_spread if false_spread > 0 else float("inf")
        ),
        "lowest_stale_policy": best_stale["policy"],
        "lowest_stale_rate": best_stale["stale_hit_rate"],
        "highest_hit_policy": best_hit["policy"],
        "highest_hit_rate": best_hit["hit_rate"],
        "hit_rate_paid_for_lowest_stale_pp": (best_hit["hit_rate"] - best_stale["hit_rate"]) * 100,
        "lowest_false_hit_policy": min(bound, key=lambda r: (r["false_hit_rate"], r["policy"]))[
            "policy"
        ],
        "lowest_false_hit_rate": min(r["false_hit_rate"] for r in bound),
        "unbounded_false_hit_rate": next(
            r["false_hit_rate"] for r in records
            if r["capacity"] is None and r["policy"] == "lru"
        ),
        "capacity_20_false_hit_rate_lru": next(
            r["false_hit_rate"] for r in bound if r["policy"] == "lru"
        ),
        "stale_reduction_ratio_fifo_over_lru": (
            next(r["stale_hit_rate"] for r in bound if r["policy"] == "lru")
            / next(r["stale_hit_rate"] for r in bound if r["policy"] == "fifo")
        ),
        "answer": (
            "the brief predicted eviction would move cost and leave correctness alone. "
            "It moves both. A smaller cache returns fewer wrong answers, because a "
            "wrong answer needs a wrong neighbour to still be resident: dropping "
            "capacity from unbounded to 20 entries cuts the false hit rate while "
            "costing hit rate. Within a fixed capacity the policies then trade against "
            "each other, with FIFO cutting staleness several fold and paying for it in "
            "both hit rate and wrong answers."
        ),
        "prediction_that_was_wrong": (
            "eviction policy changes cost and not correctness"
        ),
    }
    path = write("e3_eviction_sweep", payload)

    print(report.table(
        ["capacity", "policy", "hit rate", "wrong share", "stale share", "evictions"], rows))
    print()
    print(report.table(["ttl seconds", "hit rate", "wrong share", "stale share"], ttl_rows))
    print(f"\nacross policies at capacity 20: false hit spread "
          f"{false_spread * 100:.2f} points, stale hit spread {stale_spread * 100:.2f} points")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
