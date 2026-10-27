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

from _harness import HEADLINE_EMBEDDER, corpus, fixture, reference_threshold, write

CAPACITIES: tuple[int | None, ...] = (20, 50, 100, None)
TTLS: tuple[float | None, ...] = (600.0, 3600.0, None)
N_REQUESTS = 4000


def main() -> int:
    pairs = corpus()
    texts, matrix = fixture(HEADLINE_EMBEDDER)
    from semantic_cache_audit.audit.probe import pair_scores

    threshold = reference_threshold(pairs, pair_scores(pairs, texts, matrix))
    requests = build_stream(pairs, n_requests=N_REQUESTS, zipf_a=1.15)
    schedule = change_schedule(pairs, horizon=requests[-1].at)

    rows = []
    records = []
    for capacity in CAPACITIES:
        for policy in POLICIES:
            result = replay(
                requests, texts, matrix,
                threshold=threshold, policy=policy, capacity=capacity,
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
            threshold=threshold, policy="lru", capacity=DEFAULTS.capacity,
            ttl_seconds=ttl, schedule=schedule,
        )
        records.append({"capacity": DEFAULTS.capacity, "policy": "lru", "ttl_seconds": ttl,
                        **result.to_dict()})
        ttl_rows.append([
            str(ttl), report.pct(result.hit_rate, 1), report.pct(result.false_hit_rate, 1),
            report.pct(result.stale_hit_rate, 1),
        ])

    bound = [r for r in records if r["capacity"] == 20 and r["ttl_seconds"] == DEFAULTS.ttl_seconds]
    # ttl_only ignores capacity, so it is not a peer of the three bounding policies. It
    # is reported separately as the "let it grow" arm rather than averaged in, which
    # would blur the two levers this experiment is trying to separate.
    bounding = [r for r in bound if r["policy"] != "ttl_only"]
    unbounded = next(
        r for r in records if r["capacity"] is None and r["policy"] == "lru"
    )
    false_spread = max(r["false_hit_rate"] for r in bounding) - min(
        r["false_hit_rate"] for r in bounding)
    stale_spread = max(r["stale_hit_rate"] for r in bounding) - min(
        r["stale_hit_rate"] for r in bounding)
    best_stale = min(bounding, key=lambda r: (r["stale_hit_rate"], r["policy"]))
    best_hit = max(bounding, key=lambda r: (r["hit_rate"], r["policy"]))
    best_false = min(bounding, key=lambda r: (r["false_hit_rate"], r["policy"]))

    payload = {
        "question": "Does eviction policy change correctness or only cost?",
        "embedder": HEADLINE_EMBEDDER,
        "threshold": threshold,
        "requests": N_REQUESTS,
        # Named because it was chosen. Capacity 20 is the tightest arm of the sweep
        # and every figure below is weaker at a larger cache, so publishing the
        # number without the capacity beside it would be selection without
        # disclosure. The full grid is in `records`.
        "published_capacity": 20,
        "capacities_swept": len(CAPACITIES),
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
        "lowest_false_hit_policy": best_false["policy"],
        "lowest_false_hit_rate": best_false["false_hit_rate"],
        "hit_rate_paid_for_lowest_stale_pp": (best_hit["hit_rate"] - best_stale["hit_rate"]) * 100,
        "stale_reduction_ratio_fifo_over_lru": (
            next(r["stale_hit_rate"] for r in bounding if r["policy"] == "lru")
            / next(r["stale_hit_rate"] for r in bounding if r["policy"] == "fifo")
        ),
        "bounded_false_hit_rate_lru": next(
            r["false_hit_rate"] for r in bounding if r["policy"] == "lru"
        ),
        "unbounded_false_hit_rate": unbounded["false_hit_rate"],
        "false_hit_ratio_unbounded_over_bounded": (
            unbounded["false_hit_rate"]
            / next(r["false_hit_rate"] for r in bounding if r["policy"] == "lru")
        ),
        "answer": (
            "there are two levers here and they do different jobs. Choosing between "
            "LRU, LFU and FIFO at a fixed capacity barely moves correctness and moves "
            "staleness several fold, so policy is a staleness lever. Whether the cache "
            "is bounded at all is a correctness lever: letting it grow without limit "
            "roughly doubles the false hit rate, because a wrong answer needs a wrong "
            "neighbour to still be resident. Reporting one eviction number would hide "
            "both."
        ),
        "nuance_the_brief_missed": (
            "the brief framed eviction as a single question. Capacity and policy pull "
            "on different failure modes and have to be priced separately."
        ),
    }
    path = write("e3_eviction_sweep", payload)

    print(report.table(
        ["capacity", "policy", "hit rate", "wrong share", "stale share", "evictions"], rows))
    print()
    print(report.table(["ttl seconds", "hit rate", "wrong share", "stale share"], ttl_rows))
    print(f"\nacross bounding policies at capacity 20: false hit spread "
          f"{false_spread * 100:.2f} points, stale hit spread {stale_spread * 100:.2f} points")
    print(f"bounded LRU false hit rate {report.pct(bounding[0]['false_hit_rate'])} against "
          f"{report.pct(unbounded['false_hit_rate'])} unbounded")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
