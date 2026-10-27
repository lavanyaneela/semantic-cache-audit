"""E4: can any threshold meet a realistic error budget?

This is the decision table behind the exit codes. It is run across a grid of budgets
and across every embedder fixture, so the README can say which outcome each
combination produces rather than describing the codes in the abstract.

The row that matters most is the refusal. Below a certain budget the answer stops
depending on the embedder at all and becomes a statement about the corpus: with 120
positive control pairs, no run however clean can certify a rate tighter than the
bound a flawless run would give, so the tool declines instead of reporting a
flattering zero.
"""

from __future__ import annotations

import sys

from semantic_cache_audit import report
from semantic_cache_audit.audit.budget import Exit, decide
from semantic_cache_audit.audit.statistics import trials_needed_for
from semantic_cache_audit.audit.sweep import run_sweep
from semantic_cache_audit.config import DEFAULTS
from semantic_cache_audit.embed.registry import fixture_names

from _harness import REAL_EMBEDDERS, SHAPED_EMBEDDER, corpus, fixture, write

BUDGETS: tuple[float, ...] = (0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20)


def main() -> int:
    pairs = corpus()
    names = list(fixture_names())
    sweeps = {}
    for name in names:
        texts, matrix = fixture(name)
        sweeps[name] = run_sweep(pairs, texts, matrix, name, steps=DEFAULTS.threshold_steps,
                                 prevalence=DEFAULTS.near_miss_prevalence)

    records = []
    for budget in BUDGETS:
        for name in names:
            decision = decide(sweeps[name], budget=budget, min_hit_rate=DEFAULTS.min_hit_rate)
            point = decision.operating_point
            records.append({
                "budget": budget,
                "embedder": name,
                "separation_auc": sweeps[name].separation_auc,
                "exit_code": int(decision.exit_code),
                "exit_name": decision.exit_code.name,
                "hits_required": decision.hits_required,
                "corpus_pairs_required": decision.corpus_pairs_required,
                "threshold": point.threshold if point else None,
                "hit_rate": point.hit_rate if point else None,
                "false_hit_rate": point.false_hit_rate if point else None,
                "false_hit_rate_upper": point.false_hit_rate_upper if point else None,
            })

    positive_controls = sum(1 for p in pairs if p.same_answer)
    tightest = next(iter(sweeps.values()))
    tightest_budget = decide(tightest, budget=0.5).tightest_certifiable_budget
    refused_below = max(
        (b for b in BUDGETS if all(
            r["exit_name"] == "REFUSED" for r in records if r["budget"] == b)),
        default=0.0,
    )
    ok_rows = [r for r in records if r["exit_name"] == "OK"]

    payload = {
        "question": "Can any threshold meet a realistic error budget?",
        "budgets": list(BUDGETS),
        "records": records,
        "positive_controls": positive_controls,
        "tightest_certifiable_budget": tightest_budget,
        "every_embedder_refused_at_or_below": refused_below,
        "hits_required_at_one_percent": trials_needed_for(0.01),
        "corpus_pairs_required_at_one_percent": next(
            r["corpus_pairs_required"] for r in records
            if r["budget"] == 0.01 and r["corpus_pairs_required"]
        ),
        "real_embedders_ever_ok": sum(
            1 for r in ok_rows if r["embedder"] in REAL_EMBEDDERS
        ),
        "shaped_embedder_first_ok_budget": min(
            (r["budget"] for r in ok_rows if r["embedder"] == SHAPED_EMBEDDER), default=None
        ),
        "exit_code_counts": {
            e.name: sum(1 for r in records if r["exit_name"] == e.name) for e in Exit
        },
    }
    path = write("e4_budget_decisions", payload)

    headers = ["budget", *names]
    rows = []
    for budget in BUDGETS:
        row = [report.pct(budget, 2)]
        for name in names:
            record = next(r for r in records if r["budget"] == budget and r["embedder"] == name)
            row.append({"OK": "pass", "NO_THRESHOLD": "no", "REFUSED": "refuse"}[
                record["exit_name"]])
        rows.append(row)
    print(report.table(headers, rows))
    print(f"\ntightest budget this corpus can certify: {report.pct(tightest_budget)} "
          f"from {positive_controls} positive controls")
    print(f"every embedder is refused at or below {report.pct(refused_below)}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
