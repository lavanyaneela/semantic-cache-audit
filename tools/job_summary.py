"""Writes the uncomfortable table to the GitHub Actions job summary.

The job summary is where a human actually looks after a run, so the figures that
matter go there rather than into a log line nobody scrolls back to. The table leads
with the findings a reader would rather not see, which is the same ordering the
README uses.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROWS = (
    ("Separation of the pretrained encoder", "separation_auc", "0.5 is chance"),
    ("Most confident matches that are all wrong", "wrong_prefix_hits", "consecutive hits"),
    ("Paraphrases servable with no wrong answer", "clean_ceiling_hits",
     "out of the positive controls"),
    ("Worst near miss class", "worst_class_name", ""),
    ("Wrong share on that class", "worst_class_false_hit_pct", "percent"),
    ("Tightest budget this corpus can certify", "tightest_budget_pct", "percent"),
    ("Clean hits a one percent budget needs", "hits_required_one_pct", ""),
    ("Real embedders that ever meet a budget", "real_embedders_ever_ok", "out of four"),
    ("Threshold transfers where every hit is wrong", "real_transfers_all_wrong",
     "out of those between real embedders"),
    ("Tests", "tests_total", ""),
    ("Line coverage", "coverage_line_pct", "percent"),
)


def check_rows_resolve(rendered: dict) -> list[str]:
    """Every row must name a metric that exists.

    One did not, and the job summary printed "n/a" for it in every CI run since it
    was written. Nothing noticed, because the summary is the one output no test
    reads, so the check lives here and a test calls it.
    """
    return [metric for _, metric, _ in ROWS if metric not in rendered]


def main() -> int:
    path = ROOT / "docs" / "metrics.json"
    if not path.exists():
        print("### Receipts\n\nNo metrics.json was produced.")
        return 0
    payload = json.loads(path.read_text())
    rendered = payload["rendered"]
    missing = check_rows_resolve(rendered)
    if missing:
        raise SystemExit(f"the job summary names metrics that do not exist: {missing}")
    print("### The numbers this build re-measured\n")
    print("| Figure | Value | |")
    print("| --- | ---: | --- |")
    for label, metric, suffix in ROWS:
        print(f"| {label} | {rendered.get(metric, 'n/a')} | {suffix} |")
    print(
        f"\nEvery figure above was produced by code that ran in this job, and "
        f"`tools/check_numbers.py` verified all {len(payload['metrics'])} of them "
        f"against {len(payload['checked_documents'])} documents."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
