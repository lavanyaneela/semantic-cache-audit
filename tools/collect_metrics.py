"""Re-measures every published figure and writes docs/metrics.json.

Nothing in this script types a number. It runs the test suite, runs every experiment,
reads their machine readable output, and renders each value exactly as a document is
required to quote it. The rendered string is what the checker compares against,
because receipts rule three is about what a reader sees rather than about the float.

Test count and coverage are read from `--junitxml` and `--cov-report=json`, never
parsed from a progress line, because a parsed progress line quietly becomes whatever
the last run happened to print. In CI the suite is its own step and this script is
called with `--skip-tests` so it reads the reports that step produced: a red test
surfacing as a traceback from a metrics script is an hour of misattribution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from metrics_spec import METRICS  # noqa: E402

EXPERIMENTS = ROOT / "experiments"
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"

CHECKED_DOCUMENTS = (
    "README.md",
    "docs/defense-guide.md",
    "docs/adr/ADR-001-hand-built-adversarial-corpus.md",
    "docs/adr/ADR-002-pretrained-encoder-and-the-embedder-panel.md",
    "docs/adr/ADR-003-exact-index-and-offline-backends.md",
    "docs/adr/ADR-004-what-counts-as-a-wrong-hit.md",
    "docs/adr/ADR-005-what-this-repository-cannot-measure.md",
)


def run_suite() -> None:
    REPORTS.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q",
            f"--junitxml={REPORTS / 'junit.xml'}",
            "--cov=semantic_cache_audit",
            "--cov-report=json:reports/coverage.json",
        ],
        cwd=ROOT,
        check=True,
    )


def run_experiments() -> None:
    for script in sorted(EXPERIMENTS.glob("e*_*.py")):
        subprocess.run(
            [sys.executable, script.name],
            cwd=EXPERIMENTS,
            check=True,
            env={**_env(), "PYTHONPATH": "."},
            stdout=subprocess.DEVNULL,
        )


def _env() -> dict:
    import os

    return dict(os.environ)


def read_suite() -> dict:
    junit = REPORTS / "junit.xml"
    coverage = REPORTS / "coverage.json"
    if not junit.exists() or not coverage.exists():
        raise SystemExit(
            f"missing {junit.name} or {coverage.name}. Run the suite first, or drop --skip-tests."
        )
    root = ElementTree.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    total = int(suite.get("tests", 0))
    skipped = int(suite.get("skipped", 0))
    data = json.loads(coverage.read_text())
    return {
        # The collected count, skips included. The encoder tests skip when the
        # optional weights are not cached, so subtracting skips would make the
        # published figure differ between a laptop with the weights and a CI runner
        # without them, and the checker would fail on one of the two. The number of
        # tests in the suite is a property of the suite; whether one ran today is
        # not, so it is recorded separately and anchored nowhere.
        "tests_total": total,
        "tests_skipped": skipped,
        "coverage_line_pct": float(data["totals"]["percent_covered"]),
    }


def read_corpus() -> dict:
    manifest = json.loads((ROOT / "fixtures" / "corpus" / "manifest.json").read_text())
    manifest["__domains__"] = len(manifest["by_domain"])
    manifest["__classes__"] = sum(
        1 for cls, n in manifest["by_class"].items() if n and cls != "true_paraphrase"
    )
    return manifest


def read_experiment(stem: str) -> dict:
    path = DOCS / "experiments" / f"{stem}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}. Run the experiments first.")
    payload = json.loads(path.read_text())
    if stem == "e1_threshold_curve":
        payload["__embedders__"] = len(payload["curves"])
    return payload


def dig(payload: dict, path: tuple[str, ...]):
    node = payload
    for key in path:
        node = node[key]
    return node


def collect(skip_tests: bool) -> dict:
    if not skip_tests:
        run_suite()
        run_experiments()
    sources: dict[str, dict] = {"corpus": read_corpus(), "suite": read_suite()}
    values: dict[str, float | int | str] = {}
    rendered: dict[str, str] = {}
    anchors: dict[str, list[str]] = {}
    for metric in METRICS:
        if metric.source not in sources:
            sources[metric.source] = read_experiment(metric.source)
        raw = dig(sources[metric.source], metric.path)
        values[metric.name] = raw
        rendered[metric.name] = metric.render(raw)
        anchors[metric.name] = list(metric.anchors)
    _assert_anchors_are_distinct(anchors)
    _assert_every_metric_is_anchored(anchors)
    return {
        "metrics": values,
        "rendered": rendered,
        "anchors": anchors,
        "textual": sorted(m.name for m in METRICS if m.textual),
        "checked_documents": list(CHECKED_DOCUMENTS),
        # Carried forward when nothing measured has changed. Stamping it on every
        # run meant `make verify` left the tree dirty, which trains a reader to
        # ignore `git status` in the one repository that asks them to trust it.
        "generated_utc": "",
        "note": (
            "every value here was produced by running the suite and the experiments. "
            "`rendered` is the exact string a document must use; the checker compares "
            "against that rather than against the float, because a reader sees the "
            "string."
        ),
    }


def _timestamp_for(out: Path, payload: dict) -> str:
    """Today's stamp, or the previous one when nothing measured has moved."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not out.exists():
        return now
    previous = json.loads(out.read_text())
    comparable = {k: v for k, v in payload.items() if k != "generated_utc"}
    if all(previous.get(k) == v for k, v in comparable.items()):
        return str(previous.get("generated_utc", now))
    return now


def _assert_anchors_are_distinct(anchors: dict[str, list[str]]) -> None:
    """No two metrics may share an anchor, including by containment."""
    flat = [(name, a) for name, group in anchors.items() for a in group]
    clashes = [
        (n1, a1, n2, a2)
        for i, (n1, a1) in enumerate(flat)
        for n2, a2 in flat[i + 1 :]
        if n1 != n2 and (a1 in a2 or a2 in a1)
    ]
    if clashes:
        raise SystemExit(f"anchors overlap, so one metric could satisfy another: {clashes}")


def _assert_every_metric_is_anchored(anchors: dict[str, list[str]]) -> None:
    """An unanchored metric is a gap in the receipts, reported rather than skipped."""
    missing = sorted(name for name, group in anchors.items() if not group)
    if missing:
        raise SystemExit(f"metrics with no anchor phrase, so nothing guards them: {missing}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="read existing junit and coverage reports instead of running the suite",
    )
    parser.add_argument("--out", default=str(DOCS / "metrics.json"))
    args = parser.parse_args(argv)
    payload = collect(args.skip_tests)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["generated_utc"] = _timestamp_for(out, payload)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = Counter(m.source for m in METRICS)
    print(f"wrote {out} with {len(payload['metrics'])} metrics from {len(counts)} sources")
    for name in sorted(payload["rendered"]):
        print(f"  {name:34s} {payload['rendered'][name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
