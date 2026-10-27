"""Fails the build when a document quotes a figure the code no longer produces.

Two passes in opposite directions.

**Forward.** For every metric, every anchor phrase is turned into a pattern with a
wildcard where the number goes, and the whole document is swept for it. Each match
must carry the value the code produced. The sweep deliberately includes fenced
blocks, inline code spans and image alt text: guarding only prose closes mutation and
leaves insertion wide open, which is how a stale number ends up inside an example
command and gets quoted in an interview.

**Reverse.** Every number in the prose that matches no metric is reported, because a
number nothing re-measures is a number that will go stale silently. Fenced blocks,
code spans, HTML attributes, URLs and an explicit exemption list are excluded here,
since an example invocation is allowed to contain an invented row count and flagging
it trains the reader to ignore the section.

Run `--prove` to see the checker fail on purpose. A checker nobody has watched fail
is a checker nobody should believe.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from metrics_spec import by_name  # noqa: E402

#: Numbers in prose that are genuinely not measurements. Each needs a reason, and a
#: test asserts that no entry here hides a value a metric actually produces.
EXEMPT_NUMBERS: dict[str, str] = {
    "1": "an exit code, and an ordinal in ADR numbering",
    "2": "an exit code",
    "3": "an exit code",
    "0.5": "the chance level of the separation statistic, a definition not a measurement",
    "95": "the confidence level, declared in statistics.py rather than measured",
    "100": "a percentage ceiling used in definitions",
    "2026": "a copyright year",
    "384": "the published dimension of the encoder, a property of the model",
    "3.11": "the minimum supported Python version",
    "3.12": "a supported Python version",
    "3.13": "a supported Python version",
    "10": "the minimum hit rate expressed as a percentage, a configured input",
}

NUMBER = r"(-?\d[\d,]*(?:\.\d+)?)"
WORD = r"\b([A-Za-z_][A-Za-z0-9_]*)\b"

FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
CODE_SPAN = re.compile(r"`[^`\n]+`")
HTML_ATTR = re.compile(r'\w+="[^"]*"')
URL = re.compile(r"https?://\S+")
#: `ADR-004` and `e3_eviction_sweep` are identifiers, not measurements. Stripping
#: them keeps the reverse pass about numbers a reader would read as a figure.
IDENTIFIER = re.compile(r"\bADR-\d+|\be\d_[a-z_]+|\b[A-Za-z_]+-\d+(?:\.\d+)*\b")
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Finding:
    document: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.document}: {self.kind}: {self.detail}"


def normalise(text: str) -> str:
    return text.replace(",", "").strip()


def anchor_pattern(anchor: str, textual: bool) -> re.Pattern[str]:
    """Turn `"{} tests"` into a pattern that matches any number followed by `tests`.

    Whitespace in the anchor matches a line break, because wrapped markdown puts one
    wherever it likes and an anchor that cannot cross a newline silently matches
    nothing. Markdown emphasis around the number is tolerated for the same reason:
    a bolded figure is still the figure.
    """
    slot = WORD if textual else NUMBER
    escaped = re.escape(anchor).replace(r"\{\}", "\x00")
    escaped = re.sub(r"(\\\s|\s)+", r"\\s+", escaped)
    escaped = escaped.replace("\x00", r"[*_`]*" + slot + r"[*_`]*")
    return re.compile(escaped, re.IGNORECASE)


def strip_for_reverse(text: str) -> str:
    for pattern in (FENCE, CODE_SPAN, HTML_ATTR, URL, IDENTIFIER):
        text = pattern.sub(" ", text)
    return text


def check_document(path: Path, payload: dict) -> list[Finding]:
    """Both passes over one document."""
    name = str(path.relative_to(ROOT))
    if not path.exists():
        return [Finding(name, "missing", "listed as a checked document but not on disk")]
    text = path.read_text(encoding="utf-8")
    findings = _forward(name, text, payload)
    findings += _reverse(name, text, payload)
    return findings


def _forward(name: str, text: str, payload: dict) -> list[Finding]:
    """Every occurrence of every anchor must carry the measured value."""
    findings: list[Finding] = []
    textual = set(payload["textual"])
    for metric, anchors in payload["anchors"].items():
        expected = payload["rendered"][metric]
        for anchor in anchors:
            pattern = anchor_pattern(anchor, metric in textual)
            for match in pattern.finditer(text):
                found = match.group(1)
                ok = (
                    found.strip().lower() == expected.strip().lower()
                    if metric in textual
                    else normalise(found) == normalise(expected)
                )
                if not ok:
                    paragraph = _paragraph_around(text, match.start())
                    findings.append(
                        Finding(
                            name,
                            "stale figure",
                            f"{metric}: anchor {anchor!r} found {found!r}, code produces "
                            f"{expected!r}\n      in: {paragraph}",
                        )
                    )
    return findings


def _reverse(name: str, text: str, payload: dict) -> list[Finding]:
    """Any number in prose that no metric accounts for."""
    known = {normalise(v) for v in payload["rendered"].values()}
    known |= {normalise(k) for k in EXEMPT_NUMBERS}
    findings: list[Finding] = []
    for raw in re.findall(NUMBER, strip_for_reverse(text)):
        if normalise(raw) not in known:
            findings.append(
                Finding(name, "unguarded number", f"{raw!r} matches no metric and is not exempt")
            )
    return findings


def _unquoted_metrics(payload: dict) -> list[Finding]:
    """Report metrics no checked document quotes, so the spec cannot rot unnoticed.

    Run once over all the documents together rather than per document, because a
    figure belongs wherever it reads best and requiring every document to carry
    every metric would turn each one into the same table.
    """
    textual = set(payload["textual"])
    corpus = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in payload["checked_documents"]
        if (ROOT / relative).exists()
    )
    unused = [
        metric
        for metric, anchors in payload["anchors"].items()
        if not any(anchor_pattern(a, metric in textual).search(corpus) for a in anchors)
    ]
    return [
        Finding("(all documents)", "unquoted metric",
                f"{m} is measured but no document anchors it")
        for m in sorted(unused)
    ]


def _paragraph_around(text: str, index: int) -> str:
    start = text.rfind("\n\n", 0, index) + 2
    end = text.find("\n\n", index)
    end = len(text) if end == -1 else end
    return " ".join(text[start:end].split())[:200]


def check_exemptions_hide_nothing(payload: dict) -> list[Finding]:
    """An exemption that happens to equal a measured value would hide a stale figure."""
    measured = {normalise(v) for v in payload["rendered"].values()}
    clashes = sorted(k for k in EXEMPT_NUMBERS if normalise(k) in measured)
    return [
        Finding(
            "tools/check_numbers.py",
            "exemption hides a measurement",
            f"{c!r} is exempt but is also a value the code produces",
        )
        for c in clashes
    ]


def run(payload: dict) -> list[Finding]:
    findings = check_exemptions_hide_nothing(payload)
    for relative in payload["checked_documents"]:
        findings += check_document(ROOT / relative, payload)
    findings += _unquoted_metrics(payload)
    return findings


def prove(payload: dict) -> int:
    """Break one README figure on a copy and show the checker catching it."""
    readme = ROOT / "README.md"
    original = readme.read_text(encoding="utf-8")
    metric = "wrong_prefix_hits"
    expected = payload["rendered"][metric]
    anchor = by_name()[metric].anchors[0]
    pattern = anchor_pattern(anchor, False)
    match = pattern.search(original)
    if match is None:
        print(f"cannot prove the checker: the README does not use the anchor {anchor!r}")
        return 1
    broken = original[: match.start(1)] + str(int(expected) + 1) + original[match.end(1) :]
    try:
        readme.write_text(broken, encoding="utf-8")
        findings = check_document(readme, payload)
        stale = [f for f in findings if f.kind == "stale figure"]
        print(f"deliberately changed {metric} in the README from {expected} to {int(expected) + 1}")
        for finding in stale:
            print(f"  caught: {finding}")
        if not stale:
            print("  NOT CAUGHT. The checker does not work.")
            return 1
        return 0
    finally:
        readme.write_text(original, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default=str(ROOT / "docs" / "metrics.json"))
    parser.add_argument("--prove", action="store_true", help="watch the checker fail on purpose")
    args = parser.parse_args(argv)
    path = Path(args.metrics)
    if not path.exists():
        print(f"no {path}. Run tools/collect_metrics.py first.", file=sys.stderr)
        return 3
    payload = json.loads(path.read_text())
    if args.prove:
        return prove(payload)
    findings = run(payload)
    if findings:
        print(f"{len(findings)} problem(s):", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1
    print(
        f"checked {len(payload['metrics'])} metrics across "
        f"{len(payload['checked_documents'])} documents, all current"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
