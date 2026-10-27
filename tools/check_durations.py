"""Fails the build when a duration is published as a result anywhere in the repository.

Receipts rules five and six. A millisecond is a fact about the machine that measured
it, so this repository publishes ratios and counts instead, and no prose anchors to a
figure derived from a timing. Claims about speed live in tests, which assert an
inequality rather than quoting a number.

The promise is "anywhere in this repository", so the sweep covers every tracked text
file rather than the seven checked documents. Guarding only the documents is how the
first version of this idea passed while a stale millisecond sat in a module
docstring.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: A number immediately followed by a time unit. Thousands separators must be real
#: groups of three: the first version allowed any run of digits and commas, so the
#: expression `Math.min(110, ms * 1000)` in a pacing script read as "110 ms" and
#: failed the build. Broad otherwise, because the allow list below should carry the
#: exceptions with their reasons rather than the pattern carrying them silently.
DURATION = re.compile(
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(ms|milliseconds?|microseconds?|us|seconds?|"
    r"secs?|minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE,
)

TEXT_SUFFIXES = {".md", ".py", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".json"}

#: Paths where a duration is configuration or test data rather than a published
#: result. Each entry states why.
ALLOWED_PATHS: dict[str, str] = {
    "src/semantic_cache_audit/config.py": "ttl_seconds is a configured input, not a result",
    "src/semantic_cache_audit/cache/clock.py": "the clock module, which is about time",
    "src/semantic_cache_audit/cache/backends.py": "ttl bookkeeping and a connection timeout",
    "src/semantic_cache_audit/cache/semantic_cache.py": "ttl bookkeeping",
    "src/semantic_cache_audit/audit/stream.py": "simulated arrival spacing and delays",
    "src/semantic_cache_audit/embed/sentence_encoder.py": "a token limit, not a duration",
    "tools/check_durations.py": "this file, which contains the pattern itself",
    "tests/test_cache.py": "simulated ttl values in an injected clock",
    "tests/test_stream.py": "simulated ttl values in an injected clock",
    "tests/test_receipts.py": "holds the example strings this pattern is tested against",
    "experiments/e3_eviction_sweep.py": "the ttl grid, a configured input",
    "docs/experiments/e3_eviction_sweep.json": "the ttl grid, recorded as an input",
    "docs/metrics.json": "no duration is a metric; the file is checked by the test below",
}

#: Phrases that make a duration a stated input rather than a measurement. A line
#: matching one of these in a document is allowed, because it is describing a knob.
INPUT_PHRASES = (
    "ttl",
    "time to live",
    "configured",
    "arrival",
    "simulated",
    "injected clock",
    "token",
)


@dataclass
class Hit:
    path: str
    line_number: int
    line: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line_number}: {self.line.strip()[:140]}"


def tracked_files() -> list[Path]:
    """Every file git knows about, tracked or newly added but not yet committed.

    `git ls-files` alone lists only tracked files, which means a new file is
    invisible to this check until it is committed. That is not a theoretical gap: the
    receipts suite was written, run green, and committed, and then failed on the next
    fresh clone because the file holding the pattern's own test strings had become
    visible to the pattern. `--others` closes it, so the check sees what a clone
    would see.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [ROOT / line for line in dict.fromkeys(out.stdout.splitlines()) if line]


def scan() -> list[Hit]:
    hits: list[Hit] = []
    for path in tracked_files():
        relative = str(path.relative_to(ROOT))
        if path.suffix not in TEXT_SUFFIXES or relative in ALLOWED_PATHS:
            continue
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not DURATION.search(line):
                continue
            if any(phrase in line.lower() for phrase in INPUT_PHRASES):
                continue
            hits.append(Hit(relative, number, line))
    return hits


def main() -> int:
    hits = scan()
    if hits:
        print(f"{len(hits)} duration(s) published as a result:", file=sys.stderr)
        for hit in hits:
            print(f"  {hit}", file=sys.stderr)
        print(
            "\nPublish a ratio or a count instead, and assert the inequality in a test.",
            file=sys.stderr,
        )
        return 1
    print(f"no published durations across {len(tracked_files())} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
