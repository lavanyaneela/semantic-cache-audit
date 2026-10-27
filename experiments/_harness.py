"""Shared loading and output for the experiments.

Every experiment writes one JSON file under docs/experiments and prints one table.
`tools/collect_metrics.py` reads the JSON; nothing reads the table. Keeping the two
separate means a formatting change can never move a published number.
"""

from __future__ import annotations

import json
from pathlib import Path

from semantic_cache_audit.corpus.loader import load_pairs
from semantic_cache_audit.embed.fixtures import load_fixture

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "experiments"

#: The real embedder the README quotes. Lexical, implemented in this repository, and
#: the most defensible of the three because character n-grams are the only one of them
#: that is order sensitive across word boundaries.
HEADLINE_EMBEDDER = "charngram"

#: A member of the separation family used wherever a curve with shape is needed. Every
#: number taken from it is labelled with its name, which carries its noise level.
SHAPED_EMBEDDER = "synthetic@2.6"

#: The three real embedders, measured and published together.
REAL_EMBEDDERS = ("bow", "lsa", "charngram")


def corpus():
    return load_pairs()


def fixture(name: str):
    return load_fixture(name)


def write(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
