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
from semantic_cache_audit.embed.registry import PRETRAINED_NAME

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "experiments"

#: The embedder every headline figure is a fact about: a real, widely deployed
#: pretrained sentence encoder, read from its committed matrix.
HEADLINE_EMBEDDER = PRETRAINED_NAME

#: A member of the separation family used wherever a curve with shape is needed. Every
#: number taken from it is labelled with its name, which carries its noise level.
SHAPED_EMBEDDER = "synthetic@2.6"

#: The lexical embedders implemented in this repository, measured alongside the
#: pretrained one so E5 can show a threshold transferring between real families.
LEXICAL = ("bow", "lsa", "charngram")

#: Everything that is a genuine function of the text rather than an instrument.
REAL_EMBEDDERS = (PRETRAINED_NAME, *LEXICAL)


def reference_threshold(pairs, scores) -> float:
    """The threshold a team would pick: serve half the true paraphrases."""
    import numpy as np

    return float(np.median([s for p, s in zip(pairs, scores, strict=True) if p.same_answer]))


def corpus():
    return load_pairs()


def fixture(name: str):
    return load_fixture(name)


def write(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
