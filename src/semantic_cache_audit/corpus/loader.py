"""Reads the committed corpus, and refuses to read a corpus that does not match its manifest.

A silently mismatched fixture is the failure mode that makes every published number
wrong at once, so the hash check is on the read path rather than in a test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import Pair


def default_root() -> Path:
    return Path(__file__).resolve().parents[3] / "fixtures" / "corpus"


class CorpusError(RuntimeError):
    """The corpus on disk is missing, unreadable, or does not match its manifest."""


def load_pairs(root: Path | None = None, *, verify: bool = True) -> list[Pair]:
    root = root or default_root()
    jsonl = root / "pairs.jsonl"
    if not jsonl.exists():
        raise CorpusError(
            f"no corpus at {jsonl}. Run: python -m semantic_cache_audit.corpus.builder"
        )
    text = jsonl.read_text(encoding="utf-8")
    if verify:
        manifest = load_manifest(root)
        actual = hashlib.sha256(text.encode()).hexdigest()
        if actual != manifest["sha256"]:
            raise CorpusError(
                "corpus does not match its manifest: "
                f"file sha256 {actual[:12]} vs manifest {manifest['sha256'][:12]}"
            )
    return [Pair.from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def load_manifest(root: Path | None = None) -> dict:
    root = root or default_root()
    path = root / "manifest.json"
    if not path.exists():
        raise CorpusError(f"no manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
