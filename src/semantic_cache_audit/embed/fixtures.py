"""Writes and reads the committed embedding matrices.

Every number in this repository is computed from a matrix on disk rather than from a
model in memory, for two reasons. A reviewer with no model and no network reproduces
every figure, and swapping in real weights is a regeneration of one file rather than a
change to any component that consumes it.

The manifest records which embedder produced each matrix, its dimension, the sha256 of
the corpus it was computed over, and the date it was captured. A matrix whose corpus
hash does not match the corpus on disk is refused rather than used.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..corpus.loader import load_manifest, load_pairs
from ..corpus.schema import Pair
from .base import zero_rows
from .registry import build, corpus_documents


def default_root() -> Path:
    return Path(__file__).resolve().parents[3] / "fixtures" / "embeddings"


class FixtureError(RuntimeError):
    """A requested embedding fixture is missing or does not match the corpus."""


def _safe_name(name: str) -> str:
    return name.replace(":", "__").replace("/", "__").replace("@", "_at_")


def write_fixture(name: str, pairs: list[Pair] | None = None, root: Path | None = None) -> dict:
    """Compute and commit the matrix for one embedder."""
    root = root or default_root()
    root.mkdir(parents=True, exist_ok=True)
    pairs = pairs if pairs is not None else load_pairs()
    docs = corpus_documents(pairs)
    embedder = build(name, pairs)
    matrix = embedder.encode(docs)
    if matrix.shape[0] != len(docs):
        raise FixtureError(f"{name}: expected {len(docs)} rows, got {matrix.shape[0]}")
    path = root / f"{_safe_name(name)}.npz"
    np.savez_compressed(path, matrix=matrix.astype(np.float32), texts=np.array(docs, dtype=object))
    record = {
        "embedder": name,
        "dim": int(matrix.shape[1]),
        "rows": int(matrix.shape[0]),
        "zero_rows_before_normalisation": zero_rows(matrix),
        "corpus_sha256": load_manifest()["sha256"],
        # Today's date, unless an identical fixture was already captured. Stamping
        # the current date unconditionally made `make embeddings` dirty the manifest
        # on any day but the one it was built on, so the repository was byte
        # reproducible for exactly twenty four hours.
        "captured_utc": _previous_capture_date(root, name, matrix)
        or datetime.now(UTC).strftime("%Y-%m-%d"),
        "describes": embedder.describe(),
        "file": path.name,
    }
    _update_manifest(root, record)
    return record


def _previous_capture_date(root: Path, name: str, matrix: np.ndarray) -> str | None:
    """The date already on record, when the matrix being written is unchanged."""
    path = root / "manifest.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))["fixtures"].get(name)
    if record is None:
        return None
    existing = root / record["file"]
    if not existing.exists():
        return None
    with np.load(existing, allow_pickle=True) as data:
        if data["matrix"].shape != matrix.shape:
            return None
        if not np.array_equal(data["matrix"], matrix.astype(np.float32)):
            return None
    return str(record["captured_utc"])


def _update_manifest(root: Path, record: dict) -> None:
    path = root / "manifest.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"fixtures": {}}
    current["fixtures"][record["embedder"]] = record
    current["note"] = (
        "Each matrix is the embedding of every distinct corpus query, in the order "
        "given by corpus_documents. corpus_sha256 pins the corpus the vectors were "
        "computed over; a mismatch is refused on load rather than used."
    )
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_fixture(
    name: str, root: Path | None = None, *, verify: bool = True
) -> tuple[list[str], np.ndarray]:
    """Return the query texts and their matrix, refusing a matrix built on another corpus."""
    root = root or default_root()
    path = root / f"{_safe_name(name)}.npz"
    if not path.exists():
        raise FixtureError(
            f"no embedding fixture for {name!r} at {path}. Run: make embeddings"
        )
    if verify:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        record = manifest["fixtures"].get(name)
        if record is None:
            raise FixtureError(f"{name!r} has a matrix but no manifest entry")
        expected = load_manifest()["sha256"]
        if record["corpus_sha256"] != expected:
            raise FixtureError(
                f"{name!r} was computed over a different corpus: fixture "
                f"{record['corpus_sha256'][:12]} vs corpus {expected[:12]}. Run: make embeddings"
            )
    with np.load(path, allow_pickle=True) as data:
        return [str(t) for t in data["texts"]], data["matrix"].astype(np.float32)


def available(root: Path | None = None) -> list[str]:
    root = root or default_root()
    path = root / "manifest.json"
    if not path.exists():
        return []
    return sorted(json.loads(path.read_text(encoding="utf-8"))["fixtures"])
