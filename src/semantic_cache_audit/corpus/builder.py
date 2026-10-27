"""Expands the authored template families into the committed corpus file.

Expansion is deterministic and total: every substitution listed on a family produces
exactly one pair, in the order written. There is no sampling and no random seed here,
because a corpus that changes between runs cannot anchor a published number.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .schema import ALL_CLASSES, POSITIVE_CLASS, Pair, pair_id_for
from .templates import TEMPLATES


def expand() -> list[Pair]:
    """Build every pair from every template family, in declaration order."""
    pairs: list[Pair] = []
    for family in TEMPLATES:
        tid = family["tid"]
        cls = family["cls"]
        for sub in family["subs"]:
            query_a = family["a"].format(**sub)
            query_b = family["b"].format(**sub)
            provenance = f"{tid}:" + ",".join(f"{k}={v}" for k, v in sorted(sub.items()))
            pairs.append(
                Pair(
                    pair_id=pair_id_for(tid, query_a, query_b),
                    cls=cls,
                    domain=family["domain"],
                    query_a=query_a,
                    query_b=query_b,
                    same_answer=cls == POSITIVE_CLASS,
                    difficulty=family["difficulty"],
                    template_id=tid,
                    provenance=provenance,
                    note=family.get("note", ""),
                )
            )
    _assert_unique(pairs)
    return pairs


def _assert_unique(pairs: list[Pair]) -> None:
    ids = Counter(p.pair_id for p in pairs)
    collisions = [pid for pid, n in ids.items() if n > 1]
    if collisions:
        raise ValueError(f"duplicate pair ids: {collisions[:5]}")
    sides = Counter((p.query_a, p.query_b) for p in pairs)
    repeats = [k for k, n in sides.items() if n > 1]
    if repeats:
        raise ValueError(f"duplicate query pairs: {repeats[:3]}")


def manifest_for(pairs: list[Pair], jsonl_text: str) -> dict:
    """The provenance record committed beside the corpus.

    It carries the content hash of the corpus file so a reviewer can tell at a glance
    whether the committed vectors were computed over this exact corpus.
    """
    by_class = Counter(p.cls for p in pairs)
    by_domain = Counter(p.domain for p in pairs)
    by_difficulty = Counter(p.difficulty for p in pairs)
    near_miss = sum(1 for p in pairs if p.is_near_miss)
    return {
        "corpus_pairs_total": len(pairs),
        "templates_total": len(TEMPLATES),
        "near_miss_pairs": near_miss,
        "positive_control_pairs": len(pairs) - near_miss,
        "by_class": {c: by_class.get(c, 0) for c in ALL_CLASSES},
        "by_domain": dict(sorted(by_domain.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "sha256": hashlib.sha256(jsonl_text.encode()).hexdigest(),
        "generated_by": "python -m semantic_cache_audit.corpus.builder",
        "authored": (
            "every family hand written to sit in the high similarity, different answer "
            "region; substitutions enumerated by hand, not sampled"
        ),
    }


def to_jsonl(pairs: list[Pair]) -> str:
    return "".join(json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in pairs)


def write(root: Path) -> dict:
    """Write pairs.jsonl and manifest.json under ``root`` and return the manifest."""
    root.mkdir(parents=True, exist_ok=True)
    pairs = expand()
    text = to_jsonl(pairs)
    (root / "pairs.jsonl").write_text(text, encoding="utf-8")
    manifest = manifest_for(pairs, text)
    manifest["captured_utc"] = datetime.now(UTC).strftime("%Y-%m-%d")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":  # pragma: no cover - thin entry point
    here = Path(__file__).resolve().parents[3] / "fixtures" / "corpus"
    print(json.dumps(write(here), indent=2))
