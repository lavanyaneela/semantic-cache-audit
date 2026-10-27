"""The record shape every corpus pair carries, and the vocabulary of classes.

A pair is two queries plus a judgement about whether one answer serves both. That
judgement, not the similarity, is the ground truth the audit is measured against, so it
is stored explicitly with its provenance rather than derived at load time.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

#: Classes where one cached answer does NOT serve the other query.
NEAR_MISS_CLASSES: tuple[str, ...] = (
    "direction_reversal",
    "negation",
    "entity_swap",
    "numeric_change",
    "scope_change",
)

#: The positive control. Without it the false-hit curve has no denominator worth plotting.
POSITIVE_CLASS = "true_paraphrase"

ALL_CLASSES: tuple[str, ...] = (*NEAR_MISS_CLASSES, POSITIVE_CLASS)

#: Difficulty is declared by the author, not measured. It records intent: an "easy"
#: paraphrase shares most of its words, a "hard" one deliberately does not. It exists so
#: a reviewer can check that the positive controls are not all trivially easy.
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")


@dataclass(frozen=True, slots=True)
class Pair:
    """One query pair with its ground truth label and where it came from."""

    pair_id: str
    cls: str
    domain: str
    query_a: str
    query_b: str
    same_answer: bool
    difficulty: str
    template_id: str
    provenance: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.cls not in ALL_CLASSES:
            raise ValueError(f"unknown class {self.cls!r}")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty {self.difficulty!r}")
        expected_same = self.cls == POSITIVE_CLASS
        if self.same_answer != expected_same:
            raise ValueError(
                f"class {self.cls!r} must have same_answer={expected_same}, "
                f"got {self.same_answer}"
            )
        if self.query_a.strip() == self.query_b.strip():
            raise ValueError(f"pair {self.pair_id} has identical sides")

    @property
    def is_near_miss(self) -> bool:
        return not self.same_answer

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> Pair:
        return cls(**raw)


def pair_id_for(template_id: str, query_a: str, query_b: str) -> str:
    """A stable id derived from content, so regenerating the corpus does not renumber it.

    An index-based id looks tidy and silently reassigns every downstream reference the
    first time a template is inserted in the middle of the list.
    """
    digest = hashlib.sha256(f"{template_id}|{query_a}|{query_b}".encode()).hexdigest()
    return f"{template_id}-{digest[:8]}"
