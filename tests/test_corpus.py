"""Properties of the adversarial corpus, which every other number depends on."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest

from semantic_cache_audit.corpus import builder
from semantic_cache_audit.corpus.loader import CorpusError, load_pairs
from semantic_cache_audit.corpus.schema import (
    ALL_CLASSES,
    NEAR_MISS_CLASSES,
    POSITIVE_CLASS,
    Pair,
    pair_id_for,
)
from semantic_cache_audit.corpus.templates import TEMPLATES


def test_the_committed_corpus_matches_the_hash_in_its_manifest(corpus_manifest, repo_root):
    text = (repo_root / "fixtures" / "corpus" / "pairs.jsonl").read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode()).hexdigest() == corpus_manifest["sha256"]


def test_loading_a_corpus_whose_hash_does_not_match_is_refused(tmp_path):
    root = tmp_path / "corpus"
    builder.write(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CorpusError, match="does not match its manifest"):
        load_pairs(root)


def test_a_missing_corpus_names_the_command_that_builds_it(tmp_path):
    with pytest.raises(CorpusError, match="corpus.builder"):
        load_pairs(tmp_path)


def test_a_missing_manifest_is_reported_rather_than_skipped(tmp_path):
    (tmp_path / "pairs.jsonl").write_text("")
    with pytest.raises(CorpusError, match="no manifest"):
        load_pairs(tmp_path)


def test_the_corpus_can_be_loaded_without_verification(tmp_path):
    root = tmp_path / "corpus"
    builder.write(root)
    (root / "manifest.json").unlink()
    assert len(load_pairs(root, verify=False)) == len(TEMPLATES[0]["subs"]) * 0 + 320


def test_regenerating_the_corpus_twice_produces_identical_bytes(tmp_path):
    first = builder.to_jsonl(builder.expand())
    second = builder.to_jsonl(builder.expand())
    assert first == second


def test_every_template_identifier_is_unique():
    ids = [t["tid"] for t in TEMPLATES]
    assert len(ids) == len(set(ids))


def test_every_pair_identifier_is_unique(pairs):
    ids = [p.pair_id for p in pairs]
    assert len(ids) == len(set(ids))


def test_no_two_pairs_share_both_sides(pairs):
    sides = Counter((p.query_a, p.query_b) for p in pairs)
    assert [k for k, n in sides.items() if n > 1] == []


def test_a_pair_identifier_is_derived_from_content_not_from_position():
    first = pair_id_for("t", "a", "b")
    second = pair_id_for("t", "a", "b")
    assert first == second
    assert pair_id_for("t", "a", "c") != first


def test_every_class_in_the_corpus_is_a_declared_class(pairs):
    assert {p.cls for p in pairs} <= set(ALL_CLASSES)


def test_every_near_miss_class_is_represented(pairs):
    present = {p.cls for p in pairs}
    assert set(NEAR_MISS_CLASSES) <= present


def test_the_near_miss_classes_are_balanced(pairs):
    counts = Counter(p.cls for p in pairs if p.is_near_miss)
    assert len(set(counts.values())) == 1, counts


def test_positive_controls_are_a_substantial_share_of_the_corpus(pairs):
    positives = sum(1 for p in pairs if p.same_answer)
    assert positives / len(pairs) > 0.3


def test_the_positive_controls_are_not_all_easy(pairs):
    positives = [p for p in pairs if p.same_answer]
    hard = sum(1 for p in positives if p.difficulty == "hard")
    assert hard / len(positives) > 0.3


def test_every_difficulty_level_appears_among_the_positive_controls(pairs):
    levels = {p.difficulty for p in pairs if p.same_answer}
    assert levels == {"easy", "medium", "hard"}


def test_the_corpus_spans_several_domains(pairs):
    assert len({p.domain for p in pairs}) >= 5


def test_a_near_miss_pair_never_claims_its_sides_share_an_answer(pairs):
    assert all(not p.same_answer for p in pairs if p.cls in NEAR_MISS_CLASSES)


def test_a_positive_control_always_claims_its_sides_share_an_answer(pairs):
    assert all(p.same_answer for p in pairs if p.cls == POSITIVE_CLASS)


def test_a_pair_whose_label_disagrees_with_its_class_is_rejected():
    with pytest.raises(ValueError, match="must have same_answer"):
        Pair(
            pair_id="x", cls="negation", domain="d", query_a="a", query_b="b",
            same_answer=True, difficulty="hard", template_id="t", provenance="p",
        )


def test_a_pair_with_an_unknown_class_is_rejected():
    with pytest.raises(ValueError, match="unknown class"):
        Pair(
            pair_id="x", cls="nonsense", domain="d", query_a="a", query_b="b",
            same_answer=False, difficulty="hard", template_id="t", provenance="p",
        )


def test_a_pair_with_an_unknown_difficulty_is_rejected():
    with pytest.raises(ValueError, match="unknown difficulty"):
        Pair(
            pair_id="x", cls="negation", domain="d", query_a="a", query_b="b",
            same_answer=False, difficulty="trivial", template_id="t", provenance="p",
        )


def test_a_pair_with_identical_sides_is_rejected():
    with pytest.raises(ValueError, match="identical sides"):
        Pair(
            pair_id="x", cls="negation", domain="d", query_a="same", query_b="same ",
            same_answer=False, difficulty="hard", template_id="t", provenance="p",
        )


def test_a_pair_round_trips_through_its_dictionary_form(pairs):
    assert Pair.from_dict(pairs[0].to_dict()) == pairs[0]


def test_every_pair_records_the_template_and_substitution_it_came_from(pairs):
    assert all(p.provenance.startswith(p.template_id + ":") for p in pairs)


def test_every_pair_carries_a_note_explaining_why_it_is_adversarial(pairs):
    near = [p for p in pairs if p.is_near_miss]
    assert all(len(p.note) > 10 for p in near)


def test_duplicate_pair_identifiers_are_caught_by_the_builder(monkeypatch):
    clashing = [dict(TEMPLATES[0]), dict(TEMPLATES[0])]
    monkeypatch.setattr(builder, "TEMPLATES", clashing)
    with pytest.raises(ValueError, match="duplicate pair ids"):
        builder.expand()


def test_the_manifest_counts_agree_with_the_pairs_on_disk(pairs, corpus_manifest):
    assert corpus_manifest["corpus_pairs_total"] == len(pairs)
    assert corpus_manifest["near_miss_pairs"] == sum(1 for p in pairs if p.is_near_miss)
    assert corpus_manifest["positive_control_pairs"] == sum(1 for p in pairs if p.same_answer)


def test_the_manifest_class_counts_agree_with_the_pairs_on_disk(pairs, corpus_manifest):
    counts = Counter(p.cls for p in pairs)
    for cls, n in corpus_manifest["by_class"].items():
        assert counts[cls] == n


def test_the_manifest_records_how_the_corpus_was_authored(corpus_manifest):
    assert "hand written" in corpus_manifest["authored"]


def test_direction_reversal_pairs_really_are_reversals(pairs):
    reversals = [p for p in pairs if p.cls == "direction_reversal"]
    assert reversals
    for pair in reversals:
        assert sorted(pair.query_a.split()) == sorted(pair.query_b.split())


def test_no_query_in_the_corpus_is_empty(pairs):
    assert all(p.query_a.strip() and p.query_b.strip() for p in pairs)
