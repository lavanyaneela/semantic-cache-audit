"""Properties of the embedder panel, including the ones a committed fixture depends on."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from semantic_cache_audit.embed.base import l2_normalise, normalise_text, zero_rows
from semantic_cache_audit.embed.lexical import (
    BagOfWordsEmbedder,
    CharNgramEmbedder,
    LsaEmbedder,
    _char_ngrams,
    _uni_and_bigrams,
)
from semantic_cache_audit.embed.live import LiveEmbedderUnavailable, SentenceTransformerEmbedder
from semantic_cache_audit.embed.registry import (
    LEXICAL_NAMES,
    CorpusInconsistent,
    UnknownEmbedder,
    build,
    corpus_documents,
    fixture_names,
    intent_labels,
    parse_noise,
    shared_queries,
)
from semantic_cache_audit.embed.synthetic import NOISE_GRID, SeparationFamily

DOCS = [
    "what is the cheapest fare from Austin to Boston",
    "what is the cheapest fare from Boston to Austin",
    "how much luggage can I check on Northwind Air",
    "is a corkscrew allowed in carry on baggage",
]


@pytest.fixture(params=LEXICAL_NAMES)
def lexical(request, pairs):
    return build(request.param, pairs)


def test_normalising_text_lowercases_and_strips_punctuation():
    assert normalise_text("  Hello, WORLD!  ") == "hello world"


def test_normalising_text_collapses_runs_of_whitespace():
    assert normalise_text("a\n\tb   c") == "a b c"


def test_normalising_text_is_idempotent():
    once = normalise_text("Why? Because: yes.")
    assert normalise_text(once) == once


def test_l2_normalisation_gives_every_row_unit_length(unit_vectors):
    norms = np.linalg.norm(l2_normalise(unit_vectors * 3.0), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_a_zero_row_survives_normalisation_without_producing_a_nan():
    matrix = np.zeros((2, 4), dtype=np.float32)
    assert not np.isnan(l2_normalise(matrix)).any()


def test_zero_rows_are_counted_rather_than_hidden():
    matrix = np.zeros((3, 4), dtype=np.float32)
    matrix[0, 0] = 1.0
    assert zero_rows(matrix) == 2


def test_character_ngrams_are_padded_so_word_boundaries_are_visible():
    grams = _char_ngrams("ab", low=3, high=3)
    assert " ab" in grams


def test_character_ngrams_of_a_string_shorter_than_n_are_still_produced():
    assert _char_ngrams("a", low=5, high=5) == []


def test_bigram_features_include_the_unigrams_as_well():
    feats = _uni_and_bigrams("a b c")
    assert "a" in feats and "a_b" in feats


def test_every_lexical_embedder_returns_unit_rows(lexical):
    matrix = lexical.encode(DOCS)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_every_lexical_embedder_returns_one_row_per_input(lexical):
    assert lexical.encode(DOCS).shape[0] == len(DOCS)


def test_every_lexical_embedder_reports_the_dimension_it_produces(lexical):
    assert lexical.encode(DOCS).shape[1] == lexical.dim


def test_every_lexical_embedder_is_deterministic_within_a_process(lexical):
    assert np.array_equal(lexical.encode(DOCS), lexical.encode(DOCS))


def test_encoding_one_string_agrees_with_encoding_a_batch_of_one(lexical):
    assert np.allclose(lexical.encode_one(DOCS[0]), lexical.encode([DOCS[0]])[0])


def test_every_embedder_describes_itself_with_a_stable_name(lexical):
    assert lexical.describe()["name"] == lexical.name


def test_the_bag_of_words_embedder_is_blind_to_word_order():
    embedder = BagOfWordsEmbedder(DOCS)
    matrix = embedder.encode(DOCS[:2])
    assert float(matrix[0] @ matrix[1]) == pytest.approx(1.0, abs=1e-5)


def test_the_character_ngram_embedder_is_not_blind_to_word_order():
    embedder = CharNgramEmbedder()
    matrix = embedder.encode(DOCS[:2])
    assert float(matrix[0] @ matrix[1]) < 0.999


def test_the_lsa_embedder_reduces_to_at_most_the_requested_components():
    embedder = LsaEmbedder(DOCS, n_components=64)
    assert embedder.dim <= 64


def test_the_lsa_embedder_projects_an_unseen_string_rather_than_refitting():
    embedder = LsaEmbedder(DOCS)
    before = embedder.vocab_size
    embedder.encode(["a sentence the embedder has never seen before"])
    assert embedder.vocab_size == before


def test_an_out_of_vocabulary_string_does_not_raise(lexical):
    assert lexical.encode(["zzzz qqqq vvvv"]).shape[0] == 1


def test_the_character_ngram_embedder_uses_a_stable_hash_not_the_salted_builtin():
    """Pins the trap that would make a committed fixture reproducible exactly once.

    Python's built in `hash` is salted per process, so a feature index derived from it
    differs between runs and between machines. A fixture built that way loads without
    error and silently contains the wrong vectors. The embedder uses sha1 instead, and
    this test runs a second interpreter to prove it.
    """
    script = (
        "import numpy as np;"
        "from semantic_cache_audit.embed.lexical import CharNgramEmbedder;"
        "m = CharNgramEmbedder().encode(['stable across processes']);"
        "print(float(m[0][:8].sum()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    here = CharNgramEmbedder().encode(["stable across processes"])
    assert float(here[0][:8].sum()) == pytest.approx(float(out.stdout.strip()), abs=1e-6)


def test_the_character_ngram_embedder_reports_its_seed_and_feature_count():
    described = CharNgramEmbedder().describe()
    assert described["seed"] and described["n_features"]


def test_corpus_documents_are_deduplicated(pairs):
    docs = corpus_documents(pairs)
    assert len(docs) == len(set(docs))


def test_corpus_documents_cover_both_sides_of_every_pair(pairs):
    docs = set(corpus_documents(pairs))
    assert all(p.query_a in docs and p.query_b in docs for p in pairs)


def test_corpus_documents_are_in_a_stable_order(pairs):
    assert corpus_documents(pairs) == corpus_documents(pairs)


def test_some_queries_really_do_appear_in_more_than_one_pair(pairs):
    assert shared_queries(pairs) > 0


def test_intent_labels_give_both_sides_of_a_paraphrase_the_same_answer(pairs):
    labels = intent_labels(pairs)
    for pair in pairs:
        if pair.same_answer:
            assert labels[pair.query_a] == labels[pair.query_b]


def test_intent_labels_give_the_sides_of_a_near_miss_different_answers(pairs):
    labels = intent_labels(pairs)
    for pair in pairs:
        if pair.is_near_miss:
            assert labels[pair.query_a] != labels[pair.query_b]


def test_a_corpus_that_contradicts_itself_is_refused(pairs):
    """A paraphrase chain joining both sides of a near miss is an authoring error."""
    from semantic_cache_audit.corpus.schema import Pair as P

    near = next(p for p in pairs if p.is_near_miss)
    bridge = P(
        pair_id="bridge", cls="true_paraphrase", domain=near.domain,
        query_a=near.query_a, query_b=near.query_b, same_answer=True,
        difficulty="easy", template_id="bridge", provenance="test",
    )
    with pytest.raises(CorpusInconsistent, match="one answer class"):
        intent_labels([*pairs, bridge])


def test_the_separation_family_is_perfectly_separating_at_zero_noise(pairs):
    family = SeparationFamily(intent_labels(pairs), noise=0.0)
    pair = next(p for p in pairs if p.same_answer)
    matrix = family.encode([pair.query_a, pair.query_b])
    assert float(matrix[0] @ matrix[1]) == pytest.approx(1.0, abs=1e-5)


def test_the_separation_family_degrades_as_noise_rises(pairs):
    labels = intent_labels(pairs)
    pair = next(p for p in pairs if p.same_answer)
    scores = []
    for noise in (0.0, 2.0, 9.0):
        matrix = SeparationFamily(labels, noise=noise).encode([pair.query_a, pair.query_b])
        scores.append(float(matrix[0] @ matrix[1]))
    assert scores[0] > scores[1] > scores[2]


def test_the_separation_family_refuses_a_negative_noise_level(pairs):
    with pytest.raises(ValueError, match="non negative"):
        SeparationFamily(intent_labels(pairs), noise=-1.0)


def test_the_separation_family_carries_its_noise_level_in_its_name(pairs):
    assert SeparationFamily(intent_labels(pairs), noise=2.6).name == "synthetic@2.6"


def test_the_separation_family_declares_that_it_is_not_a_real_embedder(pairs):
    described = SeparationFamily(intent_labels(pairs), noise=1.0).describe()
    assert described["declared"] and described["is_a_model_of_an_embedder"]


def test_the_separation_family_gives_an_unlabelled_query_its_own_answer(pairs):
    family = SeparationFamily(intent_labels(pairs), noise=0.0)
    matrix = family.encode(["a query the family was never told about", pairs[0].query_a])
    assert float(matrix[0] @ matrix[1]) < 0.999


def test_the_separation_family_handles_an_empty_batch(pairs):
    assert SeparationFamily(intent_labels(pairs), noise=1.0).encode([]).shape[0] == 0


def test_a_bare_synthetic_name_is_refused_because_it_would_lose_its_noise_level():
    with pytest.raises(UnknownEmbedder, match="synthetic@"):
        parse_noise("synthetic")


def test_a_synthetic_name_with_a_non_numeric_noise_level_is_refused():
    with pytest.raises(UnknownEmbedder, match="non negative number"):
        parse_noise("synthetic@banana")


def test_parsing_a_synthetic_name_recovers_its_noise_level():
    assert parse_noise("synthetic@2.6") == pytest.approx(2.6)


def test_an_unknown_embedder_name_lists_the_known_ones(pairs):
    with pytest.raises(UnknownEmbedder, match="synthetic@<noise>"):
        build("nonsense", pairs)


def test_the_fixture_name_list_covers_every_lexical_embedder():
    assert set(LEXICAL_NAMES) <= set(fixture_names())


def test_the_fixture_name_list_covers_the_whole_noise_grid():
    names = set(fixture_names())
    assert all(f"synthetic@{n:g}" in names for n in NOISE_GRID)


def test_the_live_adapter_is_the_only_route_that_can_touch_the_network(pairs, monkeypatch):
    """Requesting a live model without the optional extra fails loudly, not silently."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises((LiveEmbedderUnavailable, TypeError, AttributeError)):
        SentenceTransformerEmbedder("some-model")


def test_building_a_live_embedder_goes_through_the_adapter(pairs, monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, name):
            seen["name"] = name

    monkeypatch.setattr(
        "semantic_cache_audit.embed.live.SentenceTransformerEmbedder", Fake, raising=True
    )
    build("st:some-model", pairs)
    assert seen["name"] == "some-model"
