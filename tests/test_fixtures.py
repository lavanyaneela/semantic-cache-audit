"""Properties of the committed embedding matrices."""

from __future__ import annotations

import json

import numpy as np
import pytest

from semantic_cache_audit.embed.fixtures import (
    FixtureError,
    available,
    default_root,
    load_fixture,
    write_fixture,
)
from semantic_cache_audit.embed.registry import PRETRAINED_NAME, fixture_names


@pytest.fixture(params=fixture_names())
def every_fixture(request):
    return request.param, *load_fixture(request.param)


def test_every_declared_embedder_has_a_committed_fixture():
    assert set(fixture_names()) <= set(available())


def test_every_fixture_has_one_row_per_distinct_corpus_query(every_fixture, pairs):
    _, texts, matrix = every_fixture
    distinct = {p.query_a for p in pairs} | {p.query_b for p in pairs}
    assert matrix.shape[0] == len(texts) == len(distinct)


def test_every_fixture_row_is_a_unit_vector(every_fixture):
    _, _, matrix = every_fixture
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4)


def test_every_fixture_is_float32(every_fixture):
    _, _, matrix = every_fixture
    assert matrix.dtype == np.float32


def test_every_fixture_text_list_is_free_of_duplicates(every_fixture):
    _, texts, _ = every_fixture
    assert len(texts) == len(set(texts))


def test_every_fixture_contains_no_nan(every_fixture):
    _, _, matrix = every_fixture
    assert not np.isnan(matrix).any()


def test_the_pretrained_fixture_records_the_model_it_came_from():
    manifest = json.loads((default_root() / "manifest.json").read_text())
    record = manifest["fixtures"][PRETRAINED_NAME]
    assert record["describes"]["pretrained"] is True
    assert record["describes"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_the_pretrained_fixture_records_a_capture_date():
    manifest = json.loads((default_root() / "manifest.json").read_text())
    assert manifest["fixtures"][PRETRAINED_NAME]["captured_utc"].count("-") == 2


def test_every_fixture_records_the_corpus_hash_it_was_computed_over(corpus_manifest):
    manifest = json.loads((default_root() / "manifest.json").read_text())
    for record in manifest["fixtures"].values():
        assert record["corpus_sha256"] == corpus_manifest["sha256"]


def test_a_fixture_built_over_a_different_corpus_is_refused(tmp_path, pairs):
    write_fixture("charngram", pairs, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["fixtures"]["charngram"]["corpus_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FixtureError, match="different corpus"):
        load_fixture("charngram", tmp_path)


def test_a_fixture_with_a_matrix_but_no_manifest_entry_is_refused(tmp_path, pairs):
    write_fixture("charngram", pairs, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["fixtures"] = {}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FixtureError, match="no manifest entry"):
        load_fixture("charngram", tmp_path)


def test_a_missing_fixture_names_the_command_that_builds_it(tmp_path):
    with pytest.raises(FixtureError, match="make embeddings"):
        load_fixture("charngram", tmp_path)


def test_a_fixture_can_be_loaded_without_verification(tmp_path, pairs):
    write_fixture("charngram", pairs, tmp_path)
    (tmp_path / "manifest.json").unlink()
    texts, matrix = load_fixture("charngram", tmp_path, verify=False)
    assert matrix.shape[0] == len(texts)


def test_a_fixture_name_with_a_colon_is_written_to_a_safe_filename(tmp_path):
    manifest = json.loads((default_root() / "manifest.json").read_text())
    name = manifest["fixtures"][PRETRAINED_NAME]["file"]
    assert ":" not in name and "/" not in name


def test_listing_available_fixtures_on_an_empty_directory_returns_nothing(tmp_path):
    assert available(tmp_path) == []


def test_regenerating_a_fixture_reproduces_it_exactly(tmp_path, pairs, charngram_fixture):
    write_fixture("charngram", pairs, tmp_path)
    _, regenerated = load_fixture("charngram", tmp_path, verify=False)
    _, committed = charngram_fixture
    assert np.allclose(regenerated, committed, atol=1e-6)
