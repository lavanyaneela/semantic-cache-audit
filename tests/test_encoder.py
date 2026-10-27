"""Properties of the pretrained encoder, skipped when its weights are not on disk.

The committed matrix is what every published number is computed from, so `make
verify` never needs these. They run when a developer has the weights cached, and they
are the only thing that checks the pooling is the pooling the model card specifies.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_cache_audit.embed.registry import PRETRAINED_NAME
from semantic_cache_audit.embed.sentence_encoder import (
    DEFAULT_MODEL,
    REQUIRED_FILES,
    EncoderUnavailable,
    OnnxSentenceEncoder,
)


def _weights_present() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return all(
        isinstance(try_to_load_from_cache(DEFAULT_MODEL, f), str) for f in REQUIRED_FILES
    )


needs_weights = pytest.mark.skipif(
    not _weights_present(),
    reason="pretrained weights are not cached locally; make verify does not need them",
)


def test_the_pretrained_name_points_at_the_declared_model():
    assert PRETRAINED_NAME.removeprefix("st:") == DEFAULT_MODEL


def test_missing_weights_raise_rather_than_fall_back(tmp_path):
    with pytest.raises(EncoderUnavailable, match="missing"):
        OnnxSentenceEncoder(DEFAULT_MODEL, local_dir=tmp_path)


@needs_weights
def test_the_encoder_reproduces_the_committed_fixture_exactly(pairs):
    """The one test that proves the committed matrix is what it says it is."""
    from semantic_cache_audit.embed.fixtures import load_fixture
    from semantic_cache_audit.embed.registry import corpus_documents

    texts, committed = load_fixture(PRETRAINED_NAME)
    assert texts == corpus_documents(pairs)
    recomputed = OnnxSentenceEncoder(DEFAULT_MODEL).encode(texts)
    assert np.allclose(recomputed, committed, atol=1e-4)


@needs_weights
def test_the_encoder_returns_unit_vectors():
    matrix = OnnxSentenceEncoder(DEFAULT_MODEL).encode(["a", "b c"])
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


@needs_weights
def test_the_encoder_produces_the_published_dimension():
    assert OnnxSentenceEncoder(DEFAULT_MODEL).dim == 384


@needs_weights
def test_batching_does_not_change_the_vectors():
    encoder = OnnxSentenceEncoder(DEFAULT_MODEL)
    texts = [f"sentence number {i}" for i in range(9)]
    assert np.allclose(encoder.encode(texts, batch_size=2), encoder.encode(texts, batch_size=64),
                       atol=1e-5)


@needs_weights
def test_padding_does_not_leak_into_the_pooled_vector():
    """Mean pooling must weight by the attention mask, not by padded length."""
    encoder = OnnxSentenceEncoder(DEFAULT_MODEL)
    alone = encoder.encode(["short"])
    with_long_neighbour = encoder.encode(["short", "a considerably longer sentence " * 8])
    assert np.allclose(alone[0], with_long_neighbour[0], atol=1e-4)


@needs_weights
def test_the_encoder_handles_an_empty_batch():
    assert OnnxSentenceEncoder(DEFAULT_MODEL).encode([]).shape[0] == 0


@needs_weights
def test_the_encoder_describes_itself_as_pretrained():
    assert OnnxSentenceEncoder(DEFAULT_MODEL).describe()["pretrained"] is True
