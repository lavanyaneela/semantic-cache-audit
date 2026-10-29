"""Shared fixtures. Loaded once per session because the corpus does not change."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from semantic_cache_audit.corpus.loader import load_manifest, load_pairs
from semantic_cache_audit.embed.fixtures import load_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def pairs():
    return load_pairs()


@pytest.fixture(scope="session")
def corpus_manifest():
    return load_manifest()


@pytest.fixture(scope="session")
def charngram_fixture():
    return load_fixture("charngram")


@pytest.fixture(scope="session")
def shaped_fixture():
    return load_fixture("synthetic@2.6")


@pytest.fixture(scope="session")
def oracle_fixture():
    return load_fixture("synthetic@0")


@pytest.fixture(autouse=True)
def isolate_shared_backends():
    """Give every test a clean hot layer and payload store.

    `fakeredis` and `mongomock` hand out fresh state per construction, so against the
    in process doubles the suite is isolated by accident. Against the real servers
    the optional CI job uses, one process shares one Redis and one MongoDB, and tests
    that assert a count saw whatever earlier tests had left behind. Flushing the
    default namespace before and after each test makes the suite say the same thing
    in both configurations, which is the only reason the real backend job is worth
    running at all.
    """
    from semantic_cache_audit.cache.backends import HotLayer, PayloadStore

    HotLayer().flush()
    PayloadStore().flush()
    yield
    HotLayer().flush()
    PayloadStore().flush()


@pytest.fixture
def unit_vectors():
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((12, 8)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


@pytest.fixture(scope="session")
def repo_root():
    return ROOT
