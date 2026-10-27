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


@pytest.fixture
def unit_vectors():
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((12, 8)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


@pytest.fixture(scope="session")
def repo_root():
    return ROOT
