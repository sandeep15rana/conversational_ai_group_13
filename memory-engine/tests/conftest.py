"""Shared pytest fixtures. Unit tests use the fast, offline HashingEmbedder and an in-memory ChromaDB."""

import uuid

import pytest

from memory_engine.config import MemoryConfig
from memory_engine.embeddings import HashingEmbedder
from memory_engine.engine import MemoryEngine, set_default_engine
from memory_engine.extractor import RuleBasedExtractor
from memory_engine.long_term import LongTermMemory


def make_config(**overrides) -> MemoryConfig:
    values = {
        "persist_dir": None,  # in-memory ChromaDB
        "collection_name": f"test_{uuid.uuid4().hex[:12]}",  # unique per test = isolated
    }
    values.update(overrides)
    return MemoryConfig(**values)


@pytest.fixture
def config() -> MemoryConfig:
    return make_config()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(config, embedder) -> LongTermMemory:
    return LongTermMemory(config, embedder)


@pytest.fixture
def engine(config, embedder) -> MemoryEngine:
    return MemoryEngine(config=config, embedder=embedder, extractor=RuleBasedExtractor())


@pytest.fixture
def default_engine(engine):
    """Installs `engine` as the shared engine used by the LangGraph nodes, then resets it."""
    set_default_engine(engine)
    yield engine
    set_default_engine(None)
