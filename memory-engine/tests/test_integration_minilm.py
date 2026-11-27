"""
Integration tests with the REAL all-MiniLM-L6-v2 model and ChromaDB.
Skipped automatically if the model cannot be loaded (e.g. offline without a cached copy).
Run only these:  pytest -m integration
"""

import sys
from pathlib import Path

import pytest

from memory_engine.embeddings import MiniLMEmbedder
from memory_engine.engine import MemoryEngine, set_default_engine
from memory_engine.exceptions import EmbeddingError
from memory_engine.extractor import RuleBasedExtractor
from memory_engine.nodes import memory_recall_node, memory_save_node
from memory_engine.schemas import MemoryCandidate, MemoryType

from conftest import make_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def minilm():
    embedder = MiniLMEmbedder()
    try:
        embedder.embed(["warm up"])
    except EmbeddingError as exc:
        pytest.skip(f"MiniLM model not available: {exc}")
    return embedder


@pytest.fixture
def real_engine(minilm):
    return MemoryEngine(config=make_config(), embedder=minilm, extractor=RuleBasedExtractor())


def test_embedding_shape_and_semantics(minilm):
    a, b, c = minilm.embed(["I prefer bullet points", "Give me answers as bullet lists", "NVIDIA revenue grew"])
    assert len(a) == 384
    dot = lambda x, y: sum(i * j for i, j in zip(x, y))
    assert dot(a, b) > 0.4 > dot(a, c)  # paraphrase close, unrelated far


def test_real_token_counter(minilm):
    assert minilm.count_tokens("hello world") == 2
    assert minilm.count_tokens("x " * 400) == 400  # no truncation when counting


def test_chunks_fit_model_window(real_engine, minilm):
    long_text = " ".join(f"Observation {i}: semiconductor demand shifted by {i} percent this quarter." for i in range(60))
    real_engine.long_term.add("u1", MemoryCandidate(text=long_text, memory_type=MemoryType.CONCLUSION))
    records = real_engine.list_memories("u1")
    assert len(records) > 1
    assert all(minilm.count_tokens(r.text) <= real_engine.config.chunk_max_tokens for r in records)


def test_semantic_recall_without_shared_words(real_engine):
    real_engine.remember("u1", "User is based in Pune, India")
    real_engine.remember("u1", "User works at ZS Associates as a data analyst")
    hits = real_engine.retriever.retrieve("u1", "Where does the user live?")
    assert hits and hits[0].record.text == "User is based in Pune, India"


def test_paraphrased_preference_conflict_resolved(real_engine):
    real_engine.record_turn("u1", "s1", "user", "I prefer bullet points.")
    real_engine.consolidate("u1", "s1")
    real_engine.record_turn("u1", "s2", "user", "From now on, please always give me a table.")
    real_engine.consolidate("u1", "s2")
    assert real_engine.recall("u1", "anything").preferences == {"format": "table"}


def test_end_to_end_through_nodes(real_engine):
    set_default_engine(real_engine)
    try:
        s1 = {"user_id": "rajarshi", "session_id": "t1", "is_safe": True,
              "sanitized_query": "I am a senior equity analyst. I prefer concise answers with citations."}
        memory_recall_node(s1)
        memory_save_node({**s1, "final_output": "NVIDIA data center revenue rose 36.28% from Q1 to Q3, driven by AI demand.",
                          "is_hallucination_detected": False})

        s2 = {"user_id": "rajarshi", "session_id": "t2", "is_safe": True,
              "sanitized_query": "How quickly has NVIDIA's AI server business expanded?"}
        out = memory_recall_node(s2)
        assert out["user_preferences"] == {"detail_level": "concise", "include_citations": True}
        assert "36.28%" in out["recalled_context"]

        other = memory_recall_node({**s2, "user_id": "someone_else"})
        assert other["recalled_context"] == ""  # isolation with the real stack
    finally:
        set_default_engine(None)


def test_retrieval_quality(minilm):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    import retrieval_eval

    metrics = retrieval_eval.run_eval(retrieval_eval.build_engine())
    assert metrics["hit@3"] >= 0.85, metrics
    assert metrics["mrr"] >= 0.75, metrics
