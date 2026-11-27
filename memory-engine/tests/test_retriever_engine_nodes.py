from datetime import timedelta

import pytest

from memory_engine import engine as engine_module
from memory_engine.nodes import (
    build_supervisor_memory_block,
    memory_context_for_synthesis,
    memory_recall_node,
    memory_save_node,
)
from memory_engine.schemas import MemoryCandidate, MemoryType, utc_now


def fact(text):
    return MemoryCandidate(text=text, memory_type=MemoryType.FACT, confidence=0.9)


# ---------------------------------------------------------------- retriever
def test_preferences_always_loaded_even_if_unrelated(engine):
    engine.remember("u1", "User prefers format: table", MemoryType.PREFERENCE, "format", "table")
    engine.remember("u1", "User prefers include citations: true", MemoryType.PREFERENCE, "include_citations", "true")
    result = engine.recall("u1", "completely unrelated question about weather")
    assert result.preferences == {"format": "table", "include_citations": True}
    assert "User preferences:" in result.context_text


def test_retrieve_filters_by_min_similarity(engine):
    engine.long_term.add("u1", fact("User works at ZS Associates in Pune"))
    engine.long_term.add("u1", fact("Quantum chromodynamics lattice gauge"))
    hits = engine.retriever.retrieve("u1", "Where does the user work at ZS")
    texts = [h.record.text for h in hits]
    assert "User works at ZS Associates in Pune" in texts
    assert "Quantum chromodynamics lattice gauge" not in texts
    assert all(h.similarity >= engine.config.min_similarity for h in hits)


def test_retrieve_empty_query(engine):
    assert engine.retriever.retrieve("u1", "   ") == []


def test_recency_bonus_breaks_ties(engine):
    now = utc_now()
    fresh = engine.retriever._recency_bonus(now, now)
    old = engine.retriever._recency_bonus(now - timedelta(days=30), now)
    assert fresh == pytest.approx(engine.config.recency_weight)
    assert old == pytest.approx(engine.config.recency_weight / 2)  # half-life = 30 days


def test_mmr_prefers_diverse_results(embedder):
    from conftest import make_config
    from memory_engine.engine import MemoryEngine

    # High duplicate threshold so the near-copy is stored; lambda 0.5 weights diversity
    # equally with relevance, so MMR must skip the near-copy.
    eng = MemoryEngine(config=make_config(duplicate_threshold=0.99, mmr_lambda=0.5), embedder=embedder)
    for text in [
        "NVIDIA revenue grew strongly in Q3 data center",
        "NVIDIA revenue grew strongly in Q3 data center segment",
        "NVIDIA revenue grew strongly but TSMC packaging limits supply",
    ]:
        eng.long_term.add("u1", MemoryCandidate(text=text, memory_type=MemoryType.CONCLUSION, confidence=0.9))
    assert eng.long_term.count("u1") == 3
    hits = eng.retriever.retrieve("u1", "NVIDIA revenue grew strongly in Q3", top_k=2)
    assert len(hits) == 2
    assert any("TSMC" in h.record.text for h in hits)  # the near-copy lost to the diverse one


def test_context_respects_token_budget(embedder):
    from conftest import make_config
    from memory_engine.engine import MemoryEngine
    from memory_engine.extractor import RuleBasedExtractor

    eng = MemoryEngine(config=make_config(context_token_budget=60), embedder=embedder, extractor=RuleBasedExtractor())
    for i in range(20):
        eng.remember("u1", f"User prefers setting{i}: value number {i} for testing budgets", MemoryType.PREFERENCE, f"setting{i}", str(i))
    context = eng.recall("u1", "anything").context_text
    assert 0 < embedder.count_tokens(context) <= 60 + 15  # within budget (+ newline joins)


# ---------------------------------------------------------------- engine
def test_consolidate_extracts_and_dedupes(engine):
    engine.record_turn("u1", "s1", "user", "My name is Rajarshi. I prefer bullet points.")
    first = engine.consolidate("u1", "s1", final_output="NVIDIA grew 36.28% from Q1 to Q3 in data center revenue.")
    assert first.added == 3 and first.extractor == "rules"  # name, preference, conclusion
    # Same turn again -> already processed, nothing new.
    assert engine.consolidate("u1", "s1", final_output="...").total_written == 0
    # New session repeating the preference -> duplicate, not a new memory.
    engine.record_turn("u1", "s2", "user", "I prefer bullet points.")
    second = engine.consolidate("u1", "s2")
    assert second.duplicates == 1 and second.added == 0


def test_consolidate_without_conclusions(engine):
    engine.record_turn("u1", "s1", "user", "Compare chip makers")
    result = engine.consolidate("u1", "s1", final_output="A long enough final answer about chips.", include_conclusions=False)
    assert result.added == 0


def test_forget_user_and_end_session(engine):
    engine.remember("u1", "User works at ZS Associates")
    engine.record_turn("u1", "s1", "user", "hello")
    assert engine.forget_user("u1") == 1 and engine.list_memories("u1") == []
    engine.end_session("u1", "s1")
    assert engine.short_term.get_turns("u1::s1") == []


# ---------------------------------------------------------------- LangGraph nodes
def base_state(**kw):
    state = {"user_id": "u1", "session_id": "t1", "sanitized_query": "", "is_safe": True,
             "final_output": "", "is_hallucination_detected": False}
    state.update(kw)
    return state


def test_nodes_full_cycle(default_engine):
    s1 = base_state(sanitized_query="I am a senior data analyst. I prefer tables.")
    out = memory_recall_node(s1)
    assert out["recalled_context"] == "" and out["user_preferences"] == {}
    save = memory_save_node({**s1, "final_output": "NVIDIA grew 36.28% from Q1 to Q3 in data center revenue."})
    assert "added 3" in save["execution_trace"][-1]

    # Word-overlap query: the unit-test embedder is bag-of-words; real semantics are tested with MiniLM.
    s2 = base_state(session_id="t2", sanitized_query="NVIDIA grew data center revenue from Q1 to Q3?")
    out2 = memory_recall_node(s2)
    assert out2["user_preferences"] == {"format": "table"}
    assert "36.28%" in out2["recalled_context"]
    assert set(out2) == {"recalled_context", "user_preferences", "execution_trace"}  # only AgentState fields


def test_recall_includes_recent_turns_of_same_session(default_engine):
    memory_recall_node(base_state(sanitized_query="first question about AMD"))
    out = memory_recall_node(base_state(sanitized_query="and what about Intel?"))
    assert "Recent conversation" in out["recalled_context"] and "first question about AMD" in out["recalled_context"]


def test_unsafe_input_skips_memory(default_engine):
    out = memory_recall_node(base_state(sanitized_query="I prefer tables", is_safe=False))
    assert out["recalled_context"] is None and "skipped" in out["execution_trace"][-1]
    save = memory_save_node(base_state(is_safe=False, final_output="fallback"))
    assert "nothing saved" in save["execution_trace"][-1]
    assert default_engine.list_memories("u1") == []


def test_hallucinated_answer_not_memorized(default_engine):
    s = base_state(sanitized_query="I prefer tables")
    memory_recall_node(s)
    save = memory_save_node({**s, "final_output": "A suspicious answer that is long enough.", "is_hallucination_detected": True})
    types = {r.memory_type for r in default_engine.list_memories("u1")}
    assert types == {MemoryType.PREFERENCE}
    assert "not memorized" in save["execution_trace"][-1]


def test_nodes_fail_soft(monkeypatch):
    def boom():
        raise RuntimeError("chroma is down")

    monkeypatch.setattr("memory_engine.nodes.get_default_engine", boom)
    out = memory_recall_node(base_state(sanitized_query="hi"))
    assert out["recalled_context"] is None and out["user_preferences"] == {}
    assert "WARNING" in out["execution_trace"][-1]
    save = memory_save_node(base_state(final_output="answer"))
    assert "WARNING" in save["execution_trace"][-1]


def test_missing_user_id_skips_memory(default_engine):
    out = memory_recall_node({"sanitized_query": "hello", "is_safe": True})
    assert out["recalled_context"] is None and "No user_id" in out["execution_trace"][0]


def test_helpers_for_other_teams():
    state = {"recalled_context": "User preferences:\n- User prefers format: table\nKnown facts about the user:\n- User works at ZS"}
    assert memory_context_for_synthesis(state) == ["User prefers format: table", "User works at ZS"]
    block = build_supervisor_memory_block(state)
    assert block.startswith("### Long-term memory") and "User works at ZS" in block
    assert memory_context_for_synthesis({}) == [] and build_supervisor_memory_block({}) == ""


def test_default_engine_is_lazy_singleton(monkeypatch, engine):
    engine_module.set_default_engine(None)
    monkeypatch.setattr(engine_module, "MemoryEngine", lambda: engine)
    assert engine_module.get_default_engine() is engine is engine_module.get_default_engine()
    engine_module.set_default_engine(None)
