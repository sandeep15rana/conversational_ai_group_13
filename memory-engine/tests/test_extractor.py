import json

import pytest

from memory_engine.config import MemoryConfig
from memory_engine.exceptions import ExtractionError
from memory_engine.extractor import (
    HybridExtractor,
    LLMExtractor,
    RuleBasedExtractor,
    build_default_extractor,
    filter_candidates,
)
from memory_engine.schemas import ConversationTurn, MemoryCandidate, MemorySource, MemoryType


def user(text):
    return ConversationTurn(role="user", content=text)


def prefs_of(candidates):
    return {c.preference_key: c.preference_value for c in candidates if c.memory_type == MemoryType.PREFERENCE}


def facts_of(candidates):
    return [c.text for c in candidates if c.memory_type == MemoryType.FACT]


rules = RuleBasedExtractor()


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("I prefer bullet points.", {"format": "bullet_points"}),
        ("Please always give me a table", {"format": "table"}),
        ("Keep it short, not too long", {"detail_level": "concise"}),
        ("I want detailed answers with sources", {"detail_level": "detailed", "include_citations": "true"}),
        ("I don't want citations", {"include_citations": "false"}),
        ("From now on use JSON", {"format": "json"}),
    ],
)
def test_preference_rules(sentence, expected):
    assert prefs_of(rules.extract([user(sentence)])) == expected


def test_no_trigger_no_preference():
    assert prefs_of(rules.extract([user("Show NVIDIA revenue in a table")])) == {}  # one-off request, not a standing preference


def test_negated_format_without_alternative_is_ignored():
    assert prefs_of(rules.extract([user("Please don't use tables")])) == {}


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("My name is Alex and I work at Acme Corp.", ["User's name is Alex", "User works at Acme Corp"]),
        ("I am a senior data analyst", ["User is a senior data analyst"]),
        ("I'm based in Pune", ["User is based in Pune"]),
        ("I am interested in semiconductor stocks", ["User is interested in semiconductor stocks"]),
        ("I am a bit confused", []),
    ],
)
def test_fact_rules(sentence, expected):
    assert facts_of(rules.extract([user(sentence)])) == expected


def test_facts_only_from_user_turns():
    turns = [ConversationTurn(role="assistant", content="My name is Claude and I work at Anthropic.")]
    assert facts_of(rules.extract(turns)) == []


def test_conclusion_from_final_output():
    out = rules.extract([user("Compare NVIDIA and AMD growth")], "### Briefing\n- **NVIDIA** grew 36.28% from Q1 to Q3. AMD grew 52.17%.")
    conclusions = [c.text for c in out if c.memory_type == MemoryType.CONCLUSION]
    assert len(conclusions) == 1
    assert "Compare NVIDIA and AMD growth" in conclusions[0] and "36.28%" in conclusions[0]
    assert "**" not in conclusions[0] and "#" not in conclusions[0]


def test_conclusion_keeps_only_the_request():
    turn = user("Hello! I work at Acme Corp. I prefer tables. Compare NVIDIA and AMD growth.")
    out = rules.extract([turn], "NVIDIA grew 36.28% while AMD grew 52.17% over the period.")
    conclusion = [c.text for c in out if c.memory_type == MemoryType.CONCLUSION][0]
    assert conclusion.startswith("Past question: Compare NVIDIA and AMD growth.")
    assert "Acme Corp" not in conclusion and "tables" not in conclusion and "Hello" not in conclusion


def test_no_conclusion_for_tiny_output():
    assert not [c for c in rules.extract([user("hi")], "ok") if c.memory_type == MemoryType.CONCLUSION]


def test_filter_candidates_drops_low_confidence_and_batch_duplicates():
    cands = [
        MemoryCandidate(text="User likes tea", memory_type=MemoryType.FACT, confidence=0.9),
        MemoryCandidate(text="user likes tea.", memory_type=MemoryType.FACT, confidence=0.9),
        MemoryCandidate(text="User maybe likes golf", memory_type=MemoryType.FACT, confidence=0.3),
    ]
    assert [c.text for c in filter_candidates(cands, 0.6)] == ["User likes tea"]


# ---------------------------------------------------------------- LLM extractor (mocked, like Pooja's MockLLM)
class MockLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        assert "Conversation:" in prompt
        return type("Msg", (), {"content": self.content})()


def test_llm_extractor_parses_and_validates():
    payload = {"memories": [
        {"text": "User prefers tables", "memory_type": "preference", "confidence": 0.9,
         "preference_key": "format", "preference_value": "table"},
        {"text": "x", "memory_type": "fact"},  # invalid: too short -> skipped, not fatal
        {"text": "User works at Acme", "memory_type": "fact", "confidence": 0.8},
    ]}
    out = LLMExtractor(MockLLM("```json\n" + json.dumps(payload) + "\n```")).extract([user("hi")])
    assert [c.text for c in out] == ["User prefers tables", "User works at Acme"]
    assert all(c.source == MemorySource.LLM for c in out)


def test_llm_extractor_bad_json_raises():
    with pytest.raises(ExtractionError):
        LLMExtractor(MockLLM("not json")).extract([user("hi")])


def test_hybrid_falls_back_to_rules():
    hybrid = HybridExtractor(LLMExtractor(MockLLM("broken")), RuleBasedExtractor())
    out = hybrid.extract([user("I prefer bullet points")])
    assert prefs_of(out) == {"format": "bullet_points"} and hybrid.last_used == "rules"


def test_hybrid_uses_llm_when_it_works():
    hybrid = HybridExtractor(LLMExtractor(MockLLM('{"memories": []}')), RuleBasedExtractor())
    assert hybrid.extract([user("I prefer bullet points")]) == [] and hybrid.last_used == "llm"


def test_default_extractor_without_key_is_rules(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert build_default_extractor(MemoryConfig()).name == "rules"
