import pytest

from memory_engine.chunking import chunk_text, split_sentences
from memory_engine.exceptions import MemoryValidationError
from memory_engine.short_term import ShortTermMemory


def words(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------- chunking
def test_short_text_is_one_chunk():
    assert chunk_text("A short memory.", words, max_tokens=50) == ["A short memory."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", words) == []


def test_split_sentences_handles_bullets_and_paragraphs():
    parts = split_sentences("First one. Second one!\n\nThird para\n- bullet a\n- bullet b")
    assert parts == ["First one.", "Second one!", "Third para", "- bullet a", "- bullet b"]


def test_long_text_respects_limit_and_overlaps():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(30)]
    chunks = chunk_text(" ".join(sentences), words, max_tokens=30, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(words(c) <= 30 for c in chunks)
    for left, right in zip(chunks, chunks[1:]):  # last sentence of a chunk starts the next one
        last_sentence = left.split(". ")[-1].rstrip(".")
        assert right.startswith(last_sentence)
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)  # nothing lost


def test_no_overlap_option():
    text = " ".join(f"Line {i} is here." for i in range(20))
    chunks = chunk_text(text, words, max_tokens=12, overlap_sentences=0)
    assert sum(words(c) for c in chunks) == words(text)


def test_oversized_sentence_is_split_on_words():
    text = " ".join(["word"] * 100)
    chunks = chunk_text(text, words, max_tokens=25)
    assert all(words(c) <= 25 for c in chunks) and sum(words(c) for c in chunks) == 100


def test_invalid_chunk_arguments():
    with pytest.raises(ValueError):
        chunk_text("x", words, max_tokens=0)
    with pytest.raises(ValueError):
        chunk_text("x", words, overlap_sentences=-1)


# ---------------------------------------------------------------- short-term memory
class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_turns_are_kept_per_session():
    stm = ShortTermMemory()
    stm.add_turn("s1", "user", "hello from s1")
    stm.add_turn("s2", "user", "hello from s2")
    assert [t.content for t in stm.get_turns("s1")] == ["hello from s1"]
    assert [t.content for t in stm.get_turns("s2")] == ["hello from s2"]
    assert stm.get_turns("unknown") == []


def test_buffer_keeps_only_last_n_turns():
    stm = ShortTermMemory(max_turns=3)
    for i in range(5):
        stm.add_turn("s", "user", f"turn {i}")
    assert [t.content for t in stm.get_turns("s")] == ["turn 2", "turn 3", "turn 4"]
    assert [t.content for t in stm.get_turns("s", last_n=2)] == ["turn 3", "turn 4"]
    assert stm.get_turns("s", last_n=0) == []


def test_sessions_expire_after_ttl():
    clock = FakeClock()
    stm = ShortTermMemory(ttl_seconds=60, clock=clock)
    stm.add_turn("s", "user", "hi")
    clock.now = 59
    assert stm.get_turns("s")
    clock.now = 59 + 61
    assert stm.get_turns("s") == []


def test_lru_session_eviction():
    stm = ShortTermMemory(max_sessions=2)
    stm.add_turn("a", "user", "1")
    stm.add_turn("b", "user", "2")
    stm.get_turns("a")  # touch a
    stm.add_turn("c", "user", "3")  # evicts b (least recently used)
    assert stm.get_turns("b") == [] and stm.get_turns("a") and stm.session_count() == 2


def test_pii_masked_in_buffer_and_format_recent():
    stm = ShortTermMemory()
    stm.add_turn("s", "user", "reach me at a.b@c.com")
    stm.add_turn("s", "assistant", "x" * 400)
    text = stm.format_recent("s", last_n=2)
    assert "[REDACTED_EMAIL]" in text and text.startswith("User:")
    assert text.endswith("...")  # long turns are trimmed


def test_scratchpad():
    stm = ShortTermMemory()
    stm.set_scratchpad("s", "plan", ["a", "b"])
    assert stm.get_scratchpad("s", "plan") == ["a", "b"]
    assert stm.get_scratchpad("s") == {"plan": ["a", "b"]}
    assert stm.get_scratchpad("other", "plan", default=1) == 1
    with pytest.raises(MemoryValidationError):
        stm.set_scratchpad("s", " ", 1)


def test_validation_errors():
    stm = ShortTermMemory()
    with pytest.raises(MemoryValidationError):
        stm.add_turn("", "user", "x")
    with pytest.raises(MemoryValidationError):
        stm.add_turn("s", "robot", "x")
    with pytest.raises(MemoryValidationError):
        stm.add_turn("s", "user", "")
    with pytest.raises(MemoryValidationError):
        ShortTermMemory(max_turns=0)


def test_clear_session():
    stm = ShortTermMemory()
    stm.add_turn("s", "user", "x")
    stm.clear_session("s")
    assert stm.get_turns("s") == []
