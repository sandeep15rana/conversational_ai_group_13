import pytest

from memory_engine.embeddings import HashingEmbedder
from memory_engine.exceptions import MemoryStoreError, MemoryValidationError
from memory_engine.long_term import LongTermMemory
from memory_engine.schemas import MemoryCandidate, MemoryType, WriteOutcome

from conftest import make_config


def fact(text, confidence=0.8):
    return MemoryCandidate(text=text, memory_type=MemoryType.FACT, confidence=confidence)


def pref(key, value, text=None):
    return MemoryCandidate(
        text=text or f"User prefers {key}: {value}",
        memory_type=MemoryType.PREFERENCE,
        preference_key=key,
        preference_value=value,
    )


# ---------------------------------------------------------------- no duplicates
def test_exact_duplicate_is_stored_once(store):
    assert store.add("u1", fact("User works at Acme Corp"))[0] == WriteOutcome.ADDED
    assert store.add("u1", fact("user works at   Acme corp."))[0] == WriteOutcome.DUPLICATE
    assert store.count("u1") == 1


def test_duplicate_refreshes_confidence(store):
    store.add("u1", fact("User works at Acme Corp", confidence=0.6))
    store.add("u1", fact("User works at Acme Corp", confidence=0.95))
    assert store.get_all("u1")[0].confidence == 0.95


def test_write_is_idempotent(store):
    for _ in range(3):
        store.add("u1", fact("User is based in Pune"))
    assert store.count("u1") == 1


def test_near_duplicate_updates_in_place(store):
    base = "User follows semiconductor stocks like NVIDIA AMD Intel TSMC and Qualcomm closely"
    store.add("u1", fact(base))
    created = store.get_all("u1")[0].created_at
    outcome, _, _ = store.add("u1", fact(base + " daily"))  # cosine ~0.95 with the bag-of-words embedder
    assert outcome == WriteOutcome.UPDATED
    records = store.get_all("u1")
    assert len(records) == 1 and records[0].text.endswith("daily")
    assert records[0].created_at == created  # history kept, text refreshed


def test_different_memories_are_both_kept(store):
    store.add("u1", fact("User works at Acme Corp"))
    store.add("u1", fact("User is based in Pune"))
    assert store.count("u1") == 2


def test_preference_conflict_newer_wins(store):
    store.add("u1", pref("format", "bullet_points"))
    outcome, _, _ = store.add("u1", pref("format", "table"))
    assert outcome == WriteOutcome.UPDATED
    prefs = store.get_by_type("u1", MemoryType.PREFERENCE)
    assert [(p.preference_key, p.preference_value) for p in prefs] == [("format", "table")]


# ---------------------------------------------------------------- partitioning
def test_users_are_isolated(store, embedder):
    store.add("alice", fact("Alice secret project is Falcon"))
    store.add("bob", fact("Bob likes tennis"))
    vector = embedder.embed(["Alice secret project is Falcon"])[0]
    bob_hits = store.search("bob", vector, n_results=10)
    assert all(h[0].user_id == store.storage_key("bob") for h in bob_hits)
    assert all("Falcon" not in h[0].text for h in bob_hits)
    assert store.count("alice") == 1 and store.count("bob") == 1


def test_same_text_for_two_users_is_two_records(store):
    store.add("alice", fact("User likes concise answers"))
    store.add("bob", fact("User likes concise answers"))
    assert store.count("alice") == 1 and store.count("bob") == 1


def test_invalid_user_rejected(store):
    with pytest.raises(MemoryValidationError):
        store.add("bad user!", fact("something"))
    with pytest.raises(MemoryValidationError):
        store.search("", [0.0] * 384, n_results=3)


# ---------------------------------------------------------------- safety
def test_pii_never_reaches_the_store(store):
    store.add("u1", fact("User email is john@corp.com and phone 555-123-4567"))
    text = store.get_all("u1")[0].text
    assert "john@corp.com" not in text and "555-123-4567" not in text
    assert "[REDACTED_EMAIL]" in text and "[REDACTED_PHONE]" in text


def test_cap_evicts_lowest_importance_first(embedder):
    store = LongTermMemory(make_config(max_memories_per_user=10), embedder)
    store.add("u1", pref("format", "table"))  # importance 1.0 * 0.8
    for i in range(12):
        store.add("u1", MemoryCandidate(text=f"Conclusion number {i} about topic {i * 7}",
                                        memory_type=MemoryType.CONCLUSION, confidence=0.7))
    assert store.count("u1") == 10
    assert store.get_by_type("u1", MemoryType.PREFERENCE)  # high-importance memory survived


# ---------------------------------------------------------------- chunking in the store
def test_long_memory_is_chunked_and_shares_parent(embedder):
    store = LongTermMemory(make_config(chunk_max_tokens=40), embedder)
    long_text = " ".join(f"Finding {i} says revenue moved by {i} percent." for i in range(25))
    outcome, parent_id, _ = store.add("u1", MemoryCandidate(text=long_text, memory_type=MemoryType.CONCLUSION))
    records = store.get_all("u1")
    assert outcome == WriteOutcome.ADDED and len(records) > 1
    assert {r.parent_id for r in records} == {parent_id}
    assert [r.chunk_index for r in records] == list(range(len(records)))
    assert all(embedder.count_tokens(r.text) <= 40 for r in records)
    assert store.count("u1") == 1  # one memory, many chunks
    assert store.delete_memory("u1", parent_id) == len(records) and store.count("u1") == 0


# ---------------------------------------------------------------- search & delete
def test_search_returns_similarity(store, embedder):
    store.add("u1", fact("User works at Acme Corp"))
    hits = store.search("u1", embedder.embed(["User works at Acme Corp"])[0], n_results=3)
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)
    assert len(hits[0][2]) == embedder.dimension


def test_search_on_empty_store(store, embedder):
    assert store.search("u1", embedder.embed(["anything"])[0], n_results=5) == []


def test_forget_user(store):
    store.add("u1", fact("User works at Acme Corp"))
    store.add("u1", pref("format", "table"))
    store.add("u2", fact("Other user fact here"))
    assert store.delete_user("u1") == 2
    assert store.count("u1") == 0 and store.count("u2") == 1


# ---------------------------------------------------------------- persistence & model guard
def test_memories_survive_restart(tmp_path, embedder):
    cfg = make_config(persist_dir=str(tmp_path / "store"))
    LongTermMemory(cfg, embedder).add("u1", fact("User is based in Pune"))
    reopened = LongTermMemory(cfg, embedder)
    assert reopened.count("u1") == 1


def test_mixing_embedding_models_is_blocked(config, embedder):
    LongTermMemory(config, embedder)

    class OtherModel(HashingEmbedder):
        name = "some-other-model"

    with pytest.raises(MemoryStoreError):
        LongTermMemory(config, OtherModel())
