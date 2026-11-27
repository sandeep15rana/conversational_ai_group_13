import pytest

from memory_engine.config import MemoryConfig
from memory_engine.embeddings import HashingEmbedder, cosine_similarity
from memory_engine.exceptions import MemoryValidationError
from memory_engine.privacy import contains_pii, mask_pii
from memory_engine.schemas import (
    MemoryCandidate,
    MemoryRecord,
    MemoryType,
    content_hash,
    normalize_text,
    validate_user_id,
)


# ---------------------------------------------------------------- config
def test_config_defaults_are_valid():
    cfg = MemoryConfig()
    assert cfg.top_k == 5 and cfg.min_similarity == 0.35 and cfg.duplicate_threshold == 0.90
    assert cfg.chunk_max_tokens <= 256  # MiniLM limit


def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("MEMORY_TOP_K", "8")
    monkeypatch.setenv("MEMORY_PERSIST_DIR", "none")
    cfg = MemoryConfig.from_env(load_dotenv_file=False)
    assert cfg.top_k == 8 and cfg.persist_dir is None


def test_config_rejects_bad_values(monkeypatch):
    monkeypatch.setenv("MEMORY_MIN_SIMILARITY", "1.5")
    with pytest.raises(MemoryValidationError):
        MemoryConfig.from_env(load_dotenv_file=False)


def test_config_rejects_inconsistent_thresholds():
    with pytest.raises(MemoryValidationError):
        MemoryConfig.from_env(load_dotenv_file=False, min_similarity=0.95, duplicate_threshold=0.9)


def test_config_is_immutable():
    with pytest.raises(Exception):
        MemoryConfig().top_k = 99


# ---------------------------------------------------------------- schemas
@pytest.mark.parametrize("bad", ["", "   ", "a b", "x" * 129, "drop;table", None])
def test_invalid_user_ids_rejected(bad):
    with pytest.raises(ValueError):
        validate_user_id(bad)


def test_valid_user_id_is_trimmed():
    assert validate_user_id("  rajarshi.pal@zs  ") == "rajarshi.pal@zs"


def test_candidate_validation():
    with pytest.raises(Exception):
        MemoryCandidate(text="  ", memory_type=MemoryType.FACT)
    with pytest.raises(Exception):
        MemoryCandidate(text="valid text", memory_type="opinion")
    with pytest.raises(Exception):
        MemoryCandidate(text="valid text", memory_type=MemoryType.FACT, confidence=1.2)
    c = MemoryCandidate(text="  User   likes\ncharts ", memory_type=MemoryType.PREFERENCE, preference_key=" Chart Type ")
    assert c.text == "User likes charts" and c.preference_key == "chart_type"


def test_normalize_and_hash_are_stable():
    assert normalize_text("  User likes   Tables. ") == "user likes tables"
    assert content_hash("u1", "User likes tables.") == content_hash("u1", "user likes   TABLES")
    assert content_hash("u1", "same") != content_hash("u2", "same")  # partitioned by user


def test_record_metadata_round_trip():
    rec = MemoryRecord(
        memory_id="m1", user_id="u1", text="hello world", memory_type=MemoryType.FACT,
        confidence=0.8, importance=0.64, content_hash="h", parent_id="m1",
    )
    meta = rec.to_metadata()
    assert all(v is not None for v in meta.values())
    assert MemoryRecord.from_chroma("m1", "hello world", meta) == rec


# ---------------------------------------------------------------- privacy
@pytest.mark.parametrize(
    "text, placeholder",
    [
        ("mail me at john.doe@corp.com", "[REDACTED_EMAIL]"),
        ("call 555-123-4567 now", "[REDACTED_PHONE]"),
        ("my number is +91 98765 43210", "[REDACTED_PHONE]"),
        ("key sk-abcdefghijklmnopqrstuvwx", "[REDACTED_API_KEY]"),
        ("aws AKIAABCDEFGHIJKLMNOP", "[REDACTED_API_KEY]"),
        ("password: hunter2", "[REDACTED_SECRET]"),
        ("ssn 123-45-6789", "[REDACTED_SSN]"),
        ("card 4111 1111 1111 1111", "[REDACTED_CARD]"),
    ],
)
def test_pii_is_masked(text, placeholder):
    assert placeholder in mask_pii(text)
    assert contains_pii(text)


def test_mask_is_idempotent_and_keeps_clean_text():
    clean = "NVIDIA grew 36.28% in Q3 2024"
    assert mask_pii(clean) == clean
    once = mask_pii("a@b.com")
    assert mask_pii(once) == once


# ---------------------------------------------------------------- embeddings helpers
def test_cosine_similarity_formula():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity([1], [1, 2])


def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder()
    a, b = emb.embed(["user likes tables", "user likes tables"])
    assert a == b and len(a) == 384
    assert sum(x * x for x in a) == pytest.approx(1.0)
