"""
Memory Engine configuration (Developer 3 - Rajarshi).

All tunable values live here. Every value can be overridden with an
environment variable (or a .env file) named MEMORY_<FIELD_NAME_IN_CAPS>,
for example MEMORY_TOP_K=8 or MEMORY_PERSIST_DIR=/data/memory.
Values are validated on load, so a bad setting fails fast with a clear error.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator

from .exceptions import MemoryValidationError

def _default_persist_dir() -> str:
    """memory-engine/.memory_store for a source/editable install (independent of the working
    directory); ~/.memory_engine/store for a regular install, so data never lands inside a venv."""
    package_dir = Path(__file__).resolve().parent
    if "site-packages" in package_dir.parts or "dist-packages" in package_dir.parts:
        return str(Path.home() / ".memory_engine" / "store")
    return str(package_dir.parents[1] / ".memory_store")


_DEFAULT_PERSIST_DIR = _default_persist_dir()


class MemoryConfig(BaseModel):
    """Validated settings for every memory component."""

    model_config = {"frozen": True}

    # Storage (ChromaDB)
    persist_dir: Optional[str] = Field(
        default=_DEFAULT_PERSIST_DIR,
        description="Folder for the ChromaDB files. None = in-memory only.",
    )
    collection_name: str = Field(default="long_term_memory", min_length=3, max_length=63)

    # Retrieval (RAG)
    top_k: int = Field(default=5, ge=1, le=50)
    fetch_k_multiplier: int = Field(default=4, ge=1, le=20)
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    recency_half_life_days: float = Field(default=30.0, gt=0.0)
    context_token_budget: int = Field(default=600, ge=50, le=8000)
    max_profile_preferences: int = Field(default=20, ge=0, le=200)
    max_profile_facts: int = Field(default=10, ge=0, le=100)

    # Writing / validation
    duplicate_threshold: float = Field(default=0.90, ge=0.5, le=1.0)
    max_memories_per_user: int = Field(default=500, ge=10, le=100_000)
    min_extraction_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    # Chunking (MiniLM reads at most 256 tokens; stay well below)
    chunk_max_tokens: int = Field(default=200, ge=32, le=256)
    chunk_overlap_sentences: int = Field(default=1, ge=0, le=5)

    # Short-term memory
    short_term_max_turns: int = Field(default=20, ge=2, le=500)
    short_term_ttl_seconds: int = Field(default=3600, ge=60)
    short_term_max_sessions: int = Field(default=1000, ge=1)
    recent_turns_in_context: int = Field(default=4, ge=0, le=50)

    # Embedding model. None = Chroma's default cache (~/.cache/chroma/onnx_models/all-MiniLM-L6-v2).
    # Set it to a folder containing onnx/model.onnx + onnx/tokenizer.json to run fully offline.
    embedding_model_dir: Optional[str] = None

    # Secret for the keyed hash (HMAC) of user ids. If unset, a random key is created once and
    # kept in <persist_dir>/.id_hash_key. Set it (e.g. from a secrets manager) so the key does
    # not sit next to the data. Changing it makes existing memories unreachable.
    id_hash_key: Optional[SecretStr] = None

    # Optional LLM extractor (used only when OPENAI_API_KEY is set)
    llm_model: str = Field(default="gpt-4o-mini", min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> "MemoryConfig":
        if self.duplicate_threshold <= self.min_similarity:
            raise ValueError("duplicate_threshold must be greater than min_similarity")
        return self

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True, **overrides: Any) -> "MemoryConfig":
        """Build a config from MEMORY_* environment variables plus explicit overrides."""
        if load_dotenv_file:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:  # python-dotenv is optional
                pass

        values: dict[str, Any] = {}
        for name in cls.model_fields:
            raw = os.getenv(f"MEMORY_{name.upper()}")
            if raw is not None and raw != "":
                values[name] = None if raw.lower() == "none" else raw
        values.update(overrides)

        try:
            return cls(**values)
        except ValidationError as exc:
            raise MemoryValidationError(f"Invalid memory configuration: {exc}") from exc
