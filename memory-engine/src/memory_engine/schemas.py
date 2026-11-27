"""
Pydantic data contracts for the Memory Engine (Developer 3 - Rajarshi).

These models validate every memory before it is stored or returned, so bad
data is rejected at the boundary instead of corrupting the vector store.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.@\-]{1,128}$")
_WS = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s.!?,;:]+$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(text: str) -> str:
    """Canonical form used for exact-duplicate detection: lower case, single spaces, no trailing punctuation."""
    return _TRAILING_PUNCT.sub("", _WS.sub(" ", text.strip().lower()))


def content_hash(user_id: str, text: str) -> str:
    """Stable SHA-256 of user + normalized text. Same fact for the same user = same hash."""
    return hashlib.sha256(f"{user_id}|{normalize_text(text)}".encode("utf-8")).hexdigest()


def validate_user_id(user_id: str) -> str:
    """User ids partition the store, so they must be present and safe."""
    if not isinstance(user_id, str):
        raise ValueError("user_id must be a string")
    cleaned = user_id.strip()
    if not USER_ID_PATTERN.match(cleaned):
        raise ValueError(
            "user_id must be 1-128 characters of letters, digits, '_', '.', '@' or '-'"
        )
    return cleaned


class MemoryType(str, Enum):
    """What kind of memory this is. Drives importance and how it is shown to the supervisor."""

    PREFERENCE = "preference"  # how the user likes answers (format, detail, citations...)
    FACT = "fact"  # stable facts about the user (name, role, company...)
    CONCLUSION = "conclusion"  # outcome of a past conversation


# Higher weight = kept longer when a user's store is full.
TYPE_IMPORTANCE: dict[MemoryType, float] = {
    MemoryType.PREFERENCE: 1.0,
    MemoryType.FACT: 0.8,
    MemoryType.CONCLUSION: 0.6,
}


class MemorySource(str, Enum):
    RULE = "rule"
    LLM = "llm"
    MANUAL = "manual"


class ConversationTurn(BaseModel):
    """One message in the short-term buffer."""

    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=20_000)
    timestamp: datetime = Field(default_factory=utc_now)
    seq: int = Field(default=0, ge=0, description="Position in the session; strictly increasing, unlike timestamps")


class MemoryCandidate(BaseModel):
    """Something the extractor thinks is worth remembering. Not stored yet."""

    # Long text is allowed here; the store splits it into <= 256-token chunks.
    text: str = Field(min_length=3, max_length=20_000)
    memory_type: MemoryType
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source: MemorySource = MemorySource.RULE
    preference_key: Optional[str] = Field(default=None, max_length=64)
    preference_value: Optional[str] = Field(default=None, max_length=256)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        value = _WS.sub(" ", value).strip()
        if len(value) < 3:
            raise ValueError("memory text must contain at least 3 non-space characters")
        return value

    @field_validator("preference_key")
    @classmethod
    def _clean_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip().lower().replace(" ", "_")
        return value or None


class MemoryRecord(BaseModel):
    """A memory (or one chunk of a long memory) as stored in ChromaDB."""

    memory_id: str = Field(min_length=1)
    user_id: str
    text: str = Field(min_length=1, max_length=4000)
    memory_type: MemoryType
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    source: MemorySource = MemorySource.RULE
    content_hash: str
    parent_id: str
    chunk_index: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=1, ge=1)
    session_id: Optional[str] = None
    preference_key: Optional[str] = None
    preference_value: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("user_id")
    @classmethod
    def _check_user(cls, value: str) -> str:
        return validate_user_id(value)

    def to_metadata(self) -> dict[str, Any]:
        """ChromaDB metadata only accepts str/int/float/bool, never None."""
        meta: dict[str, Any] = {
            "user_id": self.user_id,
            "memory_type": self.memory_type.value,
            "confidence": float(self.confidence),
            "importance": float(self.importance),
            "source": self.source.value,
            "content_hash": self.content_hash,
            "parent_id": self.parent_id,
            "chunk_index": int(self.chunk_index),
            "chunk_count": int(self.chunk_count),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "updated_ts": float(self.updated_at.timestamp()),
        }
        if self.session_id:
            meta["session_id"] = self.session_id
        if self.preference_key:
            meta["preference_key"] = self.preference_key
        if self.preference_value is not None:
            meta["preference_value"] = self.preference_value
        return meta

    @classmethod
    def from_chroma(cls, memory_id: str, document: str, metadata: dict[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=memory_id,
            user_id=metadata["user_id"],
            text=document,
            memory_type=MemoryType(metadata["memory_type"]),
            confidence=metadata.get("confidence", 0.8),
            importance=metadata.get("importance", 0.5),
            source=MemorySource(metadata.get("source", MemorySource.RULE.value)),
            content_hash=metadata["content_hash"],
            parent_id=metadata.get("parent_id", memory_id),
            chunk_index=metadata.get("chunk_index", 0),
            chunk_count=metadata.get("chunk_count", 1),
            session_id=metadata.get("session_id"),
            preference_key=metadata.get("preference_key"),
            preference_value=metadata.get("preference_value"),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
        )


class RetrievedMemory(BaseModel):
    """A memory returned by semantic search, with its scores."""

    record: MemoryRecord
    similarity: float = Field(description="Cosine similarity to the query, -1..1")
    score: float = Field(description="Final ranking score: similarity + recency bonus")


class RecallResult(BaseModel):
    """Everything the recall step hands to the rest of the pipeline."""

    context_text: str = ""
    preferences: dict[str, Any] = Field(default_factory=dict)
    memories: list[RetrievedMemory] = Field(default_factory=list)
    profile_facts: list[MemoryRecord] = Field(default_factory=list)
    recent_turns: list[ConversationTurn] = Field(default_factory=list)


class WriteOutcome(str, Enum):
    ADDED = "added"
    UPDATED = "updated"
    DUPLICATE = "duplicate"


class SaveResult(BaseModel):
    """Summary of one consolidation (save) run."""

    added: int = 0
    updated: int = 0
    duplicates: int = 0
    rejected: int = 0
    evicted: int = 0
    extractor: str = "none"

    @property
    def total_written(self) -> int:
        return self.added + self.updated
