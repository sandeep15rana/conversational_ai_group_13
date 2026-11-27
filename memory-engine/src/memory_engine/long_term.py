"""
Long-term episodic memory on ChromaDB (Developer 3 - Rajarshi).

One collection holds every user's memories; each record carries a pseudonymous
user key (HMAC-SHA256 of the user id, never the raw id) in its metadata and EVERY
read/write filters on it (tenant partitioning), so one user can never read
another user's memories. Search is exact within that partition.

Write rules, in order:
  1. PII is masked (text, preference value, session id); text that looks like a
     prompt injection is rejected.
  2. Exact duplicate (same normalized text, same user) -> no new record, just refresh it.
  3. Preference with the same key (e.g. "format") -> the newer one replaces the old (conflict resolution).
  4. Near duplicate (cosine >= duplicate_threshold, same type; keyed preferences excluded)
     -> existing record updated in place.
  5. Otherwise insert. Long text is chunked; chunks share a parent_id.
  6. If the user exceeds max_memories_per_user, the lowest-importance, oldest memories are evicted.

Record ids are derived from the content hash, so retrying a write is idempotent.
"""

from __future__ import annotations

import logging
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import ValidationError

from .chunking import chunk_text
from .config import MemoryConfig
from .embeddings import Embedder, Vector
from .exceptions import MemoryStoreError, MemoryValidationError
from .privacy import looks_like_injection, mask_pii, pseudonymize_id
from .schemas import (
    TYPE_IMPORTANCE,
    MemoryCandidate,
    MemoryRecord,
    MemoryType,
    WriteOutcome,
    content_hash,
    utc_now,
    validate_user_id,
)

logger = logging.getLogger(__name__)

_PROCESS_ID_KEY = secrets.token_bytes(32)
_CONCLUSION_SPLIT = " | Conclusion:"


def _user_supplied_part(text: str, memory_type: MemoryType) -> str:
    """The part of a memory that came from the user. A conclusion's answer summary is the
    assistant's own text, so only its 'Past question' part is checked for injection."""
    if memory_type == MemoryType.CONCLUSION and text.startswith("Past question:") and _CONCLUSION_SPLIT in text:
        return text.split(_CONCLUSION_SPLIT, 1)[0]
    return text


class LongTermMemory:
    def __init__(self, config: MemoryConfig, embedder: Embedder, client: Any = None) -> None:
        self.config = config
        self.embedder = embedder
        self._lock = threading.RLock()
        try:
            import chromadb
            from chromadb.config import Settings

            if client is None:
                settings = Settings(anonymized_telemetry=False)
                if config.persist_dir:
                    client = chromadb.PersistentClient(path=config.persist_dir, settings=settings)
                else:
                    client = chromadb.EphemeralClient(settings=settings)
            self._client = client
            self._collection = client.get_or_create_collection(
                name=config.collection_name,
                embedding_function=None,  # we always pass our own vectors
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"embedding_model": embedder.name, "dimension": embedder.dimension},
            )
        except Exception as exc:
            raise MemoryStoreError(f"Could not open the ChromaDB collection: {exc}") from exc

        # Mixing vectors from two different models makes similarity meaningless.
        stored_model = (self._collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != embedder.name:
            raise MemoryStoreError(
                f"Collection '{config.collection_name}' was built with '{stored_model}', "
                f"but the current embedder is '{embedder.name}'. Use another collection_name or rebuild the store."
            )
        self._id_key = self._load_id_key()

    def _load_id_key(self) -> bytes:
        """Secret for hashing user ids: from config, else a key file next to the store (created once)."""
        if self.config.id_hash_key is not None:
            return self.config.id_hash_key.get_secret_value().encode("utf-8")
        if not self.config.persist_dir:
            return _PROCESS_ID_KEY  # in-memory store: one key per process
        path = Path(self.config.persist_dir) / ".id_hash_key"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Exclusive create: if two processes race on a new store, exactly one key wins.
                with open(path, "x", encoding="utf-8") as handle:
                    key = secrets.token_bytes(32)
                    handle.write(key.hex())
                return key
            except FileExistsError:
                return bytes.fromhex(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise MemoryStoreError(f"Could not read or create the id hash key at {path}: {exc}") from exc

    # ================================================================== helpers
    def _key(self, user_id: Any) -> str:
        """Validate a raw user id and return its pseudonymous storage key."""
        if not isinstance(user_id, str):
            raise MemoryValidationError("user_id must be a string")
        try:
            return pseudonymize_id(validate_user_id(user_id), self._id_key)
        except ValueError as exc:
            raise MemoryValidationError(str(exc)) from exc

    def storage_key(self, user_id: str) -> str:
        """The key under which a user's memories are stored (useful for debugging and tests)."""
        return self._key(user_id)

    @staticmethod
    def _where(user_id: str, **equals: Any) -> dict[str, Any]:
        """Every query is scoped to one user. Extra conditions are AND-ed."""
        conditions = [{"user_id": user_id}] + [{k: v} for k, v in equals.items() if v is not None]
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def _get(self, where: dict[str, Any], include: Optional[list[str]] = None) -> dict[str, Any]:
        try:
            return self._collection.get(where=where, include=include or ["documents", "metadatas"])
        except Exception as exc:
            raise MemoryStoreError(f"Read from memory store failed: {exc}") from exc

    def _records(self, result: dict[str, Any]) -> list[MemoryRecord]:
        ids = result.get("ids") or []
        docs = result.get("documents") or [None] * len(ids)
        metas = result.get("metadatas") or [{}] * len(ids)
        return [MemoryRecord.from_chroma(i, d or "", m or {}) for i, d, m in zip(ids, docs, metas)]

    def _delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:
            raise MemoryStoreError(f"Delete from memory store failed: {exc}") from exc

    def _parent_chunk_ids(self, user_id: str, parent_id: str) -> list[str]:
        return list(self._get(self._where(user_id, parent_id=parent_id), include=[]).get("ids") or [])

    # ================================================================== write
    def add(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        session_id: Optional[str] = None,
    ) -> tuple[WriteOutcome, str, int]:
        """
        Store one memory. Returns (outcome, parent_id, evicted_count).
        Raises MemoryValidationError for bad ids or text that looks like a prompt injection.
        """
        user_id = self._key(user_id)  # pseudonymous storage key from here on
        text = mask_pii(candidate.text)
        if looks_like_injection(_user_supplied_part(text, candidate.memory_type)):
            raise MemoryValidationError("memory text looks like an instruction to the model (prompt injection); not stored")
        preference_value = mask_pii(candidate.preference_value) if candidate.preference_value else None
        session_id = mask_pii(session_id) if session_id else None
        digest = content_hash(user_id, text)
        now = utc_now()

        with self._lock:
            # Rule 2: exact duplicate -> refresh timestamps/confidence only.
            existing = self._records(self._get(self._where(user_id, content_hash=digest)))
            if existing:
                self._touch(existing, max_confidence=candidate.confidence, now=now)
                return WriteOutcome.DUPLICATE, existing[0].parent_id, 0

            replaced: Optional[MemoryRecord] = None
            stale_ids: list[str] = []  # deleted only after the new record is safely written

            # Rule 3: same preference key -> newer value wins.
            if candidate.memory_type == MemoryType.PREFERENCE and candidate.preference_key:
                same_key = self._records(
                    self._get(
                        self._where(
                            user_id,
                            memory_type=MemoryType.PREFERENCE.value,
                            preference_key=candidate.preference_key,
                        )
                    )
                )
                if same_key:
                    replaced = min(same_key, key=lambda r: r.created_at)
                    stale_ids.extend(r.memory_id for r in same_key)

            chunks = chunk_text(
                text,
                self.embedder.count_tokens,
                max_tokens=self.config.chunk_max_tokens,
                overlap_sentences=self.config.chunk_overlap_sentences,
            )
            if not chunks:
                raise MemoryValidationError("memory text is empty after cleaning")
            vectors = self.embedder.embed(chunks)

            # Rule 4: near duplicate of an existing single-chunk memory of the same type.
            # Keyed preferences are already handled by rule 3: two different keys
            # ("format" vs "include_citations") are never merged, however similar the text.
            keyed_preference = candidate.memory_type == MemoryType.PREFERENCE and bool(candidate.preference_key)
            if replaced is None and len(chunks) == 1 and not keyed_preference:
                near = self._nearest(user_id, vectors[0], candidate.memory_type)
                if (
                    near is not None
                    and near[1] >= self.config.duplicate_threshold
                    and not near[0].preference_key  # never overwrite a keyed preference with a keyless one
                ):
                    replaced = near[0]
                    stale_ids.extend(self._parent_chunk_ids(user_id, replaced.parent_id))

            parent_id = digest[:32]
            try:
                records = self._build_records(
                    chunks, user_id, candidate, digest, parent_id, session_id, preference_value,
                    created_at=replaced.created_at if replaced else now, now=now,
                )
            except ValidationError as exc:
                raise MemoryValidationError(f"Invalid memory record: {exc}") from exc
            self._upsert(records, vectors)
            new_ids = {r.memory_id for r in records}
            self._delete_ids([i for i in stale_ids if i not in new_ids])
            evicted = self._enforce_cap(user_id)

        outcome = WriteOutcome.UPDATED if replaced else WriteOutcome.ADDED
        logger.debug("memory %s user=%s parent=%s chunks=%d", outcome.value, user_id, parent_id, len(chunks))
        return outcome, parent_id, evicted

    @staticmethod
    def _build_records(
        chunks: list[str],
        user_key: str,
        candidate: MemoryCandidate,
        digest: str,
        parent_id: str,
        session_id: Optional[str],
        preference_value: Optional[str],
        created_at: datetime,
        now: datetime,
    ) -> list[MemoryRecord]:
        return [
            MemoryRecord(
                memory_id=parent_id if len(chunks) == 1 else f"{parent_id}#{i}",
                user_id=user_key,
                text=chunk,
                memory_type=candidate.memory_type,
                confidence=candidate.confidence,
                importance=round(TYPE_IMPORTANCE[candidate.memory_type] * candidate.confidence, 4),
                source=candidate.source,
                content_hash=digest,
                parent_id=parent_id,
                chunk_index=i,
                chunk_count=len(chunks),
                session_id=session_id,
                preference_key=candidate.preference_key,
                preference_value=preference_value,
                created_at=created_at,
                updated_at=now,
            )
            for i, chunk in enumerate(chunks)
        ]

    def _upsert(self, records: list[MemoryRecord], vectors: list[Vector]) -> None:
        try:
            self._collection.upsert(
                ids=[r.memory_id for r in records],
                documents=[r.text for r in records],
                metadatas=[r.to_metadata() for r in records],
                embeddings=vectors,
            )
        except Exception as exc:
            raise MemoryStoreError(f"Write to memory store failed: {exc}") from exc

    def _touch(self, records: list[MemoryRecord], max_confidence: float, now: datetime) -> None:
        updated = []
        for record in records:
            confidence = max(record.confidence, max_confidence)
            updated.append(
                record.model_copy(
                    update={
                        "confidence": confidence,
                        "importance": round(TYPE_IMPORTANCE[record.memory_type] * confidence, 4),
                        "updated_at": now,
                    }
                )
            )
        try:
            self._collection.update(
                ids=[r.memory_id for r in updated],
                metadatas=[r.to_metadata() for r in updated],
            )
        except Exception as exc:
            raise MemoryStoreError(f"Update of memory store failed: {exc}") from exc

    def _nearest(
        self, user_key: str, vector: Vector, memory_type: MemoryType
    ) -> Optional[tuple[MemoryRecord, float]]:
        hits = self._search_key(user_key, vector, n_results=1, memory_type=memory_type)
        if not hits:
            return None
        record, similarity, _ = hits[0]
        return (record, similarity) if record.chunk_count == 1 else None

    def _enforce_cap(self, user_id: str) -> int:
        """Keep at most max_memories_per_user memories; evict lowest importance, then oldest."""
        metas = self._get(self._where(user_id, chunk_index=0), include=["metadatas"]).get("metadatas") or []
        overflow = len(metas) - self.config.max_memories_per_user
        if overflow <= 0:
            return 0
        victims = sorted(metas, key=lambda m: (m.get("importance", 0.0), m.get("updated_ts", 0.0)))[:overflow]
        for meta in victims:
            self._delete_ids(self._parent_chunk_ids(user_id, meta["parent_id"]))
        logger.info("evicted %d memories for user=%s (cap=%d)", overflow, user_id, self.config.max_memories_per_user)
        return overflow

    # ================================================================== read
    def search(
        self,
        user_id: str,
        query_vector: Vector,
        n_results: int,
        memory_type: Optional[MemoryType] = None,
    ) -> list[tuple[MemoryRecord, float, Vector]]:
        """
        Top-n memories of ONE user by cosine similarity. Returns (record, similarity, vector).

        Exact search: a user holds at most max_memories_per_user records, so we
        load that user's vectors and rank them all. This avoids the recall loss of
        approximate (HNSW) search combined with a metadata filter on a large,
        multi-user collection, and computes cosine ourselves, so the result does
        not depend on the collection's distance setting.
        """
        return self._search_key(self._key(user_id), query_vector, n_results, memory_type)

    def _search_key(
        self,
        user_key: str,
        query_vector: Vector,
        n_results: int,
        memory_type: Optional[MemoryType] = None,
    ) -> list[tuple[MemoryRecord, float, Vector]]:
        if n_results <= 0:
            return []
        result = self._get(
            self._where(user_key, memory_type=memory_type.value if memory_type else None),
            include=["documents", "metadatas", "embeddings"],
        )
        ids = result.get("ids") or []
        embeddings = result.get("embeddings")
        if not ids or embeddings is None or len(embeddings) == 0:
            return []

        matrix = np.asarray(embeddings, dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(query))
        similarities = (matrix @ query) / np.maximum(norms, 1e-12)  # cos = A.B / (|A||B|)
        order = np.argsort(-similarities, kind="stable")[:n_results]

        docs = result.get("documents") or [""] * len(ids)
        metas = result.get("metadatas") or [{}] * len(ids)
        return [
            (
                MemoryRecord.from_chroma(ids[i], docs[i] or "", metas[i] or {}),
                float(similarities[i]),
                matrix[i].tolist(),
            )
            for i in order
        ]

    def get_by_type(self, user_id: str, memory_type: MemoryType, limit: int = 50) -> list[MemoryRecord]:
        records = self._records(self._get(self._where(self._key(user_id), memory_type=memory_type.value)))
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records[:limit]

    def get_all(self, user_id: str) -> list[MemoryRecord]:
        records = self._records(self._get(self._where(self._key(user_id))))
        records.sort(key=lambda r: (r.parent_id, r.chunk_index))
        return records

    def count(self, user_id: str) -> int:
        """Number of memories (not chunks) for a user."""
        return len(self._get(self._where(self._key(user_id), chunk_index=0), include=[]).get("ids") or [])

    # ================================================================== delete ("forget me")
    def delete_memory(self, user_id: str, parent_id: str) -> int:
        user_id = self._key(user_id)
        with self._lock:
            ids = self._parent_chunk_ids(user_id, parent_id)
            self._delete_ids(ids)
        return len(ids)

    def delete_user(self, user_id: str) -> int:
        """
        Right to be forgotten: remove every memory of this user. Returns records deleted.
        Note: ChromaDB marks deleted vectors and reclaims the disk space during its own
        compaction; deleted memories are never returned by any query.
        """
        user_id = self._key(user_id)
        with self._lock:
            ids = list(self._get(self._where(user_id), include=[]).get("ids") or [])
            self._delete_ids(ids)
        logger.info("deleted all %d memory records for user=%s", len(ids), user_id)
        return len(ids)
