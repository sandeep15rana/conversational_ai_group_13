"""
MemoryEngine - single entry point that wires every memory component together (Developer 3 - Rajarshi).

    engine = MemoryEngine()                       # settings from MEMORY_* env vars
    result = engine.recall("user_1", "query", session_id="thread_1")
    engine.record_turn("user_1", "thread_1", "assistant", "answer...")
    engine.consolidate("user_1", "thread_1", final_output="answer...")

Other teams normally do not use this class directly; they use the two LangGraph
nodes in nodes.py, which call the shared default engine.
"""

from __future__ import annotations

import logging
import threading

from typing import Any, Optional

from pydantic import ValidationError

from .config import MemoryConfig
from .embeddings import Embedder, MiniLMEmbedder
from .exceptions import MemoryEngineError, MemoryValidationError
from .extractor import MemoryExtractor, build_default_extractor, filter_candidates
from .long_term import LongTermMemory
from .retriever import MemoryRetriever
from .schemas import (
    MemoryCandidate,
    MemoryRecord,
    MemorySource,
    MemoryType,
    RecallResult,
    SaveResult,
    WriteOutcome,
    validate_user_id,
)
from .short_term import ShortTermMemory

logger = logging.getLogger(__name__)

_CONSOLIDATED_SEQ_KEY = "_memory_consolidated_seq"


class MemoryEngine:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedder: Optional[Embedder] = None,
        extractor: Optional[MemoryExtractor] = None,
        client: Any = None,
    ) -> None:
        self.config = config or MemoryConfig.from_env()
        self.embedder = embedder or MiniLMEmbedder(self.config.embedding_model_dir)
        self.short_term = ShortTermMemory(
            max_turns=self.config.short_term_max_turns,
            ttl_seconds=self.config.short_term_ttl_seconds,
            max_sessions=self.config.short_term_max_sessions,
        )
        self.long_term = LongTermMemory(self.config, self.embedder, client=client)
        self.extractor = extractor or build_default_extractor(self.config)
        self.retriever = MemoryRetriever(self.config, self.embedder, self.long_term)

    # ------------------------------------------------------------------ read path
    def recall(self, user_id: str, query: str, session_id: Optional[str] = None) -> RecallResult:
        """Profile preferences + relevant episodic memories + recent turns, as prompt-ready text."""
        preferences, preference_records = self.retriever.load_preferences(user_id)
        profile_facts = self.retriever.load_profile_facts(user_id)
        memories = self.retriever.retrieve(user_id, query, exclude_parent_ids={r.parent_id for r in profile_facts})
        recent_text, recent_turns = "", []
        if session_id and self.config.recent_turns_in_context:
            key = self._session_key(user_id, session_id)
            recent_turns = self.short_term.get_turns(key, last_n=self.config.recent_turns_in_context)
            recent_text = self.short_term.format_recent(key, last_n=self.config.recent_turns_in_context)
        context = self.retriever.build_context(preference_records, memories, recent_text, fact_records=profile_facts)
        return RecallResult(
            context_text=context,
            preferences=preferences,
            memories=memories,
            profile_facts=profile_facts,
            recent_turns=recent_turns,
        )

    # ------------------------------------------------------------------ short-term
    @staticmethod
    def _session_key(user_id: str, session_id: str) -> str:
        """Short-term buffers are keyed by user AND session, so a reused session id never leaks turns."""
        try:
            return f"{validate_user_id(user_id)}::{session_id}"
        except ValueError as exc:
            raise MemoryValidationError(str(exc)) from exc

    def record_turn(self, user_id: str, session_id: str, role: str, content: str) -> None:
        if content and content.strip():
            self.short_term.add_turn(self._session_key(user_id, session_id), role, content)

    # ------------------------------------------------------------------ write path
    def consolidate(
        self,
        user_id: str,
        session_id: str,
        final_output: str = "",
        include_conclusions: bool = True,
    ) -> SaveResult:
        """
        Extract memories from the turns not yet processed in this session and store them.
        Safe to call after every turn: already-processed turns are skipped and duplicates are merged.
        """
        key = self._session_key(user_id, session_id)
        last_seq: int = self.short_term.get_scratchpad(key, _CONSOLIDATED_SEQ_KEY, default=0)
        turns = [t for t in self.short_term.get_turns(key) if t.seq > last_seq]
        if not turns:
            return SaveResult(extractor="none")

        candidates = self.extractor.extract(turns, final_output)
        if not include_conclusions:
            candidates = [c for c in candidates if c.memory_type != MemoryType.CONCLUSION]
        kept = filter_candidates(candidates, self.config.min_extraction_confidence)

        result = SaveResult(
            rejected=len(candidates) - len(kept),
            extractor=getattr(self.extractor, "last_used", self.extractor.name),
        )
        for candidate in kept:
            try:
                outcome, _, evicted = self.long_term.add(user_id, candidate, session_id=session_id)
            except MemoryEngineError as exc:  # one bad candidate must not drop the rest of the batch
                logger.warning("Rejected memory %r: %s", candidate.text, exc)
                result.rejected += 1
                continue
            result.evicted += evicted
            if outcome == WriteOutcome.ADDED:
                result.added += 1
            elif outcome == WriteOutcome.UPDATED:
                result.updated += 1
            else:
                result.duplicates += 1

        self.short_term.set_scratchpad(key, _CONSOLIDATED_SEQ_KEY, turns[-1].seq)
        return result

    def remember(
        self,
        user_id: str,
        text: str,
        memory_type: MemoryType = MemoryType.FACT,
        preference_key: Optional[str] = None,
        preference_value: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> WriteOutcome:
        """Store one memory directly (e.g. from a settings screen or a seed script)."""
        try:
            candidate = MemoryCandidate(
                text=text,
                memory_type=memory_type,
                confidence=1.0,
                source=MemorySource.MANUAL,
                preference_key=preference_key,
                preference_value=preference_value,
            )
        except ValidationError as exc:
            raise MemoryValidationError(f"Invalid memory: {exc}") from exc
        outcome, _, _ = self.long_term.add(user_id, candidate, session_id=session_id)
        return outcome

    # ------------------------------------------------------------------ scratchpad (agent working notes)
    def set_scratchpad(self, user_id: str, session_id: str, key: str, value: Any) -> None:
        """Keep an intermediate result for the rest of this session (e.g. a plan, partial findings)."""
        if isinstance(key, str) and key.startswith("_memory_"):
            raise MemoryValidationError("scratchpad keys starting with '_memory_' are reserved")
        self.short_term.set_scratchpad(self._session_key(user_id, session_id), key, value)

    def get_scratchpad(self, user_id: str, session_id: str, key: str, default: Any = None) -> Any:
        return self.short_term.get_scratchpad(self._session_key(user_id, session_id), key, default)

    # ------------------------------------------------------------------ admin
    def list_memories(self, user_id: str) -> list[MemoryRecord]:
        return self.long_term.get_all(user_id)

    def forget_user(self, user_id: str) -> int:
        """Delete every long-term memory of the user. Returns the number of records removed."""
        return self.long_term.delete_user(user_id)

    def end_session(self, user_id: str, session_id: str) -> None:
        self.short_term.clear_session(self._session_key(user_id, session_id))


# ---------------------------------------------------------------------- shared default instance
_default_engine: Optional[MemoryEngine] = None
_default_lock = threading.Lock()


def get_default_engine() -> MemoryEngine:
    """The engine used by the LangGraph nodes. Created on first use from MEMORY_* env vars."""
    global _default_engine
    if _default_engine is None:
        with _default_lock:
            if _default_engine is None:
                _default_engine = MemoryEngine()
    return _default_engine


def set_default_engine(engine: Optional[MemoryEngine]) -> None:
    """Swap the shared engine (tests, custom config). Pass None to reset."""
    global _default_engine
    with _default_lock:
        _default_engine = engine
