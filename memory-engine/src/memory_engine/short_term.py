"""
Short-term memory: per-session conversation buffer + agent scratchpad (Developer 3 - Rajarshi).

- Each session (LangGraph thread_id / session_id) has its own buffer; sessions never mix.
- Only the last `max_turns` turns are kept (oldest drop off automatically).
- Sessions idle longer than `ttl_seconds` expire.
- At most `max_sessions` sessions are held; the least recently used is evicted first.
- Thread-safe (one re-entrant lock). Turn text and string scratchpad values are PII-masked.
- Every turn gets a per-session sequence number (timestamps can collide on fast clocks).
- Lives in process memory: run one worker process per memory store (see INTEGRATION.md).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

from .exceptions import MemoryValidationError
from .privacy import mask_pii
from .schemas import ConversationTurn

_MAX_SCRATCHPAD_KEYS = 100


@dataclass
class _Session:
    turns: Deque[ConversationTurn]
    scratchpad: dict[str, Any] = field(default_factory=dict)
    last_access: float = 0.0
    next_seq: int = 1


class ShortTermMemory:
    def __init__(
        self,
        max_turns: int = 20,
        ttl_seconds: int = 3600,
        max_sessions: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_turns < 1 or ttl_seconds < 1 or max_sessions < 1:
            raise MemoryValidationError("max_turns, ttl_seconds and max_sessions must be positive")
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._clock = clock
        self._sessions: "OrderedDict[str, _Session]" = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _check_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise MemoryValidationError("session_id must be a non-empty string")
        return session_id.strip()

    def _evict_expired(self, now: float) -> None:
        expired = [sid for sid, s in self._sessions.items() if now - s.last_access > self.ttl_seconds]
        for sid in expired:
            del self._sessions[sid]

    def _get_or_create(self, session_id: str) -> _Session:
        now = self._clock()
        self._evict_expired(now)
        session = self._sessions.get(session_id)
        if session is None:
            session = _Session(turns=deque(maxlen=self.max_turns))
            self._sessions[session_id] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)  # least recently used
        session.last_access = now
        self._sessions.move_to_end(session_id)
        return session

    # ------------------------------------------------------------------ turns
    def add_turn(self, session_id: str, role: str, content: str) -> ConversationTurn:
        session_id = self._check_session_id(session_id)
        if not isinstance(content, str):
            raise MemoryValidationError("turn content must be a string")
        with self._lock:
            session = self._get_or_create(session_id)
            try:
                turn = ConversationTurn(role=role, content=mask_pii(content), seq=session.next_seq)
            except Exception as exc:
                raise MemoryValidationError(f"Invalid conversation turn: {exc}") from exc
            session.next_seq += 1
            session.turns.append(turn)
        return turn

    def get_turns(self, session_id: str, last_n: Optional[int] = None) -> list[ConversationTurn]:
        session_id = self._check_session_id(session_id)
        with self._lock:
            self._evict_expired(self._clock())
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_access = self._clock()
            self._sessions.move_to_end(session_id)  # reading counts as use for LRU
            turns = list(session.turns)
        if last_n is not None:
            turns = turns[-last_n:] if last_n > 0 else []
        return turns

    def format_recent(self, session_id: str, last_n: int = 4) -> str:
        """Recent turns as plain text for a prompt, e.g. 'User: ...'."""
        lines = []
        for turn in self.get_turns(session_id, last_n=last_n):
            content = turn.content if len(turn.content) <= 300 else turn.content[:297] + "..."
            lines.append(f"{turn.role.capitalize()}: {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ scratchpad
    def set_scratchpad(self, session_id: str, key: str, value: Any) -> None:
        """Intermediate notes an agent wants to keep for the rest of the session."""
        session_id = self._check_session_id(session_id)
        if not isinstance(key, str) or not key.strip():
            raise MemoryValidationError("scratchpad key must be a non-empty string")
        with self._lock:
            pad = self._get_or_create(session_id).scratchpad
            if key not in pad and len(pad) >= _MAX_SCRATCHPAD_KEYS:
                raise MemoryValidationError(f"scratchpad is limited to {_MAX_SCRATCHPAD_KEYS} keys per session")
            pad[key] = mask_pii(value) if isinstance(value, str) else value

    def get_scratchpad(self, session_id: str, key: Optional[str] = None, default: Any = None) -> Any:
        session_id = self._check_session_id(session_id)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {} if key is None else default
            return dict(session.scratchpad) if key is None else session.scratchpad.get(key, default)

    # ------------------------------------------------------------------ housekeeping
    def clear_session(self, session_id: str) -> None:
        session_id = self._check_session_id(session_id)
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            self._evict_expired(self._clock())
            return len(self._sessions)
