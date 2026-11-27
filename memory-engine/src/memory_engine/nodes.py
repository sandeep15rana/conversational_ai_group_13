"""
LangGraph integration points (Developer 3 - Rajarshi).

Drop-in nodes for Sandeep's StateGraph (orchestration/graph.py). They read and
write only fields that already exist in orchestration/state.py:AgentState:

  memory_recall_node  reads : user_id, session_id, sanitized_query, is_safe
                      writes: recalled_context, user_preferences, execution_trace
                      recalled_context: str (may be "" when nothing is relevant),
                                        None only when memory was skipped or failed
  memory_save_node    reads : user_id, session_id, final_output, is_safe, is_hallucination_detected
                      writes: execution_trace

Both nodes FAIL SOFT: if anything in the memory layer breaks, they log it in the
trace and return empty memory, so the pipeline still answers.

A missing user_id means "unknown user": memory is skipped (unknown users are
never pooled into one shared bucket).

Helpers for other teams:
  memory_context_for_synthesis(state) -> list[str]  for Pooja's SynthesisRequest.memory_context
  build_supervisor_memory_block(state) -> str       for an LLM-based supervisor prompt
  save_scratchpad / load_scratchpad(state, ...)     session-scoped working notes for any agent
  no_memory_node(state)                             "without memory" baseline for ablation studies
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from .engine import get_default_engine

logger = logging.getLogger(__name__)

TRACE_PREFIX = "[Memory Engine]"


HALLUCINATION_PLACEHOLDER = "[answer withheld from memory: flagged by the output guardrail]"


def _ids(state: Mapping[str, Any]) -> tuple[str, str]:
    """
    (user_id, session_id) as clean strings. user_id may be "" (unknown user):
    callers then skip memory entirely instead of pooling unknown users together.
    """
    raw_user = state.get("user_id")
    if raw_user is not None and not isinstance(raw_user, str):
        # No str() coercion: user_id=123 must not share a partition with "123".
        raise TypeError(f"user_id must be a string, got {type(raw_user).__name__}")
    user_id = raw_user.strip() if raw_user is not None else ""
    raw_session = state.get("session_id")
    session_id = str(raw_session).strip() if raw_session is not None else ""
    return user_id, session_id or "default_session"


def memory_recall_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Runs after the input guardrail, before the supervisor."""
    empty = {"recalled_context": None, "user_preferences": {}}
    try:
        user_id, session_id = _ids(state)
    except Exception as exc:  # fail soft, even on very odd state values
        return {**empty, "execution_trace": [f"{TRACE_PREFIX}: WARNING - unreadable user/session id ({exc}). Memory skipped."]}

    if not user_id:
        return {**empty, "execution_trace": [f"{TRACE_PREFIX}: No user_id in state - memory skipped (unknown users never share memory)."]}

    trace = [f"{TRACE_PREFIX}: Recalling long-term memory for user '{user_id}'..."]
    if not state.get("is_safe", True):
        trace.append(f"{TRACE_PREFIX}: Input flagged unsafe - memory recall skipped.")
        return {**empty, "execution_trace": trace}

    query = state.get("sanitized_query") or ""
    try:
        query = query if isinstance(query, str) else str(query)
        engine = get_default_engine()
        result = engine.recall(user_id, query, session_id=session_id)
        if query.strip():
            engine.record_turn(user_id, session_id, "user", query)  # after recall, so 'recent' excludes this query
    except Exception as exc:  # fail soft
        logger.exception("memory recall failed")
        trace.append(f"{TRACE_PREFIX}: WARNING - recall failed ({type(exc).__name__}: {exc}). Continuing without memory.")
        return {"recalled_context": None, "user_preferences": {}, "execution_trace": trace}

    trace.append(
        f"{TRACE_PREFIX}: Recalled {len(result.memories)} relevant memories, "
        f"{len(result.profile_facts)} profile facts and "
        f"{len(result.preferences)} preferences {sorted(result.preferences)}."
    )
    return {
        # "" = memory ran but found nothing; None = memory skipped or unavailable.
        "recalled_context": result.context_text,
        "user_preferences": result.preferences,
        "execution_trace": trace,
    }


def memory_save_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Runs after the output guardrail, before END. Extracts and stores new memories."""
    try:
        user_id, session_id = _ids(state)
    except Exception as exc:  # fail soft
        return {"execution_trace": [f"{TRACE_PREFIX}: WARNING - unreadable user/session id ({exc}). Nothing saved."]}

    if not user_id:
        return {"execution_trace": [f"{TRACE_PREFIX}: No user_id in state - nothing saved."]}

    trace = [f"{TRACE_PREFIX}: Consolidating memories from this turn..."]
    if not state.get("is_safe", True):
        trace.append(f"{TRACE_PREFIX}: Unsafe session - nothing saved.")
        return {"execution_trace": trace}

    final_output = state.get("final_output") or ""
    hallucination = bool(state.get("is_hallucination_detected", False))
    try:
        final_output = final_output if isinstance(final_output, str) else str(final_output)
        engine = get_default_engine()
        # A distrusted answer is not replayed as "recent conversation" either.
        engine.record_turn(user_id, session_id, "assistant", HALLUCINATION_PLACEHOLDER if hallucination else final_output)
        result = engine.consolidate(
            user_id,
            session_id,
            final_output=final_output,
            include_conclusions=not hallucination,  # never memorize an answer the guardrail distrusts
        )
    except Exception as exc:  # fail soft
        logger.exception("memory save failed")
        trace.append(f"{TRACE_PREFIX}: WARNING - save failed ({type(exc).__name__}: {exc}). Answer is unaffected.")
        return {"execution_trace": trace}

    trace.append(
        f"{TRACE_PREFIX}: Saved via '{result.extractor}' - added {result.added}, updated {result.updated}, "
        f"duplicates {result.duplicates}, rejected {result.rejected}, evicted {result.evicted}."
    )
    if hallucination:
        trace.append(f"{TRACE_PREFIX}: Output guardrail flagged hallucination - conclusion not memorized.")
    return {"execution_trace": trace}


def no_memory_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """
    Baseline for ablation studies: same state shape as memory_recall_node, but
    recalls nothing. Use it in place of memory_recall_node (and drop memory_save)
    to measure the system WITHOUT memory. (mock_memory_recall is NOT a baseline:
    it injects fake preferences.)
    """
    return {
        "recalled_context": "",
        "user_preferences": {},
        "execution_trace": [f"{TRACE_PREFIX}: Memory disabled (ablation baseline)."],
    }


def save_scratchpad(state: Mapping[str, Any], key: str, value: Any) -> bool:
    """
    For any agent: keep an intermediate result (plan, partial findings...) for the
    rest of this session. Returns False instead of raising if memory is unavailable.
    """
    try:
        user_id, session_id = _ids(state)
        if not user_id:
            return False
        get_default_engine().set_scratchpad(user_id, session_id, key, value)
        return True
    except Exception:
        logger.exception("scratchpad write failed")
        return False


def load_scratchpad(state: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read a value stored with save_scratchpad in this session (default if missing or unavailable)."""
    try:
        user_id, session_id = _ids(state)
        if not user_id:
            return default
        return get_default_engine().get_scratchpad(user_id, session_id, key, default)
    except Exception:
        logger.exception("scratchpad read failed")
        return default


def memory_context_for_synthesis(state: Mapping[str, Any]) -> list[str]:
    """recalled_context as a list of memory lines, for SynthesisRequest(memory_context=...)."""
    context = state.get("recalled_context") or ""
    return [line[2:].strip() for line in context.splitlines() if line.startswith("- ")]


def build_supervisor_memory_block(state: Mapping[str, Any]) -> str:
    """Prompt section to append to an LLM supervisor prompt. Empty string if there is no memory."""
    context = state.get("recalled_context")
    if not context:
        return ""
    return (
        "### Long-term memory about this user\n"
        f"{context}\n"
        "Use these preferences and facts when planning and formatting the answer. "
        "If they conflict with the current request, the current request wins."
    )
