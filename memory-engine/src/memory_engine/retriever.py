"""
Semantic memory retrieval - the RAG step (Developer 3 - Rajarshi).

Two kinds of recall, combined (MemGPT-style "core" + "archival" memory):
  1. Profile memory (always loaded): the user's preferences and the newest facts
     about the user (role, company...). A query about "NVIDIA revenue" is not
     similar to "I prefer bullet points" or "User is an equity analyst", yet both
     must still shape the answer, so they are not left to similarity.
  2. Episodic memory (semantic): past conclusions, plus any older facts not in
     the profile, that are relevant to the query.

Episodic pipeline:
  query -> embed -> top fetch_k by cosine (this user only) -> drop below min_similarity
  -> merge chunks of the same memory -> score = similarity + recency bonus
  -> MMR to pick diverse results -> top_k -> fit into a token budget -> prompt text

  recency_bonus = recency_weight * 0.5 ** (age_days / half_life_days)
  MMR(d)        = lambda * score(d) - (1 - lambda) * max_cosine(d, already_selected)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from .config import MemoryConfig
from .embeddings import Embedder, Vector, cosine_similarity
from .long_term import LongTermMemory
from .schemas import MemoryRecord, MemoryType, RetrievedMemory, utc_now

_SECTION_TITLES = {
    MemoryType.PREFERENCE: "User preferences",
    MemoryType.FACT: "Known facts about the user",
    MemoryType.CONCLUSION: "Relevant past conclusions",
}


def _parse_preference_value(value: Optional[str]) -> Any:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    return value


class MemoryRetriever:
    def __init__(self, config: MemoryConfig, embedder: Embedder, store: LongTermMemory) -> None:
        self.config = config
        self.embedder = embedder
        self.store = store

    # ------------------------------------------------------------------ profile
    def load_preferences(self, user_id: str) -> tuple[dict[str, Any], list[MemoryRecord]]:
        """All stored preferences of the user (newest first) -> ({key: value}, records)."""
        records = self.store.get_by_type(user_id, MemoryType.PREFERENCE, limit=self.config.max_profile_preferences)
        preferences: dict[str, Any] = {}
        for record in records:  # newest first, so the first value per key wins
            if record.preference_key and record.preference_key not in preferences:
                preferences[record.preference_key] = _parse_preference_value(record.preference_value)
        return preferences, records

    def load_profile_facts(self, user_id: str) -> list[MemoryRecord]:
        """Newest facts about the user (single-chunk), always shown to the supervisor."""
        if self.config.max_profile_facts == 0:
            return []
        records = self.store.get_by_type(user_id, MemoryType.FACT, limit=self.config.max_profile_facts * 2)
        return [r for r in records if r.chunk_count == 1][: self.config.max_profile_facts]

    # ------------------------------------------------------------------ episodic
    def _recency_bonus(self, updated_at: datetime, now: datetime) -> float:
        age_days = max((now - updated_at).total_seconds() / 86400.0, 0.0)
        return self.config.recency_weight * 0.5 ** (age_days / self.config.recency_half_life_days)

    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
        exclude_parent_ids: Optional[set[str]] = None,
    ) -> list[RetrievedMemory]:
        """Facts and conclusions relevant to the query, best first (excluding memories already in the profile)."""
        query = (query or "").strip()
        if not query:
            return []
        k = top_k or self.config.top_k
        exclude = exclude_parent_ids or set()
        query_vector = self.embedder.embed([query])[0]
        hits = self.store.search(user_id, query_vector, n_results=k * self.config.fetch_k_multiplier)

        # Drop preferences (handled by the profile), profile facts, and weak matches.
        hits = [
            h for h in hits
            if h[0].memory_type != MemoryType.PREFERENCE
            and h[0].parent_id not in exclude
            and h[1] >= self.config.min_similarity
        ]

        # Merge chunks of the same memory: best similarity, chunks joined in order.
        groups: dict[str, list[tuple[MemoryRecord, float, Vector]]] = {}
        for hit in hits:
            groups.setdefault(hit[0].parent_id, []).append(hit)

        now = utc_now()
        merged: list[tuple[RetrievedMemory, Vector]] = []
        for chunks in groups.values():
            chunks.sort(key=lambda h: h[0].chunk_index)
            best = max(chunks, key=lambda h: h[1])
            record = chunks[0][0]
            if len(chunks) > 1:
                record = record.model_copy(update={"text": " ... ".join(c[0].text for c in chunks)})
            similarity = best[1]
            score = similarity + self._recency_bonus(record.updated_at, now)
            merged.append((RetrievedMemory(record=record, similarity=round(similarity, 4), score=round(score, 4)), best[2]))

        return self._mmr(merged, k)

    def _mmr(self, items: list[tuple[RetrievedMemory, Vector]], k: int) -> list[RetrievedMemory]:
        """Maximal Marginal Relevance: relevant AND not repetitive."""
        remaining = sorted(items, key=lambda x: x[0].score, reverse=True)
        selected: list[tuple[RetrievedMemory, Vector]] = []
        lam = self.config.mmr_lambda
        while remaining and len(selected) < k:
            chosen_vectors = [s[1] for s in selected if s[1]]

            def mmr_score(item: tuple[RetrievedMemory, Vector]) -> float:
                if not chosen_vectors or not item[1]:
                    return item[0].score
                redundancy = max(cosine_similarity(item[1], v) for v in chosen_vectors)
                return lam * item[0].score - (1 - lam) * redundancy

            best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
        return [item[0] for item in selected]

    # ------------------------------------------------------------------ prompt text
    def build_context(
        self,
        preference_records: list[MemoryRecord],
        memories: list[RetrievedMemory],
        recent_conversation: str = "",
        fact_records: Optional[list[MemoryRecord]] = None,
    ) -> str:
        """Sectioned text for the supervisor prompt, cut to context_token_budget tokens (priority order)."""
        budget = self.config.context_token_budget
        used = 0
        lines: list[str] = []

        facts = [r.text for r in (fact_records or [])]
        facts += [m.record.text for m in memories if m.record.memory_type == MemoryType.FACT]
        conclusions = [m.record.text for m in memories if m.record.memory_type == MemoryType.CONCLUSION]

        sections: list[tuple[str, list[str]]] = [
            (_SECTION_TITLES[MemoryType.PREFERENCE], [r.text for r in preference_records]),
            (_SECTION_TITLES[MemoryType.FACT], facts),
            (_SECTION_TITLES[MemoryType.CONCLUSION], conclusions),
            ("Recent conversation (this session)", recent_conversation.splitlines() if recent_conversation else []),
        ]
        for title, items in sections:
            title_cost = self.embedder.count_tokens(f"{title}:")
            section_lines: list[str] = []
            section_cost = title_cost
            for item in items:
                line = f"- {item}"
                cost = self.embedder.count_tokens(line)
                if used + section_cost + cost > budget:
                    break
                section_lines.append(line)
                section_cost += cost
            if section_lines:  # a title is only shown with at least one item
                lines.append(f"{title}:")
                lines.extend(section_lines)
                used += section_cost
        return "\n".join(lines)
