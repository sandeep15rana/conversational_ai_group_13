"""
Memory extraction pipeline (Developer 3 - Rajarshi).

After a turn finishes, decide what is worth remembering:
  - preferences  (how the user wants answers)  -> from USER messages
  - facts        (who the user is)              -> from USER messages
  - conclusions  (what was concluded)           -> from the final answer

Two extractors:
  - RuleBasedExtractor: regex rules, deterministic, offline, no API key. Always available.
  - LLMExtractor: asks an LLM for structured JSON, validated with Pydantic.
HybridExtractor runs the LLM first and falls back to rules on any failure.
build_default_extractor() picks LLM+rules when OPENAI_API_KEY is set, else rules only.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional, Protocol, Sequence

from .config import MemoryConfig
from .exceptions import ExtractionError
from .privacy import looks_like_injection
from .schemas import ConversationTurn, MemoryCandidate, MemorySource, MemoryType, normalize_text

logger = logging.getLogger(__name__)


class MemoryExtractor(Protocol):
    name: str

    def extract(self, turns: Sequence[ConversationTurn], final_output: str = "") -> list[MemoryCandidate]: ...


def filter_candidates(candidates: Sequence[MemoryCandidate], min_confidence: float) -> list[MemoryCandidate]:
    """Drop low-confidence candidates and duplicates inside one batch."""
    seen: set[str] = set()
    kept: list[MemoryCandidate] = []
    for candidate in candidates:
        key = normalize_text(candidate.text)
        if candidate.confidence < min_confidence or key in seen:
            continue
        seen.add(key)
        kept.append(candidate)
    return kept


# ============================================================================ rules
_SENTENCES = re.compile(r"(?<=[.!?])\s+|(?<=[。！？।])\s*|\n+")
_NEGATION = re.compile(r"\b(?:don't|don’t|do not|never|no|not|hate|dislike)\b", re.IGNORECASE)
# Standing-preference wording only. "I need ..." is deliberately NOT a trigger: it is
# usually a one-off request ("I need the revenue table for Q1").
_PREFERENCE_TRIGGER = re.compile(
    r"\b(?:i(?:\s+would|'d|’d)?\s+(?:prefer|like|want|love)|please\s+(?:always|use|keep|give|include|add)"
    r"|always|from now on|going forward|keep (?:it|answers|responses|the answers?)|i\s+(?:hate|dislike)"
    r"|(?:don't|don’t|do not|never)\s+(?:give|use|include|add|send|want|like))\b",
    re.IGNORECASE,
)
# Negation applies only inside the clause that contains the keyword:
# "don't give me tables, use bullet points" -> tables negated, bullet points not.
_CLAUSE_BREAK = re.compile(r"[,;:]|\bbut\b|\binstead\b|\brather\b|\bthen\b", re.IGNORECASE)
# "the revenue table", "the Q3 chart" name a specific object: a one-off request, not a format preference.
_SPECIFIC_OBJECT = re.compile(r"\bthe\s+(?:[\w\-]+\s+){0,2}$", re.IGNORECASE)
# ...unless the phrase names the format itself: "the table format", "the bullet point style".
_FORMAT_NOUN = re.compile(r"\s*(?:format|style|layout|view|form)\b", re.IGNORECASE)

# (regex, preference_key, value, opposite value when negated)
_PREFERENCE_KEYWORDS: list[tuple[re.Pattern[str], str, str, Optional[str]]] = [
    (re.compile(r"\bbullet(?:ed)?(?:\s+points?|\s+lists?)?\b", re.I), "format", "bullet_points", None),
    (re.compile(r"\btables?\b|\btabular\b", re.I), "format", "table", None),
    (re.compile(r"\bjson\b", re.I), "format", "json", None),
    (re.compile(r"\bparagraphs?\b|\bprose\b", re.I), "format", "paragraphs", None),
    (re.compile(r"\b(?:concise|short|brief|crisp|to the point)\b", re.I), "detail_level", "concise", "detailed"),
    (
        re.compile(r"\b(?:detailed|in[- ]depth|comprehensive|thorough)\b|\blong(?:er)?\s+(?:answers?|responses?|explanations?|reports?)\b", re.I),
        "detail_level",
        "detailed",
        "concise",
    ),
    (re.compile(r"\b(?:citations?|cite|references?)\b|\bsources?\b(?!\s+of\b)", re.I), "include_citations", "true", "false"),
    (re.compile(r"\b(?:charts?|graphs?|visuals?)\b", re.I), "include_charts", "true", "false"),
]

_FACT_RULES: list[tuple[re.Pattern[str], str, float]] = [
    # (?i:...) makes only the lead phrase case-insensitive; names must still start with a capital.
    (re.compile(r"\b(?i:my name is|call me) ([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)?)"), "User's name is {0}", 0.9),
    (re.compile(r"\bi work (?:at|for|in) ([^.,;!?\n]{2,60})", re.I), "User works at {0}", 0.85),
    (re.compile(r"\bi work as (?:an? |the )?([^.,;!?\n]{2,60})", re.I), "User works as {0}", 0.85),
    (re.compile(r"\bmy (?:role|job|title|designation) is (?:an? |the )?([^.,;!?\n]{2,60})", re.I), "User's role is {0}", 0.85),
    (
        re.compile(
            r"\bi(?:'m| am) (?:an? |the )((?:senior |junior |lead |chief |principal )?[\w\-]+ "
            r"(?:analyst|engineer|manager|developer|scientist|investor|student|consultant|director|researcher|trader|architect|officer))\b",
            re.I,
        ),
        "User is a {0}",
        0.85,
    ),
    (re.compile(r"\b(?i:i(?:'m| am) (?:based|located|living) in|i live in) ([A-Z][^.,;!?\n]{1,40})"), "User is based in {0}", 0.8),
    (
        re.compile(r"\bi(?:'m| am) (?:mostly |mainly |very )?interested in ([^.;!?\n]{3,80})", re.I),
        "User is interested in {0}",
        0.75,
    ),
    (re.compile(r"\bi (?:follow|track|cover|focus on) (?!up\b)([^.;!?\n]{3,80})", re.I), "User follows {0}", 0.7),
]

_GREETING = re.compile(r"^(?:hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening))\b[\s,!.]*$", re.IGNORECASE)
_MARKDOWN = re.compile(r"[#*_`>|]+|^\s*[-•]\s+", re.MULTILINE)


def _summarize_answer(text: str, max_chars: int = 400) -> str:
    plain = re.sub(r"\s+", " ", _MARKDOWN.sub(" ", text)).strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", plain) if len(s) > 15]
    summary = " ".join(sentences[:2]) if sentences else plain
    return summary if len(summary) <= max_chars else summary[: max_chars - 3].rstrip() + "..."


class RuleBasedExtractor:
    name = "rules"

    def extract(self, turns: Sequence[ConversationTurn], final_output: str = "") -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        user_turns = [t for t in turns if t.role == "user"]
        question_sentences: list[str] = []  # sentences of the last user turn that are the actual request

        for index, turn in enumerate(user_turns):
            is_last = index == len(user_turns) - 1
            for sentence in (s.strip() for s in _SENTENCES.split(turn.content)):
                if len(sentence) < 3:
                    continue
                found = self._preferences(sentence) + self._facts(sentence)
                candidates.extend(found)
                if is_last and not found and not _GREETING.match(sentence):
                    question_sentences.append(sentence)

        conclusion = self._conclusion(user_turns, final_output, " ".join(question_sentences))
        if conclusion:
            candidates.append(conclusion)
        return candidates

    @staticmethod
    def _preferences(sentence: str) -> list[MemoryCandidate]:
        if not _PREFERENCE_TRIGGER.search(sentence):
            return []
        found: list[MemoryCandidate] = []
        for pattern, key, value, opposite in _PREFERENCE_KEYWORDS:
            match = pattern.search(sentence)
            if not match:
                continue
            prefix = sentence[: match.start()]
            if key == "format" and _SPECIFIC_OBJECT.search(prefix) and not _FORMAT_NOUN.match(sentence[match.end():]):
                continue  # "I want the revenue table" is a one-off request
            # Negation counts only inside the keyword's own clause, close before it.
            clause = _CLAUSE_BREAK.split(prefix)[-1][-30:]
            if _NEGATION.search(clause):
                if opposite is None:
                    continue  # "don't use tables" says nothing about which format IS wanted
                value = opposite
            if any(c.preference_key == key for c in found):
                continue
            found.append(
                MemoryCandidate(
                    # Canonical text only. The user's raw sentence is never stored, because
                    # memories are replayed into the supervisor prompt (stored prompt injection).
                    text=f"User prefers {key.replace('_', ' ')}: {value.replace('_', ' ')}",
                    memory_type=MemoryType.PREFERENCE,
                    confidence=0.85,
                    source=MemorySource.RULE,
                    preference_key=key,
                    preference_value=value,
                )
            )
        return found

    @staticmethod
    def _facts(sentence: str) -> list[MemoryCandidate]:
        found = []
        for pattern, template, confidence in _FACT_RULES:
            match = pattern.search(sentence)
            if match:
                value = match.group(1).strip().rstrip(".")
                found.append(
                    MemoryCandidate(
                        text=template.format(value),
                        memory_type=MemoryType.FACT,
                        confidence=confidence,
                        source=MemorySource.RULE,
                    )
                )
        return found

    @staticmethod
    def _conclusion(
        user_turns: Sequence[ConversationTurn], final_output: str, question: str = ""
    ) -> Optional[MemoryCandidate]:
        if not final_output or len(final_output.strip()) < 20 or not user_turns:
            return None
        # Keep only the request itself (intro, preferences and facts are stored separately),
        # so the conclusion's embedding is about the topic, not about the user.
        question = re.sub(r"\s+", " ", question or user_turns[-1].content).strip()
        if looks_like_injection(question):
            question = "[question omitted]"  # keep the conclusion, drop text aimed at the model
        question = question if len(question) <= 200 else question[:197] + "..."
        return MemoryCandidate(
            text=f"Past question: {question} | Conclusion: {_summarize_answer(final_output)}",
            memory_type=MemoryType.CONCLUSION,
            confidence=0.7,
            source=MemorySource.RULE,
        )


# ============================================================================ LLM
LLM_EXTRACTION_PROMPT = """You extract long-term memories about a user from a conversation.

Return ONLY JSON of the form:
{{"memories": [{{"text": "...", "memory_type": "preference|fact|conclusion",
  "confidence": 0.0-1.0, "preference_key": "format|detail_level|include_citations|...|null",
  "preference_value": "...|null"}}]}}

Rules:
- preference: how the user wants answers (format, length, citations, tone). Set preference_key and preference_value.
- fact: stable facts the user stated about themselves (name, role, company, location, interests).
- conclusion: at most ONE short summary of what the final answer concluded.
- Only include things the USER said or the final answer concluded. Never invent.
- Never include emails, phone numbers, passwords or API keys.
- Write each memory as a short third-person sentence ("User prefers ...").
- If nothing is worth remembering, return {{"memories": []}}.

Conversation:
{conversation}

Final answer:
{final_output}
"""

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMExtractor:
    """Works with any LangChain chat model (or any object with .invoke(prompt) -> text / message)."""

    name = "llm"

    def __init__(self, model: Any) -> None:
        self.model = model

    def extract(self, turns: Sequence[ConversationTurn], final_output: str = "") -> list[MemoryCandidate]:
        conversation = "\n".join(f"{t.role}: {t.content}" for t in turns)[-6000:]
        prompt = LLM_EXTRACTION_PROMPT.format(conversation=conversation, final_output=final_output[-3000:])
        try:
            response = self.model.invoke(prompt)
            content = getattr(response, "content", response)
            payload = json.loads(_JSON_FENCE.sub("", str(content).strip()))
            items = payload.get("memories", []) if isinstance(payload, dict) else payload
            if not isinstance(items, list):
                raise ValueError("'memories' must be a list")
        except Exception as exc:
            raise ExtractionError(f"LLM extraction failed: {exc}") from exc

        candidates = []
        for item in items:
            try:
                candidates.append(MemoryCandidate(**{**item, "source": MemorySource.LLM}))
            except Exception as exc:  # one bad item must not drop the good ones
                logger.warning("Skipping invalid LLM memory %r: %s", item, exc)
        return candidates


class HybridExtractor:
    """LLM first; on any failure fall back to rules."""

    def __init__(self, primary: MemoryExtractor, fallback: MemoryExtractor) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+{fallback.name}"
        self.last_used = fallback.name

    def extract(self, turns: Sequence[ConversationTurn], final_output: str = "") -> list[MemoryCandidate]:
        try:
            result = self.primary.extract(turns, final_output)
            self.last_used = self.primary.name
            return result
        except Exception as exc:
            logger.warning("Primary extractor failed, falling back to %s: %s", self.fallback.name, exc)
            self.last_used = self.fallback.name
            return self.fallback.extract(turns, final_output)


def build_default_extractor(config: MemoryConfig) -> MemoryExtractor:
    """LLM + rules when OPENAI_API_KEY is set and langchain-openai is installed, otherwise rules only."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(model=config.llm_model, temperature=0, timeout=20, max_retries=1)
            return HybridExtractor(LLMExtractor(model), RuleBasedExtractor())
        except Exception as exc:
            logger.warning("LLM extractor unavailable (%s); using rule-based extraction", exc)
    return RuleBasedExtractor()
