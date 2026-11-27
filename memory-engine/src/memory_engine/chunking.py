"""
Sentence-aware chunking (Developer 3 - Rajarshi).

Why: all-MiniLM-L6-v2 silently ignores everything after 256 word-pieces, so a
long memory must be split or its tail is lost. Rules:
  1. Short text (<= max_tokens) stays as one chunk.
  2. Long text is split on sentence boundaries into chunks of <= max_tokens.
  3. Consecutive chunks share `overlap_sentences` sentences, so meaning that
     spans a boundary is not lost.
  4. A single sentence longer than max_tokens is split on word boundaries, and a
     single "word" longer than that (CJK text, URLs) is cut by characters.
Sentence ends include CJK (。！？) and Devanagari (।) punctuation.
"""

from __future__ import annotations

import re
from typing import Callable

TokenCounter = Callable[[str], int]

_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])"  # Latin sentence end
    r"|(?<=[。！？।॥])\s*"  # CJK 。！？ and Devanagari danda । ॥ (no space needed)
    r"|\n{2,}|\n(?=\s*[-*•]\s)"  # paragraphs and bullet lines
)


def split_sentences(text: str) -> list[str]:
    """Split on sentence endings, blank lines and bullet lines. Keeps the sentence text intact."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip())]
    return [p for p in parts if p]


def _hard_split(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Last resort for text with no spaces (CJK, long URLs, base64): cut by characters."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        # Binary search for the longest prefix that fits.
        lo, hi = start + 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_tokens(text[start:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        pieces.append(text[start:lo])
        start = lo
    return pieces


def _split_long_sentence(sentence: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    words = sentence.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        if count_tokens(word) > max_tokens:  # a single "word" too big for one chunk
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.extend(_hard_split(word, count_tokens, max_tokens))
            continue
        candidate = " ".join(current + [word])
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    text: str,
    count_tokens: TokenCounter,
    max_tokens: int = 200,
    overlap_sentences: int = 1,
) -> list[str]:
    """Return chunks of at most max_tokens tokens, with sentence overlap between neighbours."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    # Break into sentence units, splitting any oversized sentence.
    units: list[str] = []
    for sentence in split_sentences(text):
        if count_tokens(sentence) > max_tokens:
            units.extend(_split_long_sentence(sentence, count_tokens, max_tokens))
        else:
            units.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = " ".join(current + [unit])
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(" ".join(current))
            # Carry the last N sentences forward as overlap, if they still fit with the new unit.
            overlap = current[-overlap_sentences:] if overlap_sentences else []
            while overlap and count_tokens(" ".join(overlap + [unit])) > max_tokens:
                overlap = overlap[1:]
            current = overlap + [unit]
        else:
            current.append(unit)
    if current:  # always holds at least one new unit, never pure overlap
        chunks.append(" ".join(current))
    return chunks
