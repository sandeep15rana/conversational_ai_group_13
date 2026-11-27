"""
Privacy and safety filters for memory writes (Developer 3 - Rajarshi).

1. mask_pii: Developer 4's input guardrail already sanitizes the query, but
   memories live for a long time, so we mask again before anything reaches disk
   (defence in depth). Deterministic and idempotent.
   Designed for a finance assistant: numbers alone are NOT treated as PII.
   Ambiguous number formats are masked only with a checksum (cards: Luhn,
   Aadhaar: Verhoeff) or a label ("phone", "mobile", "aadhaar", "pin is"...),
   so revenue figures, years and price lists survive intact.
2. looks_like_injection: memories are replayed into the Supervisor prompt in
   later sessions, so text addressed to the model ("ignore your instructions",
   "reveal the system prompt") is never stored. It matches phrasing aimed at the
   model only, so finance language ("act as lender of last resort", "hidden
   liabilities", "override the default tax rules") is not affected.
3. pseudonymize_id: user ids may be emails; only a keyed hash (HMAC-SHA256) of
   the id is stored, never the raw id.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Callable, Union

Replacement = Union[str, Callable[[re.Match[str]], str]]


# ---------------------------------------------------------------- checksums
def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff_ok(digits: str) -> bool:
    check = 0
    for i, ch in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(ch)]]
    return check == 0


# ---------------------------------------------------------------- context-aware replacements
# A phone label, then up to 3 short words ("call me on 98765 43210", "my number is 9876543210").
# A bare "number" is NOT a label: "the total number was 9500000000" is a figure.
_PHONE_LABEL = re.compile(
    r"\b(?:phone|mobile|mob|cell|call|whatsapp|contact|tel|telephone|reach|text|ping|sms|dial"
    r"|(?:my|phone|mobile|contact|cell|whatsapp)\s+(?:number|no\.?))(?:\W+\w{1,4}){0,3}\W*$",
    re.IGNORECASE,
)
_VERSION_LABEL = re.compile(r"\b(?:version|ver\.?|v|release|build)\s*$", re.IGNORECASE)


def _preceding(match: re.Match[str], width: int = 25) -> str:
    return match.string[max(0, match.start() - width) : match.start()]


def _card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[REDACTED_CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else match.group(0)


def _aadhaar(match: re.Match[str]) -> str:
    groups = re.findall(r"\d{4}", match.group(0))
    digits = "".join(groups)
    looks_like_years = any(1900 <= int(g) <= 2099 for g in groups)
    if digits[0] in "01" or looks_like_years or not _verhoeff_ok(digits):
        return match.group(0)  # Aadhaar never starts with 0/1 and always passes Verhoeff
    return "[REDACTED_AADHAAR]"


def _labelled_phone(match: re.Match[str]) -> str:
    return "[REDACTED_PHONE]" if _PHONE_LABEL.search(_preceding(match)) else match.group(0)


def _ip(match: re.Match[str]) -> str:
    return match.group(0) if _VERSION_LABEL.search(_preceding(match, 12)) else "[REDACTED_IP]"


# Order matters: the most specific patterns run first.
_PII_PATTERNS: list[tuple[re.Pattern[str], Replacement]] = [
    # ---- secrets and tokens (labelled or with a distinctive prefix)
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{5,}\.eyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}"), "[REDACTED_TOKEN]"),  # JWT
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-~+/]{12,}=*"), "[REDACTED_TOKEN]"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED_TOKEN]"),  # Slack
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\b(?:password|passcode|passwd|pwd)\s*(?:is|[:=])\s*\S+", re.IGNORECASE), "[REDACTED_SECRET]"),
    (re.compile(r"\b(?:pin|otp|cvv)\s*(?:code\s*)?(?:is|[:=])\s*\d{3,8}\b", re.IGNORECASE), "[REDACTED_SECRET]"),
    (
        re.compile(
            r"\b(?:api[_\- ]?key|access[_\- ]?token|auth[_\- ]?token|refresh[_\- ]?token|secret[_\- ]?key"
            r"|client[_\- ]?secret|token|secret)\s*[:=]\s*[A-Za-z0-9_\-./+=]{8,}",  # credential-like value
            re.IGNORECASE,
        ),
        "[REDACTED_SECRET]",
    ),
    (re.compile(r"\bapi[_\- ]?key\s+is\s+\S+", re.IGNORECASE), "[REDACTED_SECRET]"),
    # ---- email, including "name at domain dot com"
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+\s*(?:\(at\)|\[at\]|\sat\s)\s*[A-Za-z0-9\-]+\s*(?:\(dot\)|\[dot\]|\sdot\s)\s*(?:com|net|org|in|io|co|edu|gov)\b",
            re.IGNORECASE,
        ),
        "[REDACTED_EMAIL]",
    ),
    # ---- government / bank identifiers
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),  # US SSN
    (re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b"), "[REDACTED_IBAN]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[REDACTED_PAN]"),  # Indian PAN
    (re.compile(r"\bpassport\s*(?:no\.?|number|#)?\s*[:\-]?\s*[A-Z]\d{7}\b", re.IGNORECASE), "[REDACTED_PASSPORT]"),
    (re.compile(r"\b(?:aadhaar|aadhar|uid)\s*(?:no\.?|number|#)?\s*(?:is\s*)?[:\-]?\s*\d{4}[ \-]?\d{4}[ \-]?\d{4}\b", re.IGNORECASE), "[REDACTED_AADHAAR]"),
    (re.compile(r"\b\d(?:[ \-]?\d){12,18}\b"), _card),  # cards: only if Luhn-valid
    (re.compile(r"\b\d{4}[ \-]\d{4}[ \-]\d{4}\b"), _aadhaar),  # unlabelled Aadhaar: only if Verhoeff-valid
    # ---- network
    (re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), _ip),
    # ---- dates of birth (only when labelled, so normal dates in answers survive)
    (
        re.compile(r"\b(?:dob|date of birth|born on|birthday is)\s*[:\-]?\s*\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}", re.IGNORECASE),
        "[REDACTED_DOB]",
    ),
    # ---- phone numbers: formats that are unambiguous on their own...
    (re.compile(r"(?<![\w$])\+\d{1,3}[\s.\-]?\(?\d{2,5}\)?(?:[\s.\-]?\d{2,5}){1,3}\b"), "[REDACTED_PHONE]"),  # +CC ...
    (re.compile(r"(?<![\w$.])\(\d{3}\)\s?\d{3}[\s.\-]\d{4}\b"), "[REDACTED_PHONE]"),  # (555) 123-4567
    (re.compile(r"(?<![\w$.])\d{3}-\d{3}-\d{4}\b"), "[REDACTED_PHONE]"),  # 555-123-4567
    (re.compile(r"(?<![\w$.])0\d{2,4}[\s\-]\d{3,4}[\s\-]\d{3,4}\b"), "[REDACTED_PHONE]"),  # 020 7946 0958
    # ...and bare digit groups only when labelled ("call 9876543210", "mobile 98765 43210")
    (re.compile(r"(?<![\w$.,])(?:\d{10}|\d{5}[\s\-]\d{5}|\d{3}[\s.]\d{3}[\s.]\d{4})(?![\d,.]\d)"), _labelled_phone),
]


def mask_pii(text: str) -> str:
    """Replace secrets, emails, ids, card numbers, IPs, labelled DOBs and phone numbers with placeholders."""
    if not text:
        return text
    masked = text
    for pattern, replacement in _PII_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked


def contains_pii(text: str) -> bool:
    """True if mask_pii would change the text."""
    return mask_pii(text) != text


# ---------------------------------------------------------------- prompt injection
_TARGET = r"(?:your|the|my|all|any|previous|prior|above|earlier|initial|original|system|developer'?s?|safety)"
_INSTRUCTIONS = r"(?:instructions?|rules|prompts?|guidelines|directions|directives|guardrails?|constraints|programming)"
_INJECTION_PATTERNS = re.compile(
    # "ignore previous instructions", "disregard your system rules"
    rf"\b(?:ignore|disregard|forget|override|bypass|circumvent)\s+(?:(?:all|any|of)\s+)*(?:{_TARGET}\s+){{1,3}}{_INSTRUCTIONS}\b"
    # "... over your developer's directions", "the system prompt"
    rf"|\b(?:your|the)\s+(?:system|developer'?s?|hidden|initial)\s+(?:prompt|message|instructions?|rules|directions|directives|configuration|config)\b"
    r"|\bsystem\s+prompt\b"
    r"|\byou\s+are\s+now\b|\bfrom\s+now\s+on,?\s+you\s+(?:are|will|must)\b"
    r"|\bpretend\s+(?:to\s+be|you\s+are)\b"
    r"|\bact\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|jailbroken|evil|different\s+ai|dan)\b"
    r"|\b(?:jailbreak|dan\s+mode|developer\s+mode)\b"
    r"|\b(?:reveal|print|dump|leak|repeat|show)\s+(?:me\s+)?(?:(?:your|the)\s+)?(?:hidden|secret)\s+(?:prompt|instructions|configuration|config)\b"
    r"|\btreat\s+(?:my|these)\s+words\s+as\s+(?:top|highest)\s+priority\b",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    """True if the text is addressed to the model as an instruction. Such text is never stored as memory."""
    return bool(text and _INJECTION_PATTERNS.search(text))


# ---------------------------------------------------------------- pseudonymous ids
def pseudonymize_id(user_id: str, key: bytes) -> str:
    """Keyed hash (HMAC-SHA256) of a user id: stable for the same key, not reversible by guessing without it."""
    return "u_" + hmac.new(key, user_id.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
