import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Severity = Literal["high", "low"]

_HIDDEN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")
_OBJECT = r"(?:instructions?|rules|prompts?|guidelines|directions|constraints|restrictions)"


@dataclass(frozen=True)
class InjectionHit:
    code: str
    severity: Severity


def _rule(code: str, severity: Severity, pattern: str) -> tuple[str, Severity, re.Pattern]:
    return code, severity, re.compile(pattern, re.M)


# Patterns run on normalised text (NFKC, lowercase, invisible characters removed).
# They catch careless attempts. A determined attacker can reword around any list.
_RULES = [
    _rule(
        "instruction_override", "high",
        r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:(?:the|my)\s+)*"
        r"(?:all|any|your|these|those|previous|prior|above|earlier|system)\s+(?:\w+\s+){0,2}" + _OBJECT + r"\b",
    ),
    _rule("chat_template_tokens", "high", r"<\|[a-z_]+\|>|\[/?inst\]|<</?sys>>"),
    _rule("delimiter_tags", "high", r"</?(?:evidence|question|findings|items|task)\b[^>]*>"),
    _rule("developer_mode", "high", r"\b(?:developer mode|jailbreak(?:ed)?|dan mode|do anything now)\b"),
    _rule(
        "prompt_extraction", "low",
        r"\b(?:reveal|show|print|repeat|output|display|leak)\s+(?:me\s+)?(?:your|the)\s+"
        r"(?:system|hidden|initial|original|developer)\s+(?:prompt|message|instructions)"
        r"|\b(?:reveal|show|print|repeat|output|display|leak|what (?:is|are))\s+(?:me\s+)?your\s+"
        r"(?:instructions|rules|prompt|guidelines)",
    ),
    _rule(
        "role_override", "low",
        r"\b(?:you are now|from now on,? you|pretend (?:to be|you are)|act as (?:if|though)|new persona)\b",
    ),
    _rule("role_marker", "low", r"^\s*(?:system|assistant)\s*:"),
]


def prepare(text: str) -> str:
    text = _HIDDEN.sub("", unicodedata.normalize("NFKC", text)).casefold()
    return re.sub(r"[ \t]+", " ", text)


def has_hidden_chars(text: str) -> bool:
    return bool(_HIDDEN.search(text))


def detect_injection(text: str) -> list[InjectionHit]:
    """One hit per pattern group that matches. Never returns the matched text."""
    prepared = prepare(text)
    return [InjectionHit(code, sev) for code, sev, pattern in _RULES if pattern.search(prepared)]