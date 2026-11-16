import re
import unicodedata
from typing import TypeVar

T = TypeVar("T")

_PUNCT = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})


def norm(text: str) -> str:
    """Normalise text for quote matching: forgives formatting, never wording."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT).casefold()
    return re.sub(r"\s+", " ", text).strip()


def good_refs(refs: list[T], lookup: dict[str, str]) -> list[T]:
    """Keep refs whose label exists and whose quote appears in that evidence.
    `lookup` maps label -> already-normalised evidence text. Refs need .label and .quote."""
    return [r for r in refs if r.label in lookup and norm(r.quote) in lookup[r.label]]

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "sixty": 60, "ninety": 90,
}
_DIGITS = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_WORD_NUM = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.I)


def numbers_in(text: str) -> set[float]:
    """Every number in the text, written as digits or as a common number word."""
    found = {float(m.group(0).replace(",", "")) for m in _DIGITS.finditer(text)}
    found |= {float(NUMBER_WORDS[m.group(1).lower()]) for m in _WORD_NUM.finditer(text)}
    return found