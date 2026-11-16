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