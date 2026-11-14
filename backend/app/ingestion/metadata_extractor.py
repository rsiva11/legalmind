import re

from app.schemas.document import Chunk, ChunkMetadata

MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
)

DATE_PATTERNS = [
    re.compile(rf"\b{MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\b"),   # March 5, 2026
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS},?\s+\d{{4}}\b"),  # 5th March 2026
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                    # 2026-03-05
    re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b"),        # 05/03/2026
]

MONEY = re.compile(
    r"(?:(?:USD|INR|EUR|GBP|Rs\.?|\$|₹|€|£)\s?\d[\d,]*(?:\.\d+)?"
    r"(?:\s?(?:million|billion|crore|lakh))?)",
    re.I,
)

DURATION = re.compile(
    r"\b(?:\w+\s*)?(?:\(\d+\)\s*)?\d*\s*"
    r"(?:calendar\s+|business\s+|working\s+)?"
    r"(?:days?|weeks?|months?|years?)\b",
    re.I,
)
DURATION_STRICT = re.compile(
    r"\b(?:\d+|\w+\s*\(\d+\))\s+(?:calendar\s+|business\s+|working\s+)?"
    r"(?:days?|weeks?|months?|years?)\b",
    re.I,
)

CLAUSE_KEYWORDS: dict[str, list[str]] = {
    "termination": ["terminat", "expir", "notice period"],
    "confidentiality": ["confidential", "non-disclosure", "nondisclosure"],
    "payment": ["payment", "invoice", "fees", "compensation"],
    "liability": ["liabilit", "limitation of liability", "consequential damages"],
    "indemnity": ["indemnif", "hold harmless"],
    "renewal": ["renew", "auto-renew", "extension of term"],
    "governing_law": ["governing law", "jurisdiction", "governed by the laws"],
    "dispute_resolution": ["arbitration", "dispute resolution", "mediation"],
    "intellectual_property": ["intellectual property", "copyright", "patent"],
    "force_majeure": ["force majeure"],
    "data_protection": ["personal data", "data protection", "gdpr"],
}


def _find_all(patterns: list[re.Pattern], text: str) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(m.group(0).strip() for m in pattern.finditer(text))
    return list(dict.fromkeys(found))  # de-duplicate, keep order


def extract_metadata(text: str) -> ChunkMetadata:
    lower = text.lower()
    clause_types = [
        name for name, words in CLAUSE_KEYWORDS.items()
        if any(w in lower for w in words)
    ]
    return ChunkMetadata(
        clause_types=clause_types,
        dates=_find_all(DATE_PATTERNS, text),
        money=_find_all([MONEY], text),
        durations=_find_all([DURATION_STRICT], text),
    )


def enrich_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [
        c.model_copy(update={"metadata": extract_metadata(c.text)})
        for c in chunks
    ]