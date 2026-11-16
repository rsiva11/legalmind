import re
from dataclasses import dataclass

# Verhoeff checksum tables (used by Aadhaar numbers)
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _valid_card(s: str) -> bool:
    d = _digits(s)
    return 13 <= len(d) <= 19 and luhn_valid(d)


def _valid_aadhaar(s: str) -> bool:
    d = _digits(s)
    return len(d) == 12 and d[0] in "23456789" and verhoeff_valid(d)


EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PAN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{3}[PCHFABLJGT][A-Z]\d{4}[A-Z](?![A-Za-z0-9])", re.I)
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)(?![ -]\d{4})")
PHONE = re.compile(r"(?<![\d+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)")

# Order matters: longer, validated numbers claim their span before shorter patterns can
_PATTERNS = [
    ("email", EMAIL, None),
    ("pan", PAN, None),
    ("card", CARD, _valid_card),
    ("aadhaar", AADHAAR, _valid_aadhaar),
    ("phone", PHONE, None),
]


@dataclass(frozen=True)
class PiiMatch:
    kind: str
    start: int
    end: int


def find_pii(text: str) -> list[PiiMatch]:
    found: list[PiiMatch] = []
    for kind, pattern, validate in _PATTERNS:
        for m in pattern.finditer(text):
            if validate is not None and not validate(m.group(0)):
                continue
            if any(m.start() < f.end and f.start < m.end() for f in found):
                continue  # overlaps something already claimed
            found.append(PiiMatch(kind, m.start(), m.end()))
    return sorted(found, key=lambda f: f.start)


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace identifiers with [REDACTED:KIND]. Returns the masked text and a count per kind."""
    counts: dict[str, int] = {}
    masked = text
    for f in reversed(find_pii(text)):
        masked = masked[: f.start] + f"[REDACTED:{f.kind.upper()}]" + masked[f.end :]
        counts[f.kind] = counts.get(f.kind, 0) + 1
    return masked, counts