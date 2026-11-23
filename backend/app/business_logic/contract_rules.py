import calendar
import re
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel

from app.agents.contract.schemas import ContractAnalysis
from app.utils.text_utils import NUMBER_WORDS
from app.utils.time_utils import utc_now

DateKind = Literal["ok", "none", "ambiguous", "invalid", "multiple"]
ExpiryStatus = Literal["expired", "expiring_30", "expiring_60", "expiring_90", "active", "unresolved"]
ExpiryBasis = Literal["stated_date", "effective_plus_term"]

EXPIRY_WINDOWS = ((30, "expiring_30"), (60, "expiring_60"), (90, "expiring_90"))
RENEWAL_NOTE = (
    "A renewal clause was found. Renewal dates and notice windows are not computed; review them manually."
)

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)
_DMY_TEXT = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_NAMES})\b\.?,?\s+(\d{{4}})(?!\d)", re.I
)
_MDY_TEXT = re.compile(rf"\b({_NAMES})\b\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})(?!\d)", re.I)
_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_NUMERIC = re.compile(r"(?<!\d)(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?!\d)")

_QUALIFIER = r"(business\s+|working\s+|calendar\s+)?"
_UNIT = r"(day|week|month|year)s?\b"
_DUR_DIGITS = re.compile(rf"(?<![\d.])(\d+(?:\.\d+)?)\)?\s*{_QUALIFIER}{_UNIT}", re.I)
_DUR_WORDS = re.compile(rf"\b({'|'.join(NUMBER_WORDS)})\s+{_QUALIFIER}{_UNIT}", re.I)


def _from_parts(year: int, month: int, day: int) -> date | None:
    if not 1900 <= year <= 2200:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date_text(text: str) -> tuple[DateKind, date | None]:
    """Find the one date in a piece of text. Never guesses: an ambiguous numeric date is reported as such."""
    found: set[date] = set()
    flags = {"ambiguous": False, "invalid": False}

    def add(d: date | None) -> None:
        if d is None:
            flags["invalid"] = True
        else:
            found.add(d)

    for m in _DMY_TEXT.finditer(text):
        add(_from_parts(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1))))
    for m in _MDY_TEXT.finditer(text):
        add(_from_parts(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2))))
    for m in _ISO.finditer(text):
        add(_from_parts(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _NUMERIC.finditer(text):
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a == b or (a > 12 and b <= 12):
            add(_from_parts(year, b, a))  # day first
        elif b > 12 and a <= 12:
            add(_from_parts(year, a, b))  # month first
        elif a <= 12 and b <= 12:
            flags["ambiguous"] = True  # 05/03/2026 could be 5 March or 3 May
        else:
            add(None)

    if flags["ambiguous"]:
        return "ambiguous", None
    if flags["invalid"]:
        return "invalid", None
    if len(found) > 1:
        return "multiple", None
    if found:
        return "ok", next(iter(found))
    return "none", None


def _parse_term(text: str) -> tuple[str, object]:
    """Returns ("date", date), ("duration", (amount, unit)) or ("error", reason_code)."""
    kind, parsed = parse_date_text(text)
    if kind == "ambiguous":
        return "error", "term_ambiguous_date"
    if kind == "invalid":
        return "error", "term_invalid_date"
    if kind == "multiple":
        return "error", "term_multiple_dates"

    durations = {
        (float(m.group(1)), (m.group(2) or "").strip().lower(), m.group(3).lower())
        for m in _DUR_DIGITS.finditer(text)
    } | {
        (float(NUMBER_WORDS[m.group(1).lower()]), (m.group(2) or "").strip().lower(), m.group(3).lower())
        for m in _DUR_WORDS.finditer(text)
    }
    if kind == "ok":
        return ("error", "term_mixed_date_and_duration") if durations else ("date", parsed)
    if not durations:
        return "error", "term_unparseable"
    if len(durations) > 1:
        return "error", "term_multiple_durations"

    amount, qualifier, unit = next(iter(durations))
    if qualifier in ("business", "working"):
        return "error", "term_business_days"
    if amount <= 0:
        return "error", "term_unparseable"
    if amount != int(amount):
        return "error", "term_fractional"
    return "duration", (int(amount), unit)


def _add_months(d: date, months: int) -> date:
    year_offset, month_index = divmod(d.month - 1 + months, 12)
    year, month = d.year + year_offset, month_index + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))  # clamp month ends


def _add_term(start: date, amount: int, unit: str) -> date:
    if unit == "day":
        return start + timedelta(days=amount)
    if unit == "week":
        return start + timedelta(weeks=amount)
    if unit == "month":
        return _add_months(start, amount)
    return _add_months(start, 12 * amount)


def _status(days: int | None) -> ExpiryStatus:
    if days is None:
        return "unresolved"
    if days < 0:
        return "expired"
    for limit, name in EXPIRY_WINDOWS:
        if days <= limit:
            return name  # type: ignore[return-value]
    return "active"


def _fact_text(analysis: ContractAnalysis, field: str, verified: set[str] | None) -> tuple[str | None, str | None]:
    fact = getattr(analysis, field)
    if fact is None:
        return None, "missing"
    if verified is not None and f"fact:{field}" not in verified:
        return None, "not_verified"
    return fact.value, None


class ContractDates(BaseModel):
    as_of: date
    effective_date: date | None = None
    expiry_date: date | None = None
    expiry_basis: ExpiryBasis | None = None
    days_remaining: int | None = None
    status: ExpiryStatus
    not_yet_started: bool = False
    has_renewal_clause: bool = False
    renewal_note: str | None = None
    reasons: list[str] = []  # fixed codes, never contract text


def analyze_dates(
    analysis: ContractAnalysis, today: date | None = None, verified: set[str] | None = None
) -> ContractDates:
    """Pure date logic. `verified` is the set of verified item ids from the Verification report;
    when given, facts and clauses outside it are ignored."""
    today = today or utc_now().date()
    reasons: list[str] = []

    effective: date | None = None
    text, problem = _fact_text(analysis, "effective_date", verified)
    if problem:
        reasons.append(f"effective_date_{problem}")
    else:
        kind, parsed = parse_date_text(text)
        if kind == "ok":
            effective = parsed
        else:
            reasons.append("effective_date_unparseable" if kind == "none" else f"effective_date_{kind}")

    expiry: date | None = None
    basis: ExpiryBasis | None = None
    text, problem = _fact_text(analysis, "term_or_expiry", verified)
    if problem:
        reasons.append(f"term_{problem}")
    else:
        kind, value = _parse_term(text)
        if kind == "error":
            reasons.append(str(value))
        elif kind == "date":
            expiry, basis = value, "stated_date"  # type: ignore[assignment]
        elif effective is None:
            reasons.append("effective_date_needed_for_term")
        else:
            try:
                expiry, basis = _add_term(effective, *value), "effective_plus_term"  # type: ignore[misc]
            except (ValueError, OverflowError):
                reasons.append("term_out_of_range")

    if expiry is not None and effective is not None and expiry < effective:
        reasons.append("expiry_before_effective")
        expiry = basis = None

    days = (expiry - today).days if expiry is not None else None
    renewal = any(
        c.clause_type == "renewal" and (verified is None or f"clause:{i}" in verified)
        for i, c in enumerate(analysis.clauses)
    )
    return ContractDates(
        as_of=today,
        effective_date=effective,
        expiry_date=expiry,
        expiry_basis=basis,
        days_remaining=days,
        status=_status(days),
        not_yet_started=effective is not None and effective > today,
        has_renewal_clause=renewal,
        renewal_note=RENEWAL_NOTE if renewal else None,
        reasons=reasons,
    )