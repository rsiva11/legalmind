import json
from datetime import date

import pytest

from factories import fact, good_analysis, ref
from app.agents.contract.schemas import ExtractedClause
from app.business_logic.contract_rules import RENEWAL_NOTE, analyze_dates, parse_date_text

TODAY = date(2026, 6, 1)


def dates(effective, term, *, verified=None, today=TODAY, **kw):
    a = good_analysis(
        effective_date=fact(effective) if effective else None,
        term_or_expiry=fact(term) if term else None,
        **kw,
    )
    return analyze_dates(a, today, verified)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1 March 2026", date(2026, 3, 1)),
        ("1st March 2026", date(2026, 3, 1)),
        ("March 1, 2026", date(2026, 3, 1)),
        ("Sept 5 2026", date(2026, 9, 5)),
        ("effective as of 5 January 2027, unless ended", date(2027, 1, 5)),
        ("2026-03-01", date(2026, 3, 1)),
        ("13/03/2026", date(2026, 3, 13)),
        ("03/13/2026", date(2026, 3, 13)),
        ("05/05/2026", date(2026, 5, 5)),
        ("15.08.2026", date(2026, 8, 15)),
        ("1 March 2026 and 1st March 2026", date(2026, 3, 1)),
        ("the 3rd of April 2026", date(2026, 4, 3)),
    ],
)
def test_parse_date_ok(text, expected):
    assert parse_date_text(text) == ("ok", expected)


@pytest.mark.parametrize(
    "text, kind",
    [
        ("05/03/2026", "ambiguous"),
        ("05/03/2026 and 13/03/2026", "ambiguous"),
        ("31 February 2026", "invalid"),
        ("2026-13-01", "invalid"),
        ("1 March 1850", "invalid"),
        ("14/15/2026", "invalid"),
        ("1 March 2026 and 5 April 2026", "multiple"),
        ("no date here", "none"),
        ("", "none"),
    ],
)
def test_parse_date_other(text, kind):
    assert parse_date_text(text) == (kind, None)


@pytest.mark.parametrize(
    "term, days, status",
    [
        ("31 May 2026", -1, "expired"),
        ("1 June 2026", 0, "expiring_30"),
        ("1 July 2026", 30, "expiring_30"),
        ("2 July 2026", 31, "expiring_60"),
        ("31 July 2026", 60, "expiring_60"),
        ("1 August 2026", 61, "expiring_90"),
        ("30 August 2026", 90, "expiring_90"),
        ("31 August 2026", 91, "active"),
    ],
)
def test_expiry_windows(term, days, status):
    d = dates("1 January 2026", term)
    assert (d.days_remaining, d.status, d.expiry_basis, d.reasons) == (days, status, "stated_date", [])


@pytest.mark.parametrize(
    "effective, term, expected",
    [
        ("1 March 2026", "twelve (12) months", date(2027, 3, 1)),
        ("31 January 2026", "one (1) month", date(2026, 2, 28)),
        ("29 February 2024", "1 year", date(2025, 2, 28)),
        ("1 March 2026", "three (3) years", date(2029, 3, 1)),
        ("1 March 2026", "30 days", date(2026, 3, 31)),
        ("1 March 2026", "2 weeks", date(2026, 3, 15)),
        ("31 March 2026", "six (6) months", date(2026, 9, 30)),
    ],
)
def test_duration_arithmetic(effective, term, expected):
    d = dates(effective, term)
    assert (d.expiry_date, d.expiry_basis, d.reasons) == (expected, "effective_plus_term", [])


def test_standard_term():
    d = dates("1 March 2026", "twelve (12) months")
    assert (d.effective_date, d.expiry_date) == (date(2026, 3, 1), date(2027, 3, 1))
    assert (d.days_remaining, d.status, d.reasons, d.not_yet_started, d.as_of) == (273, "active", [], False, TODAY)


def test_expiring_today():
    d = dates("1 March 2026", "3 months")
    assert (d.expiry_date, d.days_remaining, d.status) == (date(2026, 6, 1), 0, "expiring_30")


def test_expired_by_duration():
    d = dates("1 March 2026", "2 months")
    assert (d.expiry_date, d.days_remaining, d.status) == (date(2026, 5, 1), -31, "expired")


def test_contract_that_has_not_started():
    d = dates("1 September 2026", "twelve (12) months")
    assert d.not_yet_started is True
    assert (d.expiry_date, d.days_remaining, d.status) == (date(2027, 9, 1), 457, "active")


@pytest.mark.parametrize(
    "effective, term, reasons",
    [
        ("05/03/2026", "12 months", ["effective_date_ambiguous", "effective_date_needed_for_term"]),
        ("1 March 2026", "05/03/2026", ["term_ambiguous_date"]),
        ("1 March 2026", "ten business days", ["term_business_days"]),
        ("1 March 2026", "three years from 1 March 2026", ["term_mixed_date_and_duration"]),
        ("1 March 2026", "12 months, extendable by 6 months", ["term_multiple_durations"]),
        ("1 March 2026", "1.5 years", ["term_fractional"]),
        ("1 March 2026", "perpetual", ["term_unparseable"]),
        ("1 March 2026", "1 January 2026", ["expiry_before_effective"]),
        ("1 March 2026", "31 February 2027", ["term_invalid_date"]),
        ("1 March 2026", "5 April 2026 or 6 May 2026", ["term_multiple_dates"]),
        ("nothing here", "12 months", ["effective_date_unparseable", "effective_date_needed_for_term"]),
        ("1 March 2026 or 2 March 2026", "12 months", ["effective_date_multiple", "effective_date_needed_for_term"]),
        ("1 March 2026", "99999 years", ["term_out_of_range"]),
    ],
)
def test_unresolved_cases(effective, term, reasons):
    d = dates(effective, term)
    assert d.reasons == reasons
    assert (d.status, d.expiry_date, d.expiry_basis, d.days_remaining) == ("unresolved", None, None, None)


def test_missing_facts():
    d = dates(None, None)
    assert d.reasons == ["effective_date_missing", "term_missing"]
    assert (d.status, d.expiry_date, d.effective_date) == ("unresolved", None, None)


def test_stated_expiry_without_an_effective_date():
    d = dates(None, "15 July 2026")
    assert d.reasons == ["effective_date_missing"]
    assert (d.expiry_date, d.days_remaining, d.status, d.expiry_basis) == (
        date(2026, 7, 15), 44, "expiring_60", "stated_date",
    )


def test_unverified_facts_are_ignored():
    d = dates("1 March 2026", "twelve (12) months", verified=set())
    assert d.reasons == ["effective_date_not_verified", "term_not_verified"] and d.status == "unresolved"

    ok = dates("1 March 2026", "twelve (12) months", verified={"fact:effective_date", "fact:term_or_expiry"})
    assert (ok.reasons, ok.expiry_date) == ([], date(2027, 3, 1))

    partial = dates("1 March 2026", "15 July 2026", verified={"fact:term_or_expiry"})
    assert partial.reasons == ["effective_date_not_verified"]
    assert (partial.expiry_date, partial.status) == (date(2026, 7, 15), "expiring_60")


def test_renewal_clause_is_flagged_not_calculated():
    renewal = ExtractedClause(
        clause_type="renewal", summary="Renews automatically", evidence=[ref("E2", "renews automatically")]
    )
    d = dates("1 March 2026", "twelve (12) months", clauses=[renewal])
    assert (d.has_renewal_clause, d.renewal_note) == (True, RENEWAL_NOTE)
    assert d.expiry_date == date(2027, 3, 1)  # the stated term is untouched

    plain = dates("1 March 2026", "twelve (12) months")
    assert (plain.has_renewal_clause, plain.renewal_note) == (False, None)

    facts_only = {"fact:effective_date", "fact:term_or_expiry"}
    assert dates("1 March 2026", "twelve (12) months", clauses=[renewal], verified=facts_only).has_renewal_clause is False
    assert dates("1 March 2026", "twelve (12) months", clauses=[renewal], verified=facts_only | {"clause:0"}).has_renewal_clause is True


def test_result_is_json_safe():
    d = dates("1 March 2026", "twelve (12) months")
    dumped = json.loads(json.dumps(d.model_dump(mode="json")))
    assert dumped["expiry_date"] == "2027-03-01" and dumped["as_of"] == "2026-06-01"
    assert dumped["status"] == "active" and dumped["reasons"] == []