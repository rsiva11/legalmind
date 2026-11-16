import logging
import re
from collections import Counter
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from app.agents.contract.schemas import ContractAnalysis
from app.config import get_settings
from app.schemas.compliance import ComplianceFinding, ComplianceRule, DurationValue

logger = logging.getLogger(__name__)

NOT_FOUND_DETAIL = (
    "Not found in the retrieved evidence. This does not prove the contract lacks it: "
    "run a targeted search or review manually."
)

# 12 months == 1 year exactly, so equivalent wording compares as equal
_UNIT_DAYS = {"day": 1.0, "week": 7.0, "month": 365.25 / 12, "year": 365.25}
_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "sixty": 60, "ninety": 90,
}
_QUALIFIER = r"(business\s+|working\s+|calendar\s+)?"
_UNIT = r"(day|week|month|year)s?\b"
_DIGITS = re.compile(rf"(\d+(?:\.\d+)?)\)?\s*{_QUALIFIER}{_UNIT}", re.I)
_WORD_NUM = re.compile(rf"\b({'|'.join(_WORDS)})\s+{_QUALIFIER}{_UNIT}", re.I)


def load_rules(directory: Path | None = None) -> list[ComplianceRule]:
    base = Path(directory or get_settings().policies_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Policies folder not found: {base.resolve()}")

    rules: list[ComplianceRule] = []
    for path in sorted(p for p in base.iterdir() if p.suffix in {".yaml", ".yml"}):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(data, list):
            raise ValueError(f"{path.name}: expected a list of rules")
        rules.extend(TypeAdapter(list[ComplianceRule]).validate_python(data))

    duplicates = [i for i, n in Counter(r.id for r in rules).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate rule ids: {duplicates}")
    return rules


def parse_duration_days(text: str) -> tuple[float | None, str]:
    """Return (days, "") or (None, reason). Never guesses."""
    found: set[tuple[float, str, str]] = set()
    for m in _DIGITS.finditer(text):
        found.add((float(m.group(1)), (m.group(2) or "").strip().lower(), m.group(3).lower()))
    for m in _WORD_NUM.finditer(text):
        found.add((float(_WORDS[m.group(1).lower()]), (m.group(2) or "").strip().lower(), m.group(3).lower()))

    if not found:
        return None, "no duration found in the value"
    if len(found) > 1:
        return None, "more than one duration in the value"
    amount, qualifier, unit = next(iter(found))
    if qualifier in ("business", "working"):
        return None, "business or working days cannot be converted exactly"
    return amount * _UNIT_DAYS[unit], ""


def _rule_days(value: DurationValue) -> float:
    return value.amount * _UNIT_DAYS[value.unit.rstrip("s")]


def evaluate_rule(rule: ComplianceRule, analysis: ContractAnalysis) -> ComplianceFinding:
    def make(status, detail, evidence=None) -> ComplianceFinding:
        return ComplianceFinding(
            rule_id=rule.id, rule_name=rule.name, source=rule.source,
            severity=rule.severity, status=status, detail=detail, evidence=evidence or [],
        )

    if rule.check == "clause_present":
        for clause in analysis.clauses:
            if clause.clause_type == rule.clause_type:
                return make("compliant", f"A {rule.clause_type} clause was found", clause.evidence)
        return make("not_found", NOT_FOUND_DETAIL)

    fact = getattr(analysis, rule.field)
    if fact is None:
        return make("not_found", NOT_FOUND_DETAIL)

    if rule.check == "field_present":
        return make("compliant", f"{rule.field} was found", fact.evidence)

    if rule.check == "value_in_list":
        for allowed in rule.value:
            if re.search(rf"\b{re.escape(allowed)}\b", fact.value, re.I):
                return make("compliant", f"Found '{fact.value}', which matches '{allowed}'", fact.evidence)
        return make(
            "non_compliant",
            f"Found '{fact.value}'; approved values are {', '.join(rule.value)}",
            fact.evidence,
        )

    days, reason = parse_duration_days(fact.value)
    if days is None:
        return make("needs_review", f"Could not judge '{fact.value}': {reason}", fact.evidence)

    required = _rule_days(rule.value)
    if rule.check == "duration_at_least":
        ok, word = days >= required - 1e-6, "at least"
    else:
        ok, word = days <= required + 1e-6, "at most"
    detail = f"Found '{fact.value}' ({days:g} days); requires {word} {required:g} days"
    return make("compliant", f"{rule.field.replace('_', ' ')} was found", fact.evidence)


def evaluate(analysis: ContractAnalysis, rules: list[ComplianceRule]) -> list[ComplianceFinding]:
    findings = [evaluate_rule(rule, analysis) for rule in rules]
    logger.info("Compliance evaluated: %s", dict(Counter(f.status for f in findings)))
    return findings