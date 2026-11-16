import logging
from collections import Counter
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from app.agents.contract.schemas import ContractAnalysis
from app.business_logic.compliance_rules import evaluate_rule
from app.config import get_settings
from app.schemas.risk import Outcome, RiskFinding, RiskRule

logger = logging.getLogger(__name__)

# Compliance-engine status -> risk outcome. Only a failed expectation scores points.
_OUTCOME: dict[str, Outcome] = {
    "non_compliant": "triggered",
    "compliant": "not_triggered",
    "not_found": "unassessed",
    "needs_review": "needs_review",
}


def load_risk_rules(directory: Path | None = None) -> list[RiskRule]:
    base = Path(directory or get_settings().risk_rules_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Risk rules folder not found: {base.resolve()}")

    rules: list[RiskRule] = []
    for path in sorted(p for p in base.iterdir() if p.suffix in {".yaml", ".yml"}):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(data, list):
            raise ValueError(f"{path.name}: expected a list of rules")
        rules.extend(TypeAdapter(list[RiskRule]).validate_python(data))

    duplicates = [i for i, n in Counter(r.id for r in rules).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate rule ids: {duplicates}")
    return rules


def evaluate_risks(analysis: ContractAnalysis, rules: list[RiskRule]) -> list[RiskFinding]:
    findings: list[RiskFinding] = []
    for rule in rules:
        check = evaluate_rule(rule, analysis)
        outcome = _OUTCOME[check.status]
        findings.append(
            RiskFinding(
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                source=rule.source,
                severity=rule.severity,
                points=rule.points if outcome == "triggered" else 0,
                outcome=outcome,
                detail=check.detail,
                evidence=check.evidence,
            )
        )
    logger.info("Risks evaluated: %s", dict(Counter(f.outcome for f in findings)))
    return findings