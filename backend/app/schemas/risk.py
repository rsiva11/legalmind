from typing import Literal

from pydantic import BaseModel, Field

from app.agents.contract.schemas import EvidenceRef
from app.schemas.compliance import ComplianceRule, Severity

RiskCategory = Literal["legal", "financial", "operational", "compliance", "business"]
Outcome = Literal["triggered", "not_triggered", "unassessed", "needs_review"]
RiskLevel = Literal["not_assessed", "low", "medium", "high", "critical"]


class RiskRule(ComplianceRule):
    category: RiskCategory
    points: int = Field(gt=0, le=100)


class RiskFinding(BaseModel):
    rule_id: str
    rule_name: str
    category: RiskCategory
    source: str
    severity: Severity
    points: int  # points counted: the rule's points if triggered, else 0
    outcome: Outcome
    detail: str
    evidence: list[EvidenceRef] = []


class RiskScore(BaseModel):
    total_points: int
    level: RiskLevel
    triggered: int
    not_triggered: int
    unassessed: int
    needs_review: int
    rules_total: int
    coverage: float  # share of rules that could be assessed
    complete: bool  # True only if every rule was assessed
    by_category: dict[str, int]