from typing import Literal

from pydantic import BaseModel, Field

from app.agents.compliance.schemas import ComplianceReport
from app.agents.contract.schemas import ContractAnalysis
from app.agents.risk.schemas import RiskReport
from app.schemas.compliance import ComplianceRule
from app.schemas.document import BuiltContext
from app.schemas.risk import RiskRule

Verdict = Literal["verified", "failed_check", "unsupported", "unclear"]
ItemKind = Literal["fact", "party", "clause", "obligation", "candidate"]
Result = Literal["pass", "pass_with_warnings", "fail"]


class VerificationInput(BaseModel):
    document_id: str
    context: BuiltContext
    analysis: ContractAnalysis
    compliance: ComplianceReport | None = None
    risk: RiskReport | None = None
    compliance_rules: list[ComplianceRule] | None = None  # None = load from the policies folder
    risk_rules: list[RiskRule] | None = None  # None = load from the risk rules folder
    include_semantic: bool = True


class SemanticVerdict(BaseModel):
    item_id: str
    verdict: Literal["supported", "unsupported", "unclear"]
    reason: str = Field(max_length=200)


class SemanticDraft(BaseModel):
    """What the LLM produces. It can only downgrade items, never upgrade them."""

    verdicts: list[SemanticVerdict] = Field(default_factory=list, max_length=60)


class ItemVerification(BaseModel):
    item_id: str
    kind: ItemKind
    label: str
    verdict: Verdict
    reasons: list[str] = []


class VerificationIssue(BaseModel):
    code: str
    severity: Literal["warning", "error"]
    detail: str


class VerificationReport(BaseModel):
    document_id: str
    result: Result
    items: list[ItemVerification]
    withheld_item_ids: list[str]
    withheld_finding_ids: list[str]  # "compliance:EX-001" / "risk:RK-002"
    issues: list[VerificationIssue]
    semantic_checked: bool
    counts: dict[str, int]