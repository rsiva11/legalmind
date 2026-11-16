from pydantic import BaseModel, Field

from app.agents.contract.schemas import ContractAnalysis
from app.schemas.compliance import ComplianceFinding, ComplianceRule


class ComplianceInput(BaseModel):
    document_id: str
    analysis: ContractAnalysis
    rules: list[ComplianceRule] | None = None  # None = load from the policies folder


class FindingExplanation(BaseModel):
    rule_id: str
    explanation: str = Field(max_length=500)
    suggested_action: str = Field(max_length=300)


class ComplianceExplanationDraft(BaseModel):
    """What the LLM produces: text only, never statuses."""

    explanations: list[FindingExplanation] = Field(default_factory=list, max_length=20)


class ReviewedFinding(ComplianceFinding):
    explanation: str | None = None
    suggested_action: str | None = None


class ComplianceReport(BaseModel):
    document_id: str
    findings: list[ReviewedFinding]
    status_counts: dict[str, int]
    rules_evaluated: int