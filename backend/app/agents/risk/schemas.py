from typing import Literal

from pydantic import BaseModel, Field

from app.agents.contract.schemas import ContractAnalysis, EvidenceRef
from app.schemas.document import BuiltContext
from app.schemas.risk import RiskCategory, RiskFinding, RiskRule, RiskScore


class RiskInput(BaseModel):
    document_id: str
    analysis: ContractAnalysis
    context: BuiltContext | None = None  # needed only for candidate risks
    rules: list[RiskRule] | None = None  # None = load from the risk rules folder
    include_candidates: bool = True


class RiskExplanation(BaseModel):
    rule_id: str
    explanation: str = Field(max_length=500)
    suggested_action: str = Field(max_length=300)


class CandidateRisk(BaseModel):
    title: str = Field(max_length=150)
    category: RiskCategory
    description: str = Field(max_length=400)
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=2)
    status: Literal["unverified_suggestion"] = "unverified_suggestion"


class RiskDraft(BaseModel):
    """What the LLM produces: text and suggestions only, never scores or levels."""

    explanations: list[RiskExplanation] = Field(default_factory=list, max_length=20)
    candidates: list[CandidateRisk] = Field(default_factory=list, max_length=5)


class ReviewedRisk(RiskFinding):
    explanation: str | None = None
    suggested_action: str | None = None


class RiskReport(BaseModel):
    document_id: str
    findings: list[ReviewedRisk]
    score: RiskScore
    candidates: list[CandidateRisk]
    candidates_dropped: int
    rules_evaluated: int