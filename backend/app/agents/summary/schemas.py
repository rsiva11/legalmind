from typing import Literal

from pydantic import BaseModel, Field

from app.agents.compliance.schemas import ComplianceReport
from app.agents.contract.schemas import ContractAnalysis
from app.agents.risk.schemas import RiskReport
from app.agents.verification.schemas import VerificationReport
from app.schemas.document import BuiltContext
from app.schemas.risk import RiskLevel


class SummaryInput(BaseModel):
    document_id: str
    context: BuiltContext
    analysis: ContractAnalysis
    verification: VerificationReport
    compliance: ComplianceReport | None = None
    risk: RiskReport | None = None
    include_overview: bool = True


class OverviewDraft(BaseModel):
    """What the LLM produces. Code validates it before it is shown."""

    overview: str = Field(min_length=20, max_length=700)


class SummaryLine(BaseModel):
    text: str
    citations: list[str] = []  # for example "E2 p.3"


class RiskSummary(BaseModel):
    level: RiskLevel
    total_points: int
    coverage: float
    complete: bool
    note: str
    lines: list[SummaryLine]


class SummaryReport(BaseModel):
    document_id: str
    status: Literal["ok", "ok_with_gaps", "unverified"]
    verification_result: str
    overview: str
    overview_source: Literal["model", "template"]
    key_facts: list[SummaryLine]
    compliance: list[SummaryLine]
    risk: RiskSummary | None
    suggestions: list[SummaryLine]
    could_not_confirm: list[str]
    disclaimer: str
    markdown: str