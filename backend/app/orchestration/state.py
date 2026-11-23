import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.base_agent import AgentResult
from app.agents.compliance.schemas import ComplianceReport
from app.agents.contract.schemas import ContractAnalysis
from app.agents.document_qa.schemas import DocumentAnswer
from app.agents.legal_assistant.schemas import LegalAnswer
from app.agents.risk.schemas import RiskReport
from app.agents.summary.schemas import SummaryReport
from app.agents.verification.schemas import VerificationReport

from app.guardrails.result import GuardrailFinding

from app.business_logic.recommendation_engine import Recommendation

from app.agents.research.schemas import ResearchResult

Workflow = Literal["legal_question", "document_qa", "full_analysis"]


class WorkflowRequest(BaseModel):
    question: str | None = Field(None, max_length=2000)
    document_id: str | None = Field(None, min_length=1, max_length=100)
    mode: Workflow | None = None


class RouteDecision(BaseModel):
    workflow: Workflow
    reason: str


class StepRecord(BaseModel):
    agent: str
    status: Literal["ok", "failed"]
    duration_ms: int
    llm_calls: int
    tokens: int
    error_type: str | None = None


class WorkflowState(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    workflow: Workflow
    document_id: str | None = None
    question: str | None = Field(None, exclude=True)  # raw question: used in memory, never stored
    question_masked: str | None = None  # what is stored: personal identifiers removed
    status: Literal["running", "completed", "failed"] = "running"
    failed_step: str | None = None
    steps: list[StepRecord] = []
    warnings: list[str] = []
    guardrail_findings: list[GuardrailFinding] = []
    duration_ms: int = 0

    # Results, filled in by the workflows
    legal_answer: LegalAnswer | None = None
    research: ResearchResult | None = None
    document_answer: DocumentAnswer | None = None
    contract: ContractAnalysis | None = None
    compliance: ComplianceReport | None = None
    risk: RiskReport | None = None
    verification: VerificationReport | None = None
    summary: SummaryReport | None = None

    # Placeholder until Phase 9/10 (escalation and human review)
    needs_human_review: bool = False
    review_reasons: list[str] = []
    review_codes: list[str] = []  # fixed codes behind review_reasons, in a stable order
    recommendations: list[Recommendation] = []

    @classmethod
    def start(cls, request: WorkflowRequest, decision: RouteDecision) -> "WorkflowState":
        question = request.question.strip() if request.question else None
        uses_document = decision.workflow != "legal_question"
        return cls(
            workflow=decision.workflow,
            document_id=request.document_id if uses_document else None,
            question=question,
        )

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.steps)

    @property
    def total_llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.steps)

    def record_step(self, result: AgentResult, *, fatal: bool = True) -> None:
        self.steps.append(
            StepRecord(
                agent=result.agent,
                status=result.status,
                duration_ms=result.duration_ms,
                llm_calls=result.usage.llm_calls,
                tokens=result.usage.prompt_tokens + result.usage.completion_tokens,
                error_type=result.error_type,
            )
        )
        if result.status == "failed":
            if fatal:
                self.fail(result.agent)
            else:
                self.warnings.append(
                    f"{result.agent} step failed ({result.error_type}); its results are missing"
                )

    def record_error(self, step: str, error_type: str) -> None:
        """A step that raised outside an agent (for example retrieval). Always fatal."""
        self.steps.append(
            StepRecord(agent=step, status="failed", duration_ms=0, llm_calls=0, tokens=0, error_type=error_type)
        )
        self.fail(step)

    def fail(self, step: str) -> None:
        self.status = "failed"
        self.failed_step = step

    def complete(self) -> None:
        if self.status == "running":
            self.status = "completed"