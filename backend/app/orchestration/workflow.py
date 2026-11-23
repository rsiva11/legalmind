import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.agents.compliance.agent import ComplianceAgent
from app.agents.compliance.schemas import ComplianceInput
from app.agents.contract.agent import ContractAgent
from app.agents.contract.schemas import ContractAnalysisInput
from app.agents.document_qa.agent import DocumentQAAgent
from app.agents.document_qa.schemas import DocumentQuestionInput
from app.agents.legal_assistant.agent import LegalAssistantAgent
from app.agents.legal_assistant.schemas import LegalQuestionInput
from app.agents.research.agent import ResearchAgent
from app.agents.research.schemas import ResearchInput
from app.agents.risk.agent import RiskAgent
from app.agents.risk.schemas import RiskInput
from app.agents.summary.agent import SummaryAgent, render_markdown
from app.agents.summary.schemas import SummaryInput, SummaryReport
from app.agents.verification.agent import VerificationAgent
from app.agents.verification.schemas import VerificationInput
from app.business_logic.escalation_rules import error_decision, evaluate_escalation
from app.business_logic.recommendation_engine import recommend
from app.guardrails.guardrail_manager import GuardrailManager
from app.knowledge.analysis_retrieval import build_analysis_context
from app.knowledge.rag_pipeline import run_rag
from app.orchestration.execution_context import ExecutionContext
from app.orchestration.state import WorkflowState

logger = logging.getLogger(__name__)


@dataclass
class WorkflowDeps:
    """Everything a workflow calls. Tests replace these with fakes.
    Agents only need a `name` and an async `run(input)` returning an AgentResult."""

    legal: Any = field(default_factory=LegalAssistantAgent)
    research: Any = field(default_factory=ResearchAgent)
    qa: Any = field(default_factory=DocumentQAAgent)
    contract: Any = field(default_factory=ContractAgent)
    compliance: Any = field(default_factory=ComplianceAgent)
    risk: Any = field(default_factory=RiskAgent)
    verification: Any = field(default_factory=VerificationAgent)
    summary: Any = field(default_factory=SummaryAgent)
    guardrails: Any = field(default_factory=GuardrailManager)
    escalate: Callable[[Any], Any] = evaluate_escalation
    recommender: Callable[[Any], Any] = recommend
    rag: Callable[[str, str], Awaitable[Any]] = run_rag
    analysis_context: Callable[[str], Awaitable[Any]] = build_analysis_context


async def _run_agent(state: WorkflowState, agent: Any, build_input: Callable[[], Any], *, fatal: bool = True):
    """Run one agent, record the step, and return its output (None on failure)."""
    try:
        result = await agent.run(build_input())
    except Exception as exc:  # BaseAgent.run never raises, so this is a bug or a bad input
        logger.error("Workflow step %s raised %s", agent.name, type(exc).__name__)
        state.record_error(agent.name, type(exc).__name__)
        return None
    state.record_step(result, fatal=fatal)
    return result.output if result.status == "ok" else None


async def _retrieve(state: WorkflowState, call: Callable[[], Awaitable[Any]]):
    try:
        return await call()
    except Exception as exc:
        logger.error("Workflow retrieval raised %s", type(exc).__name__)
        state.record_error("retrieval", type(exc).__name__)
        return None


def _note_retrieval(state: WorkflowState, context: Any) -> None:
    """Record retrieval guardrail findings, and tell the user when passages were withheld."""
    findings = list(context.guardrail_findings)
    if not findings:
        return
    state.guardrail_findings.extend(findings)
    withheld = sum(1 for f in findings if f.action == "block")
    if withheld:
        state.warnings.append(f"{withheld} retrieved passage(s) were withheld as suspicious and not used")

def _note_research(state: WorkflowState, research: Any) -> None:
    """Record knowledge-retrieval findings, and tell the user when passages were withheld."""
    state.guardrail_findings.extend(research.findings)
    if research.withheld_count:
        state.warnings.append(f"{research.withheld_count} knowledge passage(s) were withheld as suspicious and not used")


def _with_warnings(summary: SummaryReport, warnings: list[str]) -> SummaryReport:
    """Add skipped-step warnings to the report so a missing section can never go unnoticed."""
    if not warnings:
        return summary
    status = summary.status if summary.status == "unverified" else "ok_with_gaps"
    updated = summary.model_copy(
        update={"could_not_confirm": [*summary.could_not_confirm, *warnings], "status": status}
    )
    return updated.model_copy(update={"markdown": render_markdown(updated)})


def _withhold_outputs(state: WorkflowState) -> None:
    """Drop every model-written output. contract and verification (verified, deterministic data) stay."""
    state.legal_answer = None
    state.document_answer = None
    state.summary = None
    state.compliance = None
    state.risk = None


def _sanitize(state: WorkflowState, deps: WorkflowDeps, which: str) -> None:
    """Screen model-written text in the compliance or risk report. Fails closed on any error."""
    report = getattr(state, which)
    if report is None:
        return
    try:
        screen = deps.guardrails.screen_compliance if which == "compliance" else deps.guardrails.screen_risk
        cleaned, findings = screen(report)
    except Exception as exc:
        logger.error("Guardrail %s screening raised %s", which, type(exc).__name__)
        state.record_error(f"guardrail:{which}", type(exc).__name__)
        _withhold_outputs(state)
        return

    setattr(state, which, cleaned)
    state.guardrail_findings.extend(findings)
    replaced = sum(1 for f in findings if f.code == "text_replaced")
    dropped = sum(1 for f in findings if f.code == "candidate_dropped")
    if replaced:
        state.warnings.append(
            f"{replaced} model-written explanation or action text(s) were replaced by an automated safety check"
        )
    if dropped:
        state.warnings.append(f"{dropped} suggested risk(s) were dropped by an automated safety check")


def _screen(state: WorkflowState, deps: WorkflowDeps, context: Any) -> None:
    """Output guardrail. Fails closed: a block or an error withholds every model-written output."""
    if state.status != "running":
        return
    try:
        result = deps.guardrails.check_outputs(state, context)
    except Exception as exc:
        logger.error("Output guardrail raised %s", type(exc).__name__)
        state.record_error("guardrail:output", type(exc).__name__)
        _withhold_outputs(state)
        return

    state.guardrail_findings.extend(result.findings)
    if result.action == "block":
        logger.warning("Output guardrail blocked %s: %s", state.workflow, sorted({f.code for f in result.findings}))
        state.fail("guardrail:output")
        _withhold_outputs(state)
    elif result.action == "flag":
        state.warnings.append(f"Output guardrail flagged {len(result.findings)} item(s) for review")

async def _legal_question(state: WorkflowState, deps: WorkflowDeps) -> None:
    question = state.question
    research = await _run_agent(state, deps.research, lambda: ResearchInput(question=question))
    if research is None:  # the research step failed: never answer from memory
        return
    state.research = research
    _note_research(state, research)
    if research.status == "unavailable":
        state.record_error("research", research.unavailable_reason or "unavailable")
        return
    state.legal_answer = await _run_agent(
        state, deps.legal, lambda: LegalQuestionInput(question=question, knowledge=research)
    )
    _screen(state, deps, None)

async def _document_qa(state: WorkflowState, deps: WorkflowDeps) -> None:
    doc, question = state.document_id, state.question
    rag = await _retrieve(state, lambda: deps.rag(question, doc))
    if rag is None:
        return
    _note_retrieval(state, rag.context)
    state.document_answer = await _run_agent(
        state,
        deps.qa,
        lambda: DocumentQuestionInput(document_id=doc, question=question, context=rag.context),
    )
    _screen(state, deps, rag.context)


async def _full_analysis(state: WorkflowState, deps: WorkflowDeps) -> None:
    doc = state.document_id
    context = await _retrieve(state, lambda: deps.analysis_context(doc))
    if context is None:
        return
    _note_retrieval(state, context)
    if not context.evidence:
        state.record_error("retrieval", "NoEvidence")
        return

    contract = await _run_agent(
        state, deps.contract, lambda: ContractAnalysisInput(document_id=doc, context=context)
    )
    if contract is None:
        return
    state.contract = contract

    state.compliance = await _run_agent(
        state, deps.compliance, lambda: ComplianceInput(document_id=doc, analysis=contract), fatal=False
    )
    _sanitize(state, deps, "compliance")
    if state.status == "failed":
        return
    state.risk = await _run_agent(
        state,
        deps.risk,
        lambda: RiskInput(document_id=doc, analysis=contract, context=context),
        fatal=False,
    )
    _sanitize(state, deps, "risk")
    if state.status == "failed":
        return

    verification = await _run_agent(
        state,
        deps.verification,
        lambda: VerificationInput(
            document_id=doc, context=context, analysis=contract,
            compliance=state.compliance, risk=state.risk,
        ),
    )
    if verification is None:
        return
    state.verification = verification

    summary = await _run_agent(
        state,
        deps.summary,
        lambda: SummaryInput(
            document_id=doc, context=context, analysis=contract, verification=verification,
            compliance=state.compliance, risk=state.risk,
        ),
    )
    if summary is None:
        return
    state.summary = _with_warnings(summary, state.warnings)
    _screen(state, deps, context)


_RUNNERS = {
    "legal_question": _legal_question,
    "document_qa": _document_qa,
    "full_analysis": _full_analysis,
}


async def run_workflow(state: WorkflowState, deps: WorkflowDeps | None = None) -> WorkflowState:
    if deps is None:
        deps = WorkflowDeps()
    ctx = ExecutionContext.begin(state.request_id, state.workflow)
    try:
        await _RUNNERS[state.workflow](state, deps)
    except Exception as exc:  # a workflow bug must never leave the state "running"
        logger.error("Workflow %s crashed: %s", state.workflow, type(exc).__name__)
        state.record_error("workflow", type(exc).__name__)

    try:
        decision = deps.escalate(state)
    except Exception as exc:  # fail closed: a broken check must never read as "no review needed"
        logger.error("Escalation raised %s", type(exc).__name__)
        decision = error_decision()
    state.needs_human_review = decision.needs_human_review
    state.review_codes = decision.codes
    state.review_reasons = decision.reasons

    try:
        state.recommendations = deps.recommender(state)
    except Exception as exc:  # advice text only: a failure is a warning, never a failed job
        logger.error("Recommendation engine raised %s", type(exc).__name__)
        state.recommendations = []
        state.warnings.append("Recommendations could not be generated")
    
    state.duration_ms = ctx.elapsed_ms
    state.complete()
    logger.info(
        "Workflow %s id=%s status=%s steps=%d llm_calls=%d tokens=%d duration_ms=%d",
        state.workflow, state.request_id, state.status, len(state.steps),
        state.total_llm_calls, state.total_tokens, state.duration_ms,
    )
    return state