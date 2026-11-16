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
from app.agents.risk.agent import RiskAgent
from app.agents.risk.schemas import RiskInput
from app.agents.summary.agent import SummaryAgent, render_markdown
from app.agents.summary.schemas import SummaryInput, SummaryReport
from app.agents.verification.agent import VerificationAgent
from app.agents.verification.schemas import VerificationInput
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
    qa: Any = field(default_factory=DocumentQAAgent)
    contract: Any = field(default_factory=ContractAgent)
    compliance: Any = field(default_factory=ComplianceAgent)
    risk: Any = field(default_factory=RiskAgent)
    verification: Any = field(default_factory=VerificationAgent)
    summary: Any = field(default_factory=SummaryAgent)
    guardrails: Any = field(default_factory=GuardrailManager)
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
    state.legal_answer = await _run_agent(
        state, deps.legal, lambda: LegalQuestionInput(question=state.question)
    )
    _screen(state, deps, None)


async def _document_qa(state: WorkflowState, deps: WorkflowDeps) -> None:
    doc, question = state.document_id, state.question
    rag = await _retrieve(state, lambda: deps.rag(question, doc))
    if rag is None:
        return
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
    if state.status == "failed":
        return
    state.risk = await _run_agent(
        state,
        deps.risk,
        lambda: RiskInput(document_id=doc, analysis=contract, context=context),
        fatal=False,
    )
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

    state.duration_ms = ctx.elapsed_ms
    state.complete()
    logger.info(
        "Workflow %s id=%s status=%s steps=%d llm_calls=%d tokens=%d duration_ms=%d",
        state.workflow, state.request_id, state.status, len(state.steps),
        state.total_llm_calls, state.total_tokens, state.duration_ms,
    )
    return state