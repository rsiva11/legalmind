import asyncio

import pytest

from factories import comp_report, good_analysis, make_context, risk_report
from app.agents.base_agent import AgentResult, AgentUsage
from app.agents.document_qa.schemas import DocumentAnswer
from app.agents.legal_assistant.schemas import LegalAnswer
from app.agents.summary.schemas import SummaryReport
from app.agents.verification.schemas import VerificationInput, VerificationReport
from app.config import get_settings
from app.llm.token_manager import get_tracker
from app.orchestration.router import route
from app.orchestration.state import WorkflowRequest, WorkflowState
from app.orchestration.workflow import WorkflowDeps, run_workflow
from app.schemas.document import BuiltContext, RAGResult

CTX = make_context()
MODES = ("contract", "compliance", "risk", "verification", "summary")


class Fake:
    """A stand-in agent. Records inputs and the token budget it saw."""

    def __init__(self, name, output=None, *, status="ok", error=None, tokens=0, calls=0, boom=False):
        self.name, self.output, self.status, self.error = name, output, status, error
        self.tokens, self.calls, self.boom = tokens, calls, boom
        self.inputs, self.budgets = [], []

    async def run(self, data):
        self.inputs.append(data)
        tracker = get_tracker()
        self.budgets.append(tracker.budget if tracker else None)
        if self.boom:
            raise RuntimeError("boom")
        return AgentResult(
            agent=self.name,
            status=self.status,
            output=self.output if self.status == "ok" else None,
            error_type=self.error,
            duration_ms=1,
            usage=AgentUsage(llm_calls=self.calls, prompt_tokens=self.tokens, completion_tokens=0),
        )


class FakeRetrieval:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    async def __call__(self, *args):
        self.calls.append(args)
        if self.error:
            raise self.error
        return self.result


def verification_report(result="pass"):
    return VerificationReport(
        document_id="d1", result=result, items=[], withheld_item_ids=[], withheld_finding_ids=[],
        issues=[], semantic_checked=True, counts={},
    )


def summary_report():
    return SummaryReport(
        document_id="d1", status="ok", verification_result="pass", overview="o", overview_source="template",
        key_facts=[], compliance=[], risk=None, suggestions=[], could_not_confirm=[],
        disclaimer="d", markdown="## Overview\no",
    )


def qa_answer(status="answered"):
    return DocumentAnswer(
        document_id="d1", status=status, answer="a", citations=[], evidence_labels=["E1"], context_truncated=False
    )


def full_deps(**override):
    a = good_analysis()
    parts = dict(
        contract=Fake("contract", a, tokens=100, calls=1),
        compliance=Fake("compliance", comp_report(a), tokens=50, calls=1),
        risk=Fake("risk", risk_report(a), tokens=60, calls=1),
        verification=Fake("verification", verification_report(), tokens=10, calls=1),
        summary=Fake("summary", summary_report(), tokens=20, calls=1),
        analysis_context=FakeRetrieval(CTX),
    )
    parts.update(override)
    return WorkflowDeps(**parts)


def start(**kw):
    req = WorkflowRequest(**kw)
    return WorkflowState.start(req, route(req))


def run(state, deps):
    return asyncio.run(run_workflow(state, deps))


def test_legal_question_workflow():
    answer = LegalAnswer(
        in_scope=True, answer="x", key_points=[], jurisdiction_dependent=False,
        seek_professional_advice=False, disclaimer="d",
    )
    legal = Fake("legal_assistant", answer, tokens=120, calls=1)
    rag, ctxf = FakeRetrieval(), FakeRetrieval()
    state = run(start(question="What is an NDA?"), WorkflowDeps(legal=legal, rag=rag, analysis_context=ctxf))
    assert state.status == "completed" and state.legal_answer == answer
    assert legal.inputs[0].question == "What is an NDA?"
    assert rag.calls == [] and ctxf.calls == []  # a general question never touches documents
    assert legal.budgets == [get_settings().request_token_budget]
    assert (state.total_tokens, state.total_llm_calls) == (120, 1)


def test_full_analysis_happy_path():
    deps = full_deps()
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step, state.warnings) == ("completed", None, [])
    assert [s.agent for s in state.steps] == list(MODES)
    assert (state.total_tokens, state.total_llm_calls) == (240, 5)
    assert state.contract and state.compliance and state.risk and state.verification and state.summary
    assert deps.analysis_context.calls == [("d1",)]
    assert deps.contract.inputs[0].context == CTX
    assert isinstance(deps.verification.inputs[0], VerificationInput)
    assert deps.summary.inputs[0].verification == deps.verification.output
    seen = {b for name in MODES for b in getattr(deps, name).budgets}
    assert seen == {get_settings().analysis_token_budget}  # analysis gets the larger budget


def test_contract_failure_stops_everything():
    deps = full_deps(contract=Fake("contract", status="failed", error="StructuredOutputError"))
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step) == ("failed", "contract")
    assert [s.agent for s in state.steps] == ["contract"]
    for name in MODES[1:]:
        assert getattr(deps, name).inputs == []
    assert state.summary is None


@pytest.mark.parametrize("which", ["compliance", "risk"])
def test_optional_step_failure_is_a_warning(which):
    deps = full_deps(**{which: Fake(which, status="failed", error="TokenBudgetExceeded")})
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    warning = f"{which} step failed (TokenBudgetExceeded); its results are missing"
    assert (state.status, state.warnings) == ("completed", [warning])
    assert [s.status for s in state.steps if s.agent == which] == ["failed"]
    assert len(deps.verification.inputs) == 1 and len(deps.summary.inputs) == 1
    assert getattr(deps.verification.inputs[0], which) is None
    assert state.summary.status == "ok_with_gaps"
    assert warning in state.summary.could_not_confirm
    assert warning in state.summary.markdown  # the missing section is visible to the user


def test_verification_failure_is_fatal():
    deps = full_deps(verification=Fake("verification", status="failed", error="ValueError"))
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step) == ("failed", "verification")
    assert deps.summary.inputs == [] and state.summary is None


def test_summary_crash_fails_the_state():
    deps = full_deps(summary=Fake("summary", boom=True))
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step) == ("failed", "summary")
    assert state.steps[-1].error_type == "RuntimeError"


def test_retrieval_error_fails_before_any_agent():
    deps = full_deps(analysis_context=FakeRetrieval(error=ConnectionError("db down")))
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step) == ("failed", "retrieval")
    assert state.steps[0].error_type == "ConnectionError"
    assert deps.contract.inputs == []


def test_no_evidence_fails():
    empty = BuiltContext(text="", evidence=[], truncated=False)
    deps = full_deps(analysis_context=FakeRetrieval(empty))
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step) == ("failed", "retrieval")
    assert state.steps[0].error_type == "NoEvidence"
    assert deps.contract.inputs == []


def test_document_qa_workflow():
    rag = FakeRetrieval(RAGResult(question="q", document_id="d1", found=True, context=CTX))
    qa = Fake("document_qa", qa_answer(), tokens=80, calls=1)
    state = run(
        start(document_id="d1", question="What is the notice period?", mode="document_qa"),
        WorkflowDeps(qa=qa, rag=rag),
    )
    assert state.status == "completed" and state.document_answer.status == "answered"
    assert rag.calls == [("What is the notice period?", "d1")]
    assert qa.inputs[0].context == CTX and qa.inputs[0].question == "What is the notice period?"
    assert qa.budgets == [get_settings().request_token_budget]
    assert [s.agent for s in state.steps] == ["document_qa"]


def test_unverified_answer_still_completes():
    rag = FakeRetrieval(RAGResult(question="q", document_id="d1", found=True, context=CTX))
    qa = Fake("document_qa", qa_answer("unverified"), calls=1)
    state = run(
        start(document_id="d1", question="What is the notice period?", mode="document_qa"),
        WorkflowDeps(qa=qa, rag=rag),
    )
    assert state.status == "completed" and state.document_answer.status == "unverified"


def test_qa_retrieval_error():
    rag = FakeRetrieval(error=TimeoutError())
    qa = Fake("document_qa", qa_answer())
    state = run(
        start(document_id="d1", question="What is the notice period?", mode="document_qa"),
        WorkflowDeps(qa=qa, rag=rag),
    )
    assert (state.status, state.failed_step) == ("failed", "retrieval")
    assert qa.inputs == [] and state.document_answer is None