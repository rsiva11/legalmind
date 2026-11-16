from factories import good_analysis, make_context, risk_report
from test_workflow import CTX, Fake, FakeRetrieval, full_deps, run, start
from app.agents.document_qa.schemas import DocumentAnswer
from app.agents.legal_assistant.schemas import LegalAnswer
from app.guardrails.result import GuardrailResult
from app.orchestration.workflow import WorkflowDeps
from app.schemas.document import Citation, RAGResult


class Counting:
    """An allow-everything guardrail that records how it was called."""

    def __init__(self):
        self.seen = []

    def check_outputs(self, state, context):
        self.seen.append(context)
        return GuardrailResult(guardrail="output")


class Exploding:
    def check_outputs(self, state, context):
        raise RuntimeError("boom")


def legal_answer(text):
    return LegalAnswer(
        in_scope=True, answer=text, key_points=[], jurisdiction_dependent=False,
        seek_professional_advice=False, disclaimer="d",
    )


def test_clean_run_screens_once_with_the_context():
    guard = Counting()
    state = run(start(document_id="d1", mode="full_analysis"), full_deps(guardrails=guard))
    assert (state.status, state.guardrail_findings, state.warnings) == ("completed", [], [])
    assert guard.seen == [CTX]


def test_blocked_output_fails_the_job_and_withholds_text():
    bad = risk_report(good_analysis())
    first = bad.findings[0].model_copy(update={"explanation": "This payment term is illegal."})
    bad = bad.model_copy(update={"findings": [first, *bad.findings[1:]]})
    deps = full_deps(risk=Fake("risk", bad, tokens=60, calls=1))
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step) == ("failed", "guardrail:output")
    assert state.summary is None and state.risk is None and state.compliance is None
    assert state.contract is not None and state.verification is not None
    assert [(f.code, f.field) for f in state.guardrail_findings] == [("legal_conclusion", "risk.RK-001.explanation")]
    assert len(deps.summary.inputs) == 1  # the pipeline finished; only the output was withheld


def test_flagged_output_completes_with_a_warning():
    ctx = make_context(["Disputes go to Delhi and the court will have exclusive jurisdiction."])
    rag = FakeRetrieval(RAGResult(question="q", document_id="d1", found=True, context=ctx))
    answer = DocumentAnswer(
        document_id="d1", status="answered",
        answer="Disputes go to Delhi, where the court will have exclusive jurisdiction. [E1]",
        citations=[Citation(label="E1", page_number=1, chunk_index=1, document_id="d")],
        evidence_labels=["E1"], context_truncated=False,
    )
    qa = Fake("document_qa", answer, calls=1)
    state = run(
        start(document_id="d1", question="Which court?", mode="document_qa"),
        WorkflowDeps(qa=qa, rag=rag),
    )
    assert state.status == "completed" and state.document_answer is not None
    assert [f.action for f in state.guardrail_findings] == ["flag"]
    assert state.warnings == ["Output guardrail flagged 1 item(s) for review"]


def test_guardrail_error_fails_closed():
    legal = Fake("legal_assistant", legal_answer("A notice period is the time to give notice."), calls=1)
    state = run(start(question="What is a notice period?"), WorkflowDeps(legal=legal, guardrails=Exploding()))
    assert (state.status, state.failed_step) == ("failed", "guardrail:output")
    assert state.legal_answer is None
    assert (state.steps[-1].agent, state.steps[-1].error_type) == ("guardrail:output", "RuntimeError")


def test_guardrail_not_run_after_a_failure():
    guard = Counting()
    deps = full_deps(contract=Fake("contract", status="failed", error="StructuredOutputError"), guardrails=guard)
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step) == ("failed", "contract")
    assert guard.seen == []