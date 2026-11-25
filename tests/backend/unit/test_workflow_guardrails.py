from factories import good_analysis, make_context, ref, risk_report
from test_workflow import CTX, Fake, FakeRetrieval, full_deps, run, start, summary_report
from app.agents.document_qa.schemas import DocumentAnswer
from app.agents.legal_assistant.schemas import LegalAnswer
from app.agents.risk.schemas import CandidateRisk
from app.guardrails.output_guardrail import EXPLANATION_FALLBACK
from app.guardrails.result import GuardrailFinding, GuardrailResult
from app.orchestration.workflow import WorkflowDeps
from app.schemas.document import Citation, RAGResult

from app.agents.research.schemas import ResearchResult

WITHHELD = "1 retrieved passage(s) were withheld as suspicious and not used"
REPLACED = "1 model-written explanation or action text(s) were replaced by an automated safety check"
DROPPED = "1 suggested risk(s) were dropped by an automated safety check"


class Counting:
    """An allow-everything guardrail that records how it was called."""

    def __init__(self):
        self.seen = []

    def screen_compliance(self, report):
        return report, []

    def screen_risk(self, report):
        return report, []

    def check_outputs(self, state, context):
        self.seen.append(context)
        return GuardrailResult(guardrail="output")


class Exploding:
    def check_outputs(self, state, context):
        raise RuntimeError("boom")


class ScreenBoom(Counting):
    def screen_risk(self, report):
        raise RuntimeError("boom")


def legal_answer(text):
    return LegalAnswer(
        in_scope=True, answer=text, key_points=[], jurisdiction_dependent=False,
        seek_professional_advice=False, disclaimer="d",
    )


def withheld_finding():
    return GuardrailFinding(
        guardrail="retrieval", action="block", code="suspicious_content", field="chunk[3]",
        detail="Withheld from the model; matched: instruction_override",
    )


def test_clean_run_screens_once_with_the_context():
    guard = Counting()
    state = run(start(document_id="d1", mode="full_analysis"), full_deps(guardrails=guard))
    assert (state.status, state.guardrail_findings, state.warnings) == ("completed", [], [])
    assert guard.seen == [CTX]


def test_bad_explanation_is_replaced_and_the_job_completes():
    bad = risk_report(good_analysis())
    first = bad.findings[0].model_copy(update={"explanation": "This payment term is illegal."})
    bad = bad.model_copy(update={"findings": [first, *bad.findings[1:]]})
    deps = full_deps(risk=Fake("risk", bad, tokens=60, calls=1))
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.warnings) == ("completed", [REPLACED])
    assert state.risk.findings[0].explanation == EXPLANATION_FALLBACK
    assert state.risk.score == bad.score and state.risk.findings[0].outcome == bad.findings[0].outcome
    assert deps.verification.inputs[0].risk.findings[0].explanation == EXPLANATION_FALLBACK
    assert [(f.code, f.field) for f in state.guardrail_findings] == [("text_replaced", "risk.RK-001.explanation")]
    assert REPLACED in state.summary.could_not_confirm
    assert "illegal" not in state.model_dump_json()


def test_summary_overview_with_an_unknown_citation_is_blocked():
    bad = summary_report().model_copy(update={"overview": "Notice is 30 days [E9].", "overview_source": "model"})
    deps = full_deps(summary=Fake("summary", bad, tokens=20, calls=1))
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step) == ("failed", "guardrail:output")
    assert state.summary is None and state.risk is None and state.compliance is None
    assert state.contract is not None and state.verification is not None
    assert [(f.code, f.field) for f in state.guardrail_findings] == [("unknown_citation", "summary.overview")]


def test_bad_candidate_is_dropped_with_a_warning():
    cand = CandidateRisk(
        title="Unlimited liability", category="legal",
        description="The liability clause is void and unlawful.",
        evidence=[ref("E1", "liability shall be unlimited")],
    )
    rr = risk_report(good_analysis()).model_copy(update={"candidates": [cand]})
    state = run(start(document_id="d1", mode="full_analysis"), full_deps(risk=Fake("risk", rr, tokens=60, calls=1)))
    assert (state.status, state.warnings) == ("completed", [DROPPED])
    assert state.risk.candidates == []
    assert [f.code for f in state.guardrail_findings] == ["candidate_dropped"]


def test_screening_error_fails_closed():
    deps = full_deps(guardrails=ScreenBoom())
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step) == ("failed", "guardrail:risk")
    assert state.risk is None and state.compliance is None and state.contract is not None
    assert deps.verification.inputs == [] and deps.summary.inputs == []
    assert (state.steps[-1].agent, state.steps[-1].error_type) == ("guardrail:risk", "RuntimeError")


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
    research = Fake("research", ResearchResult(status="not_found"))
    legal = Fake("legal_assistant", legal_answer("A notice period is the time to give notice."), calls=1)
    state = run(
        start(question="What is a notice period?"),
        WorkflowDeps(research=research, legal=legal, guardrails=Exploding()),
    )
    assert (state.status, state.failed_step) == ("failed", "guardrail:output")
    assert state.legal_answer is None
    assert (state.steps[-1].agent, state.steps[-1].error_type) == ("guardrail:output", "RuntimeError")


def test_guardrail_not_run_after_a_failure():
    guard = Counting()
    deps = full_deps(contract=Fake("contract", status="failed", error="StructuredOutputError"), guardrails=guard)
    state = run(start(document_id="d1", mode="full_analysis"), deps)
    assert (state.status, state.failed_step) == ("failed", "contract")
    assert guard.seen == []


def test_qa_withheld_passages_become_a_warning():
    ctx = CTX.model_copy(update={"guardrail_findings": [withheld_finding()]})
    rag = FakeRetrieval(RAGResult(question="q", document_id="d1", found=True, context=ctx))
    answer = DocumentAnswer(
        document_id="d1", status="answered", answer="Notice is thirty (30) days. [E2]", citations=[],
        evidence_labels=["E1", "E2", "E3"], context_truncated=False,
    )
    qa = Fake("document_qa", answer, calls=1)
    state = run(
        start(document_id="d1", question="What is the notice period?", mode="document_qa"),
        WorkflowDeps(qa=qa, rag=rag),
    )
    assert (state.status, state.warnings) == ("completed", [WITHHELD])
    assert [f.code for f in state.guardrail_findings] == ["suspicious_content"]


def test_full_analysis_report_shows_withheld_passages():
    ctx = CTX.model_copy(update={"guardrail_findings": [withheld_finding()]})
    state = run(start(document_id="d1", mode="full_analysis"), full_deps(analysis_context=FakeRetrieval(ctx)))
    assert (state.status, state.warnings) == ("completed", [WITHHELD])
    assert WITHHELD in state.summary.could_not_confirm and state.summary.status == "ok_with_gaps"