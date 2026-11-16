from factories import comp_report, good_analysis, make_context, ref, risk_report
from app.agents.document_qa.schemas import DocumentAnswer
from app.agents.legal_assistant.schemas import LegalAnswer
from app.agents.risk.schemas import CandidateRisk
from app.agents.summary.schemas import SummaryReport
from app.guardrails.guardrail_manager import GuardrailManager
from app.orchestration.router import route
from app.orchestration.state import WorkflowRequest, WorkflowState
from app.schemas.document import Citation

CTX = make_context()
MANAGER = GuardrailManager()


def new_state(**kw):
    req = WorkflowRequest(**kw)
    return WorkflowState.start(req, route(req))


def legal(answer="A notice period is the time one party must give before ending a contract.", points=()):
    return LegalAnswer(
        in_scope=True, answer=answer, key_points=list(points), jurisdiction_dependent=False,
        seek_professional_advice=False, disclaimer="d",
    )


def summary(overview="o", source="template"):
    return SummaryReport(
        document_id="d1", status="ok", verification_result="pass", overview=overview, overview_source=source,
        key_facts=[], compliance=[], risk=None, suggestions=[], could_not_confirm=[], disclaimer="d", markdown="",
    )


def doc_answer(answer, status="answered", citations=()):
    return DocumentAnswer(
        document_id="d1", status=status, answer=answer, citations=list(citations),
        evidence_labels=["E1"], context_truncated=False,
    )


def where(result):
    return [(f.code, f.field, f.action) for f in result.findings]


def test_clean_outputs_are_allowed():
    s = new_state(question="What is a notice period?")
    s.legal_answer = legal(points=["It gives the other side time to prepare."])
    assert MANAGER.check_outputs(s).action == "allow"

    a = good_analysis()
    s2 = new_state(document_id="d1", mode="full_analysis")
    s2.compliance, s2.risk, s2.summary = comp_report(a), risk_report(a), summary()
    assert MANAGER.check_outputs(s2, CTX).action == "allow"


def test_legal_answer_advice_is_blocked():
    s = new_state(question="Should I sign this contract?")
    s.legal_answer = legal("You should sign this contract.")
    assert where(MANAGER.check_outputs(s)) == [("legal_advice", "legal_answer.answer", "block")]

    s.legal_answer = legal(points=["Never sign without reading it."])
    assert where(MANAGER.check_outputs(s)) == [("legal_advice", "legal_answer.key_points[0]", "block")]


def test_compliance_explanation_is_screened():
    rep = comp_report(good_analysis())
    first = rep.findings[0].model_copy(update={"explanation": "This notice period is unenforceable."})
    s = new_state(document_id="d1", mode="full_analysis")
    s.compliance = rep.model_copy(update={"findings": [first, *rep.findings[1:]]})
    res = MANAGER.check_outputs(s, CTX)
    assert (res.action, where(res)) == ("block", [("legal_conclusion", "compliance.EX-001.explanation", "block")])

    ok = rep.findings[0].model_copy(update={"suggested_action": "Ask the counterparty to confirm the period."})
    s.compliance = rep.model_copy(update={"findings": [ok, *rep.findings[1:]]})
    assert MANAGER.check_outputs(s, CTX).action == "allow"


def test_risk_candidate_is_screened():
    a = good_analysis()
    quote = ref("E1", "liability shall be unlimited")

    def candidate(description):
        return CandidateRisk(title="Unlimited liability", category="legal", description=description, evidence=[quote])

    s = new_state(document_id="d1", mode="full_analysis")
    s.risk = risk_report(a).model_copy(update={"candidates": [candidate("The liability clause is void and unlawful.")]})
    assert where(MANAGER.check_outputs(s, CTX)) == [("legal_conclusion", "risk.candidates[0]", "block")]

    s.risk = risk_report(a).model_copy(update={"candidates": [candidate("The liability is stated as unlimited.")]})
    assert MANAGER.check_outputs(s, CTX).action == "allow"


def test_model_overview_is_screened_template_is_not():
    s = new_state(document_id="d1", mode="full_analysis")
    s.summary = summary("Fees are due in 45 days [E3]. Notice is 30 days [E9].", "model")
    assert where(MANAGER.check_outputs(s, CTX)) == [("unknown_citation", "summary.overview", "block")]

    s.summary = summary("This is illegal [E9].", "template")
    assert MANAGER.check_outputs(s, CTX).action == "allow"


def test_document_answer_uses_the_cited_evidence():
    ctx = make_context(["Disputes go to Delhi and the court will have exclusive jurisdiction."])
    answer = "Disputes go to Delhi, where the court will have exclusive jurisdiction. [E1]"
    cite = Citation(label="E1", page_number=1, chunk_index=1, document_id="d")
    s = new_state(document_id="d1", question="Which court?", mode="document_qa")
    for citations in ([cite], []):
        s.document_answer = doc_answer(answer, citations=citations)
        res = MANAGER.check_outputs(s, ctx)
        assert (res.action, where(res)) == ("flag", [("outcome_prediction", "document_answer.answer", "flag")])


def test_unverified_and_not_found_answers_are_not_screened():
    s = new_state(document_id="d1", question="Which court?", mode="document_qa")
    for status in ("unverified", "not_found"):
        s.document_answer = doc_answer("You should sign this.", status=status)
        assert MANAGER.check_outputs(s, CTX).action == "allow"