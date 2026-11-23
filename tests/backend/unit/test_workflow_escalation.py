from datetime import date

from factories import good_analysis, risk_report
from test_workflow import CTX, Fake, FakeRetrieval, full_deps, qa_answer, run, start, summary_report, verification_report
from app.business_logic.escalation_rules import MESSAGES, evaluate_escalation
from app.orchestration.workflow import WorkflowDeps
from app.schemas.document import RAGResult

TODAY = date(2026, 6, 1)


def escalate(state):
    return evaluate_escalation(state, TODAY)  # a fixed date, so tests never depend on the clock


def review(state):
    return state.status, state.needs_human_review, state.review_codes, state.review_reasons


def full(**override):
    return run(start(document_id="d1", mode="full_analysis"), full_deps(escalate=escalate, **override))


def test_clean_run_is_not_escalated():
    assert review(full()) == ("completed", False, [], [])


def test_failed_verification_escalates():
    state = full(verification=Fake("verification", verification_report("fail"), tokens=10, calls=1))
    assert review(state) == ("completed", True, ["verification_failed"], [MESSAGES["verification_failed"]])


def test_technical_failure_is_not_escalated():
    state = full(contract=Fake("contract", status="failed", error="StructuredOutputError"))
    assert review(state) == ("failed", False, [], [])


def test_guardrail_stop_escalates():
    bad = summary_report().model_copy(update={"overview": "Notice is 30 days [E9].", "overview_source": "model"})
    state = full(summary=Fake("summary", bad, tokens=20, calls=1))
    assert (state.status, state.failed_step) == ("failed", "guardrail:output")
    assert review(state)[1:] == (True, ["guardrail_failure"], [MESSAGES["guardrail_failure"]])


def test_escalation_error_fails_closed():
    def boom(state):
        raise RuntimeError("boom")

    state = run(start(document_id="d1", mode="full_analysis"), full_deps(escalate=boom))
    assert review(state) == ("completed", True, ["escalation_error"], [MESSAGES["escalation_error"]])


def test_unverified_qa_answer_is_not_escalated():
    rag = FakeRetrieval(RAGResult(question="q", document_id="d1", found=True, context=CTX))
    qa = Fake("document_qa", qa_answer("unverified"), calls=1)
    state = run(
        start(document_id="d1", question="What is the notice period?", mode="document_qa"),
        WorkflowDeps(qa=qa, rag=rag, escalate=escalate),
    )
    assert review(state) == ("completed", False, [], [])


def test_review_fields_are_stored_with_the_job():
    state = full(verification=Fake("verification", verification_report("fail"), tokens=10, calls=1))
    dumped = state.model_dump(mode="json")
    assert dumped["needs_human_review"] is True
    assert dumped["review_codes"] == ["verification_failed"]
    assert dumped["review_reasons"] == [MESSAGES["verification_failed"]]

from app.business_logic.recommendation_engine import ACTIONS


def test_skipped_step_escalates():
    state = full(compliance=Fake("compliance", status="failed", error="TimeoutError"))
    assert review(state)[:3] == ("completed", True, ["step_skipped"])


def test_recommendations_for_a_clean_run():
    assert full().recommendations == []


def test_recommendations_follow_the_review_codes():
    state = full(verification=Fake("verification", verification_report("fail"), tokens=10, calls=1))
    assert [(r.source, r.code) for r in state.recommendations] == [("review", "verification_failed")]
    assert state.recommendations[0].text == ACTIONS["verification_failed"]


def test_recommender_error_is_a_warning_not_a_failure():
    def boom(state):
        raise RuntimeError("boom")

    state = run(start(document_id="d1", mode="full_analysis"), full_deps(escalate=escalate, recommender=boom))
    assert (state.status, state.recommendations) == ("completed", [])
    assert state.warnings == ["Recommendations could not be generated"]


def test_recommendations_are_stored_with_the_job():
    state = full(verification=Fake("verification", verification_report("fail"), tokens=10, calls=1))
    assert state.model_dump(mode="json")["recommendations"] == [
        {"code": "verification_failed", "text": ACTIONS["verification_failed"], "source": "review", "ref": None}
    ]