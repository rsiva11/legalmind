from datetime import date

import pytest

from factories import comp_report, fact, good_analysis, ref, risk_report
from test_workflow import qa_answer, summary_report
from app.agents.contract.schemas import ContractAnalysis
from app.agents.legal_assistant.schemas import LegalAnswer
from app.agents.risk.schemas import CandidateRisk
from app.agents.summary.schemas import RiskSummary
from app.agents.verification.schemas import ItemVerification, VerificationReport
from app.business_logic.escalation_rules import MESSAGES, error_decision, evaluate_escalation
from app.config import get_settings
from app.guardrails.result import GuardrailFinding
from app.orchestration.router import route
from app.orchestration.state import WorkflowRequest, WorkflowState
from test_router_state import result as agent_result

TODAY = date(2026, 6, 1)
DATE_IDS = ("fact:effective_date", "fact:term_or_expiry")


def new_state(**kw):
    req = WorkflowRequest(**kw)
    return WorkflowState.start(req, route(req))


def verification(ids=DATE_IDS, result="pass", withheld=()):
    items = [ItemVerification(item_id=i, kind="fact", label=i, verdict="verified") for i in ids]
    return VerificationReport(
        document_id="d1", result=result, items=items, withheld_item_ids=list(withheld),
        withheld_finding_ids=[], issues=[], semantic_checked=True, counts={},
    )


def build(analysis=None, *, ids=DATE_IDS, result="pass", withheld=(), **extra):
    a = analysis if analysis is not None else good_analysis()
    s = new_state(document_id="d1", mode="full_analysis")
    s.contract, s.compliance, s.risk = a, comp_report(a), risk_report(a)
    s.verification = verification(ids, result, withheld)
    for name, value in extra.items():
        setattr(s, name, value)
    return s


def codes(state):
    return evaluate_escalation(state, TODAY).codes


def finding(guardrail, action, code):
    return GuardrailFinding(guardrail=guardrail, action=action, code=code, field="f", detail="d")


def with_unresolved(state, total, unresolved):
    kept = state.compliance.findings[:total]
    kept = [f.model_copy(update={"status": "not_found"}) if i < unresolved else f for i, f in enumerate(kept)]
    state.compliance = state.compliance.model_copy(update={"findings": kept})


def with_coverage(state, value):
    score = state.risk.score.model_copy(update={"coverage": value})
    state.risk = state.risk.model_copy(update={"score": score})


def risk_summary(level, coverage=1.0):
    return RiskSummary(level=level, total_points=0, coverage=coverage, complete=True, note="n", lines=[])


def test_clean_state_needs_no_review():
    d = evaluate_escalation(build(), TODAY)
    assert (d.needs_human_review, d.codes, d.reasons) == (False, [], [])


def test_verification_failed():
    assert codes(build(result="fail")) == ["verification_failed"]


def test_withheld_items():
    assert codes(build(withheld=["fact:payment_terms"])) == ["items_withheld"]


def test_low_severity_noncompliance_and_medium_risk_do_not_escalate():
    s = build(good_analysis(payment_terms=fact("ninety (90) days")))
    assert s.risk.score.level == "medium"  # sanity check of the fixture
    assert codes(s) == []


def test_high_severity_noncompliance():
    assert codes(build(good_analysis(termination_notice=fact("15 days")))) == ["high_severity_noncompliance"]


def test_compliance_needs_review():
    s = build(good_analysis(termination_notice=fact("ten business days")))
    assert s.risk.score.coverage == 0.6  # RK-002 and RK-005 are both unreadable
    assert codes(s) == ["compliance_needs_review"]


def test_many_unresolved_rules_and_low_coverage_when_nothing_is_found():
    empty = ContractAnalysis(document_id="d", evidence_labels=["E1"], context_truncated=False)
    s = build(empty)
    assert s.risk.score.level == "not_assessed"
    assert codes(s) == ["low_risk_coverage", "many_unresolved_rules"]


def test_unresolved_share_boundary():
    half = build()
    with_unresolved(half, 4, 2)  # 2 of 4 = exactly 50%: escalates ("at or above")
    assert codes(half) == ["many_unresolved_rules"]
    quarter = build()
    with_unresolved(quarter, 4, 1)
    assert codes(quarter) == []


@pytest.mark.parametrize(
    "extra, level",
    [
        ({}, "high"),
        ({"governing_law": fact("the laws of New York"), "term_or_expiry": fact("five (5) years")}, "critical"),
    ],
)
def test_high_and_critical_risk_levels(extra, level):
    a = good_analysis(payment_terms=fact("ninety (90) days"), termination_notice=fact("15 days"), **extra)
    s = build(a)
    assert s.risk.score.level == level  # sanity check of the fixture
    assert codes(s) == ["high_severity_noncompliance", "high_risk_level"]


def test_risk_coverage_boundary():
    at_limit = build()
    with_coverage(at_limit, 0.5)  # "below 50%" does not include 50%
    assert codes(at_limit) == []
    below = build()
    with_coverage(below, 0.49)
    assert codes(below) == ["low_risk_coverage"]


def test_risk_view_follows_the_summary():
    high = build()
    high.summary = summary_report().model_copy(update={"risk": risk_summary("high")})
    assert codes(high) == ["high_risk_level"]

    # the report says critical, but the user sees "low" (findings were withheld): the user's view wins
    critical = build(
        good_analysis(
            payment_terms=fact("ninety (90) days"), termination_notice=fact("15 days"),
            governing_law=fact("the laws of New York"), term_or_expiry=fact("five (5) years"),
        )
    )
    assert critical.risk.score.level == "critical"
    critical.summary = summary_report().model_copy(update={"risk": risk_summary("low")})
    assert codes(critical) == ["high_severity_noncompliance"]

    thin = build()
    thin.summary = summary_report().model_copy(update={"risk": risk_summary("low", coverage=0.3)})
    assert codes(thin) == ["low_risk_coverage"]


@pytest.mark.parametrize("code", ["suspicious_content", "wrong_document"])
def test_suspicious_retrieval_findings(code):
    assert codes(build(guardrail_findings=[finding("retrieval", "block", code)])) == ["suspicious_document_content"]


def test_findings_that_do_not_escalate():
    flags = [
        finding("retrieval", "flag", "suspicious_content"),
        finding("input", "flag", "pii_masked"),
        finding("input", "flag", "prompt_extraction"),
        finding("output", "flag", "legal_conclusion"),
        finding("tool", "flag", "suspicious_output"),
    ]
    assert codes(build(guardrail_findings=flags)) == []


@pytest.mark.parametrize("code", ["text_replaced", "candidate_dropped"])
def test_guardrail_interventions(code):
    assert codes(build(guardrail_findings=[finding("risk", "flag", code)])) == ["guardrail_intervention"]


@pytest.mark.parametrize("step", ["guardrail:output", "guardrail:risk"])
def test_guardrail_failures(step):
    s = build()
    s.fail(step)
    assert codes(s) == ["guardrail_failure"]


def test_non_guardrail_failure_does_not_escalate():
    s = new_state(document_id="d1", mode="full_analysis")
    s.fail("contract")
    assert evaluate_escalation(s, TODAY).needs_human_review is False


def test_candidate_risks():
    s = build()
    candidate = CandidateRisk(
        title="Unlimited liability", category="legal", description="The liability is stated as unlimited.",
        evidence=[ref("E1", "liability shall be unlimited")],
    )
    s.risk = s.risk.model_copy(update={"candidates": [candidate]})
    assert codes(s) == ["candidate_risks_present"]


@pytest.mark.parametrize(
    "term, escalates",
    [("3 months", True), ("2 months", True), ("4 months", True), ("5 months", False)],
)
def test_contract_expiry(term, escalates):
    # effective 1 March 2026, as of 1 June 2026: 3 months = 0 days left, 4 months = 30, 5 months = 61
    s = build(good_analysis(term_or_expiry=fact(term)))
    assert codes(s) == (["contract_expiring"] if escalates else [])


def test_unverified_dates_are_ignored():
    s = build(good_analysis(term_or_expiry=fact("2 months")), ids=())  # expired, but not verified
    assert codes(s) == []


def test_dates_need_a_verification_report():
    s = build(good_analysis(term_or_expiry=fact("2 months")))
    s.verification = None
    assert codes(s) == []


@pytest.mark.parametrize(
    "effective, term",
    [("05/03/2026", "twelve (12) months"), ("1 March 2026", "05/03/2026")],
)
def test_ambiguous_dates(effective, term):
    assert codes(build(good_analysis(effective_date=fact(effective), term_or_expiry=fact(term)))) == ["ambiguous_date"]


def test_date_problems_that_do_not_escalate():
    cases = [
        {"effective_date": None},
        {"effective_date": fact("nothing here")},
        {"term_or_expiry": fact("ten business days")},
        {"term_or_expiry": fact("perpetual")},
    ]
    for override in cases:
        assert codes(build(good_analysis(**override))) == [], override


def test_reasons_follow_table_order():
    s = build(result="fail")
    s.guardrail_findings = [finding("retrieval", "block", "suspicious_content")]
    s.fail("guardrail:output")
    d = evaluate_escalation(s, TODAY)
    assert d.codes == ["verification_failed", "suspicious_document_content", "guardrail_failure"]
    assert d.reasons == [MESSAGES[c] for c in d.codes] and d.needs_human_review is True


def test_messages_are_fixed():
    assert list(MESSAGES) == [
        "verification_failed", "items_withheld", "high_severity_noncompliance", "compliance_needs_review",
        "high_risk_level", "low_risk_coverage", "many_unresolved_rules", "suspicious_document_content",
        "guardrail_intervention", "guardrail_failure", "step_skipped", "candidate_risks_present",
        "contract_expiring", "ambiguous_date", "escalation_error",
    ]
    assert len(set(MESSAGES.values())) == len(MESSAGES)


def test_skipped_step_escalates():
    s = build()
    s.record_step(
        agent_result("compliance", status="failed", calls=0, prompt=0, completion=0, error="TimeoutError"),
        fatal=False,
    )
    assert s.status == "running" and s.failed_step is None
    assert codes(s) == ["step_skipped"]


def test_fatal_failure_is_not_a_skipped_step():
    s = build()
    s.record_step(agent_result("contract", status="failed", calls=0, prompt=0, completion=0, error="ValueError"))
    assert s.failed_step == "contract"
    assert codes(s) == []


def test_qa_and_legal_answers_are_not_escalated():
    qa = new_state(document_id="d1", question="What is the notice period?", mode="document_qa")
    qa.document_answer = qa_answer("unverified")
    assert evaluate_escalation(qa, TODAY).needs_human_review is False

    legal = new_state(question="Should I sue my landlord?")
    legal.legal_answer = LegalAnswer(
        in_scope=True, answer="x", key_points=[], jurisdiction_dependent=True,
        seek_professional_advice=True, disclaimer="d",
    )
    assert evaluate_escalation(legal, TODAY).codes == []


def test_qa_with_suspicious_passages_is_escalated():
    qa = new_state(document_id="d1", question="What is the notice period?", mode="document_qa")
    qa.guardrail_findings = [finding("retrieval", "block", "suspicious_content")]
    assert evaluate_escalation(qa, TODAY).codes == ["suspicious_document_content"]


def test_error_decision():
    d = error_decision()
    assert (d.needs_human_review, d.codes, d.reasons) == (True, ["escalation_error"], [MESSAGES["escalation_error"]])


def test_thresholds_are_configurable(monkeypatch):
    s = build(good_analysis(termination_notice=fact("ten business days")))  # coverage 0.6
    assert codes(s) == ["compliance_needs_review"]
    monkeypatch.setattr(get_settings(), "escalation_min_risk_coverage", 0.9)
    assert codes(s) == ["compliance_needs_review", "low_risk_coverage"]