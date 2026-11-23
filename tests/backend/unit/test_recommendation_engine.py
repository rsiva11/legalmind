from factories import fact, good_analysis
from test_escalation_rules import TODAY, build, new_state
from test_workflow import qa_answer
from app.agents.contract.schemas import ContractAnalysis
from app.business_logic.escalation_rules import MESSAGES, evaluate_escalation
from app.business_logic.recommendation_engine import ACTIONS, recommend


def recs(state):
    state.review_codes = evaluate_escalation(state, TODAY).codes
    return recommend(state)


def shape(items):
    return [(r.source, r.code, r.ref) for r in items]


def test_clean_state_has_no_recommendations():
    assert recs(build()) == []


def test_every_review_code_has_a_fixed_action():
    assert set(ACTIONS) == set(MESSAGES)
    assert len(set(ACTIONS.values())) == len(ACTIONS)


def test_review_recommendations_follow_the_code_order():
    s = new_state(document_id="d1", mode="full_analysis")
    s.review_codes = ["verification_failed", "guardrail_failure"]
    out = recommend(s)
    assert shape(out) == [("review", "verification_failed", None), ("review", "guardrail_failure", None)]
    assert [r.text for r in out] == [ACTIONS["verification_failed"], ACTIONS["guardrail_failure"]]


def test_non_compliant_and_triggered_findings():
    out = recs(build(good_analysis(termination_notice=fact("15 days"))))
    assert shape(out) == [
        ("review", "high_severity_noncompliance", None),
        ("finding", "compliance:non_compliant", "EX-001"),
        ("finding", "risk:triggered", "RK-002"),
    ]
    assert out[0].text == ACTIONS["high_severity_noncompliance"]
    assert out[1].text == (
        "Discuss 'Termination notice of at least 30 days' with the counterparty or your legal team: "
        "the contract does not meet this rule."
    )
    assert out[2].text == "Review the 'Short termination notice' risk (operational, high severity) with your legal team."


def test_needs_review_findings():
    out = recs(build(good_analysis(termination_notice=fact("ten business days"))))
    assert shape(out) == [
        ("review", "compliance_needs_review", None),
        ("finding", "compliance:needs_review", "EX-001"),
        ("finding", "risk:needs_review", "RK-002"),
        ("finding", "risk:needs_review", "RK-005"),
    ]
    assert out[1].text == (
        "Check the wording for 'Termination notice of at least 30 days' manually: "
        "it could not be judged automatically."
    )


def test_not_found_and_unassessed_rules_get_no_individual_recommendation():
    empty = ContractAnalysis(document_id="d", evidence_labels=["E1"], context_truncated=False)
    assert shape(recs(build(empty))) == [
        ("review", "low_risk_coverage", None),
        ("review", "many_unresolved_rules", None),
    ]


def test_withheld_findings_are_skipped():
    s = build(good_analysis(termination_notice=fact("15 days")))
    s.verification = s.verification.model_copy(update={"withheld_finding_ids": ["compliance:EX-001"]})
    assert shape(recs(s)) == [
        ("review", "high_severity_noncompliance", None),
        ("finding", "risk:triggered", "RK-002"),
    ]


def test_answers_that_are_not_escalated_have_no_recommendations():
    s = new_state(document_id="d1", question="What is the notice period?", mode="document_qa")
    s.document_answer = qa_answer("unverified")
    assert recs(s) == []


def test_recommendations_contain_no_contract_text():
    out = recs(build(good_analysis(governing_law=fact("the laws of Narnia"))))
    assert shape(out) == [
        ("finding", "compliance:non_compliant", "EX-003"),
        ("finding", "risk:triggered", "RK-003"),
    ]
    assert all("Narnia" not in r.text for r in out)