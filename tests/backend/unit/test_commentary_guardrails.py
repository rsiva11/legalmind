from factories import comp_report, good_analysis, ref, risk_report
from app.agents.risk.schemas import CandidateRisk
from app.guardrails.compliance_guardrail import screen_compliance_report
from app.guardrails.guardrail_manager import GuardrailManager
from app.guardrails.output_guardrail import ACTION_FALLBACK, EXPLANATION_FALLBACK, check_text
from app.guardrails.risk_guardrail import screen_risk_report

A = good_analysis()


def with_first(report, **update):
    first = report.findings[0].model_copy(update=update)
    return report.model_copy(update={"findings": [first, *report.findings[1:]]})


def summarize(findings):
    return [(f.guardrail, f.action, f.code, f.field) for f in findings]


def candidate(description):
    return CandidateRisk(
        title="Unlimited liability", category="legal", description=description,
        evidence=[ref("E1", "liability shall be unlimited")],
    )


def test_clean_compliance_report_is_unchanged():
    rep = comp_report(A)
    out, findings = screen_compliance_report(rep)
    assert out == rep and findings == []


def test_bad_compliance_explanation_is_replaced():
    rep = with_first(comp_report(A), explanation="This notice period is unenforceable.")
    out, findings = screen_compliance_report(rep)
    assert out.findings[0].explanation == EXPLANATION_FALLBACK
    assert summarize(findings) == [("compliance", "flag", "text_replaced", "compliance.EX-001.explanation")]
    assert findings[0].detail == "Model-written text was replaced with a fixed message; reason: legal_conclusion"


def test_only_the_bad_text_is_replaced():
    rep = with_first(
        comp_report(A),
        explanation="The period is unenforceable.",
        suggested_action="Ask the counterparty to confirm the period.",
    )
    out, _ = screen_compliance_report(rep)
    assert out.findings[0].explanation == EXPLANATION_FALLBACK
    assert out.findings[0].suggested_action == "Ask the counterparty to confirm the period."


def test_bad_action_is_replaced_with_the_action_fallback():
    rep = with_first(comp_report(A), suggested_action="You should sign this contract.")
    out, findings = screen_compliance_report(rep)
    assert out.findings[0].suggested_action == ACTION_FALLBACK
    assert findings[0].detail == "Model-written text was replaced with a fixed message; reason: legal_advice"


def test_wording_from_the_source_is_kept_and_flagged():
    rep = with_first(
        comp_report(A),
        explanation="The clause says it is unenforceable if signed under duress.",
        evidence=[ref("E1", "The Agreement is unenforceable if signed under duress.")],
    )
    out, findings = screen_compliance_report(rep)
    assert out.findings[0].explanation == rep.findings[0].explanation
    assert summarize(findings) == [("compliance", "flag", "legal_conclusion", "compliance.EX-001.explanation")]


def test_statuses_and_details_are_never_touched():
    rep = with_first(comp_report(A), explanation="This is illegal.", suggested_action="You should sign this.")
    out, _ = screen_compliance_report(rep)
    keep = {"explanation", "suggested_action"}
    assert [f.model_dump(exclude=keep) for f in out.findings] == [f.model_dump(exclude=keep) for f in rep.findings]
    assert out.status_counts == rep.status_counts


def test_findings_do_not_echo_the_text():
    rep = with_first(comp_report(A), explanation="This is unenforceable and the secret is hunter2.")
    _, findings = screen_compliance_report(rep)
    assert findings and all("hunter2" not in f.detail and "unenforceable" not in f.detail.lower() for f in findings)


def test_bad_risk_explanation_is_replaced_and_the_score_is_unchanged():
    rep = with_first(risk_report(A), explanation="The payment term is illegal.")
    out, findings = screen_risk_report(rep)
    assert out.findings[0].explanation == EXPLANATION_FALLBACK
    assert out.score == rep.score and out.findings[0].outcome == rep.findings[0].outcome
    assert summarize(findings) == [("risk", "flag", "text_replaced", "risk.RK-001.explanation")]


def test_bad_candidates_are_dropped_good_ones_kept():
    rep = risk_report(A).model_copy(
        update={
            "candidates": [
                candidate("The liability clause is void and unlawful."),
                candidate("The liability is stated as unlimited."),
            ]
        }
    )
    out, findings = screen_risk_report(rep)
    assert [c.description for c in out.candidates] == ["The liability is stated as unlimited."]
    assert summarize(findings) == [("risk", "flag", "candidate_dropped", "risk.candidates[0]")]
    assert findings[0].detail == "Suggested risk was dropped; reason: legal_conclusion"


def test_fallbacks_pass_every_profile():
    for profile in ("general", "document", "commentary"):
        for text in (EXPLANATION_FALLBACK, ACTION_FALLBACK):
            assert check_text("f", text, profile, "", set()) == []


def test_manager_delegates_to_the_screens():
    m = GuardrailManager()
    comp, risk = comp_report(A), risk_report(A)
    assert m.screen_compliance(comp) == (comp, [])
    assert m.screen_risk(risk) == (risk, [])