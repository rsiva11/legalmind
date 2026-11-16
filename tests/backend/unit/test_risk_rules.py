from factories import fact, good_analysis
from app.agents.contract.schemas import ContractAnalysis
from app.business_logic.risk_rules import evaluate_risks, load_risk_rules
from app.business_logic.scoring_engine import score_risks

RULES = load_risk_rules()


def run(a):
    findings = evaluate_risks(a, RULES)
    return findings, score_risks(findings)


def outcomes(findings):
    return {f.rule_id: f.outcome for f in findings}


def test_risk_rules_load():
    assert [r.id for r in RULES] == ["RK-001", "RK-002", "RK-003", "RK-004", "RK-005"]


def test_all_clear():
    findings, s = run(good_analysis())
    assert set(outcomes(findings).values()) == {"not_triggered"}
    assert (s.total_points, s.level, s.coverage, s.complete) == (0, "low", 1.0, True)


def test_critical_when_everything_triggers():
    a = good_analysis(
        payment_terms=fact("ninety (90) days"),
        termination_notice=fact("15 days"),
        governing_law=fact("the laws of New York"),
        term_or_expiry=fact("five (5) years"),
    )
    findings, s = run(a)
    assert outcomes(findings) == {
        "RK-001": "triggered", "RK-002": "triggered", "RK-003": "triggered",
        "RK-004": "triggered", "RK-005": "not_triggered",
    }
    assert (s.total_points, s.level, s.triggered, s.not_triggered) == (70, "critical", 4, 1)
    assert s.by_category == {"financial": 20, "operational": 25, "legal": 15, "business": 10}


def test_partial_evidence_is_a_lower_bound():
    a = ContractAnalysis(
        document_id="d", evidence_labels=["E1"], context_truncated=False,
        payment_terms=fact("thirty days or sixty days"), governing_law=fact("India"),
    )
    findings, s = run(a)
    assert outcomes(findings) == {
        "RK-001": "needs_review", "RK-002": "unassessed", "RK-003": "not_triggered",
        "RK-004": "unassessed", "RK-005": "unassessed",
    }
    # "low" here means "at least low": coverage and complete must say so
    assert (s.total_points, s.level, s.coverage, s.complete) == (0, "low", 0.2, False)


def test_nothing_assessable_is_not_low():
    a = ContractAnalysis(document_id="d", evidence_labels=[], context_truncated=False)
    findings, s = run(a)
    assert set(outcomes(findings).values()) == {"unassessed"}
    assert (s.level, s.coverage, s.complete) == ("not_assessed", 0.0, False)