from app.agents.contract.schemas import (
    ContractAnalysis,
    EvidenceRef,
    ExtractedClause,
    ExtractedFact,
)
from app.business_logic.compliance_rules import evaluate, load_rules, parse_duration_days

# These tests assume the example rules in data/company_policies/example_policy.yaml
RULES = load_rules()


def fact(value):
    return ExtractedFact(value=value, evidence=[EvidenceRef(label="E1", quote=value)])


def clause(kind):
    return ExtractedClause(
        clause_type=kind, summary="x", evidence=[EvidenceRef(label="E1", quote="quote here")]
    )


def analysis(**kw):
    return ContractAnalysis(document_id="d", evidence_labels=["E1"], context_truncated=False, **kw)


def statuses(a):
    return {f.rule_id: f.status for f in evaluate(a, RULES)}


def details(a):
    return {f.rule_id: f.detail for f in evaluate(a, RULES)}


def test_example_rules_load():
    assert [r.id for r in RULES] == ["EX-001", "EX-002", "EX-003", "EX-004", "EX-005"]


def test_scenario_compliant():
    a = analysis(
        termination_notice=fact("thirty (30) days"),
        payment_terms=fact("45 days"),
        clauses=[clause("confidentiality")],
    )
    assert statuses(a) == {
        "EX-001": "compliant", "EX-002": "compliant", "EX-003": "not_found",
        "EX-004": "compliant", "EX-005": "compliant",
    }


def test_scenario_violations():
    a = analysis(
        termination_notice=fact("15 days"),
        payment_terms=fact("ninety (90) days"),
        governing_law=fact("the laws of New York"),
    )
    assert statuses(a) == {
        "EX-001": "non_compliant", "EX-002": "compliant", "EX-003": "non_compliant",
        "EX-004": "not_found", "EX-005": "non_compliant",
    }


def test_scenario_needs_review():
    a = analysis(
        termination_notice=fact("ten business days"),
        payment_terms=fact("thirty days or sixty days"),
        governing_law=fact("India"),
    )
    assert statuses(a) == {
        "EX-001": "needs_review", "EX-002": "compliant", "EX-003": "compliant",
        "EX-004": "not_found", "EX-005": "needs_review",
    }


def test_missing_evidence_is_not_a_failure():
    assert set(statuses(analysis()).values()) == {"not_found"}


def test_details_show_threshold_comparison():
    bad = details(analysis(termination_notice=fact("15 days"), payment_terms=fact("ninety (90) days")))
    assert bad["EX-001"] == "Found '15 days' (15 days); requires at least 30 days"
    assert bad["EX-005"] == "Found 'ninety (90) days' (90 days); requires at most 60 days"
    good = details(analysis(payment_terms=fact("45 days")))
    assert good["EX-005"] == "Found '45 days' (45 days); requires at most 60 days"


def test_field_present_detail_has_no_underscore():
    assert details(analysis(payment_terms=fact("45 days")))["EX-002"] == "payment terms was found"


def test_parse_duration():
    assert parse_duration_days("30 days")[0] == 30
    assert parse_duration_days("twelve (12) months")[0] == 365.25
    assert parse_duration_days("ten business days")[0] is None
    assert parse_duration_days("thirty days or sixty days")[0] is None