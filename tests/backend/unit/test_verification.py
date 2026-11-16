import asyncio

from factories import (
    comp_report, fact, good_analysis, make_context, party, ref, risk_report,
)
from app.agents.contract.schemas import ContractAnalysis, ExtractedClause
from app.agents.verification.agent import VerificationAgent
from app.agents.verification.schemas import VerificationInput

CTX = make_context()


def verify(analysis, comp=None, risk=None):
    """Code-only run: no LLM involved."""
    res = asyncio.run(
        VerificationAgent().run(
            VerificationInput(
                document_id="d", context=CTX, analysis=analysis,
                compliance=comp, risk=risk, include_semantic=False,
            )
        )
    )
    assert res.status == "ok", res.error_type
    assert res.usage.llm_calls == 0
    return res.output


def codes(report):
    return sorted(i.code for i in report.issues)


def test_good_data_code_only():
    a = good_analysis()
    r = verify(a, comp_report(a), risk_report(a))
    assert r.result == "pass_with_warnings"  # a code-only run can never be a plain pass
    assert r.counts == {"verified": 8}
    assert codes(r) == ["semantic_check_skipped"]
    assert r.withheld_item_ids == [] and r.withheld_finding_ids == []


def test_tampered_items_are_withheld():
    a = good_analysis(
        payment_terms=fact("60 days", "E3", "pay all invoices within 45 days"),
        governing_law=fact("Delaware", "E1", "governed by the laws of Delaware"),
        parties=[
            party("Acme Ltd", "Supplier", "E1", "Acme Ltd (the Supplier)"),
            party("Beta Corp", "Client", "E9", "Beta Corp (the Client)"),
        ],
    )
    r = verify(a, comp_report(a), risk_report(a))
    assert r.result == "pass_with_warnings"
    assert r.counts == {"verified": 5, "failed_check": 3}
    assert r.withheld_item_ids == ["fact:payment_terms", "fact:governing_law", "party:1"]
    assert r.withheld_finding_ids == [
        "compliance:EX-002", "compliance:EX-003", "compliance:EX-005", "risk:RK-001", "risk:RK-003",
    ]
    reasons = {i.item_id: i.reasons for i in r.items if i.verdict != "verified"}
    assert reasons["fact:payment_terms"] == ["number(s) in the value are not in its quote: 60"]
    assert reasons["fact:governing_law"] == ["quote for E1 was not found in that evidence"]
    assert reasons["party:1"] == ["evidence label E9 is not in the retrieved evidence"]


def test_tampered_reports_fail():
    good = good_analysis()
    other = good_analysis(termination_notice=fact("15 days", "E2", "on thirty (30) days written notice"))
    bad_risk = risk_report(good)
    bad_risk = bad_risk.model_copy(update={"score": bad_risk.score.model_copy(update={"total_points": 99})})
    r = verify(good, comp_report(other), bad_risk)
    assert r.result == "fail"
    assert codes(r) == ["compliance_mismatch", "risk_mismatch", "semantic_check_skipped"]


def test_empty_analysis():
    a = ContractAnalysis(document_id="d", evidence_labels=["E1", "E2", "E3"], context_truncated=False)
    r = verify(a)
    assert r.result == "pass_with_warnings"
    assert r.counts == {}
    assert codes(r) == ["nothing_to_verify"]


def test_failed_clause_withholds_its_rule():
    a = good_analysis(
        clauses=[
            ExtractedClause(
                clause_type="confidentiality",
                summary="x",
                evidence=[ref("E2", "the Supplier shall pay a penalty of one million dollars")],
            )
        ]
    )
    r = verify(a, comp_report(a), risk_report(a))
    assert r.withheld_item_ids == ["clause:0"]
    assert r.withheld_finding_ids == ["compliance:EX-004"]