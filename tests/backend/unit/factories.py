from collections import Counter

from app.agents.compliance.schemas import ComplianceReport, ReviewedFinding
from app.agents.contract.schemas import (
    ContractAnalysis,
    EvidenceRef,
    ExtractedClause,
    ExtractedFact,
    Party,
)
from app.agents.risk.schemas import ReviewedRisk, RiskReport
from app.business_logic.compliance_rules import evaluate, load_rules
from app.business_logic.risk_rules import evaluate_risks, load_risk_rules
from app.business_logic.scoring_engine import score_risks
from app.schemas.document import BuiltContext, EvidenceItem

TEXTS = [
    "SERVICE AGREEMENT between Acme Ltd (the Supplier) and Beta Corp (the Client), effective 1 March 2026. "
    "This Agreement is governed by the laws of India.",
    "Either party may terminate this Agreement on thirty (30) days written notice. "
    "The initial term of this Agreement is twelve (12) months. "
    "The Supplier shall keep all Client data confidential.",
    "The Client shall pay all invoices within 45 days of receipt. "
    "Interest accrues on late payments after thirty (30) days from the due date.",
]

# Tests assume the example rules in data/company_policies and data/risk_rules
C_RULES = load_rules()
R_RULES = load_risk_rules()


def make_context(texts=TEXTS) -> BuiltContext:
    ev = [
        EvidenceItem(label=f"E{i}", document_id="d", chunk_index=i, page_number=i, text=t, score=0.8)
        for i, t in enumerate(texts, 1)
    ]
    body = "\n\n".join(f"[{e.label}] (page {e.page_number})\n{e.text}" for e in ev)
    return BuiltContext(text="<evidence>\n" + body + "\n</evidence>", evidence=ev, truncated=False)


def ref(label, quote):
    return EvidenceRef(label=label, quote=quote)


def fact(value, label="E1", quote=None):
    return ExtractedFact(value=value, evidence=[ref(label, quote or value)])


def party(name, role, label, quote):
    return Party(name=name, role=role, evidence=[ref(label, quote)])


def good_analysis(**override) -> ContractAnalysis:
    base = dict(
        parties=[
            party("Acme Ltd", "Supplier", "E1", "Acme Ltd (the Supplier)"),
            party("Beta Corp", "Client", "E1", "Beta Corp (the Client)"),
        ],
        effective_date=fact("1 March 2026", "E1", "effective 1 March 2026"),
        term_or_expiry=fact("twelve (12) months", "E2", "initial term of this Agreement is twelve (12) months"),
        termination_notice=fact("thirty (30) days", "E2", "on thirty (30) days written notice"),
        payment_terms=fact("45 days", "E3", "pay all invoices within 45 days"),
        governing_law=fact("India", "E1", "governed by the laws of India"),
        clauses=[
            ExtractedClause(
                clause_type="confidentiality",
                summary="Supplier must keep Client data confidential",
                evidence=[ref("E2", "The Supplier shall keep all Client data confidential")],
            )
        ],
    )
    base.update(override)
    return ContractAnalysis(document_id="d", evidence_labels=["E1", "E2", "E3"], context_truncated=False, **base)


def comp_report(a: ContractAnalysis) -> ComplianceReport:
    fs = evaluate(a, C_RULES)
    return ComplianceReport(
        document_id="d",
        findings=[ReviewedFinding(**f.model_dump()) for f in fs],
        status_counts=dict(Counter(f.status for f in fs)),
        rules_evaluated=len(C_RULES),
    )


def risk_report(a: ContractAnalysis) -> RiskReport:
    fs = evaluate_risks(a, R_RULES)
    return RiskReport(
        document_id="d",
        findings=[ReviewedRisk(**f.model_dump()) for f in fs],
        score=score_risks(fs),
        candidates=[],
        candidates_dropped=0,
        rules_evaluated=len(R_RULES),
    )