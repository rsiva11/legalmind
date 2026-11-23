from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:  # annotation only: business_logic must not import orchestration at run time
    from app.orchestration.state import WorkflowState

Source = Literal["review", "finding"]

# One fixed sentence per review code. A test checks this covers every escalation code.
ACTIONS: dict[str, str] = {
    "verification_failed": "Do not rely on this analysis. Have the document reviewed manually, or re-run the analysis after the cause is fixed.",
    "items_withheld": "Check the withheld items directly in the contract; they could not be verified automatically.",
    "high_severity_noncompliance": "Raise the unmet high-severity rules with the counterparty or your legal team.",
    "compliance_needs_review": "Check the flagged contract wording manually; the automatic checks could not judge it.",
    "high_risk_level": "Have a qualified person review the highest-risk findings before relying on this analysis.",
    "low_risk_coverage": "Treat the risk level as a minimum, and run a targeted search or manual review for the risk rules that could not be assessed.",
    "many_unresolved_rules": "Review the full document for the rules that were not found; the retrieved text may have missed them.",
    "suspicious_document_content": "Inspect the document for unusual or inconsistent passages that were withheld from the analysis.",
    "guardrail_intervention": "Read the affected findings directly in the contract; some explanatory text was replaced by a safety check.",
    "guardrail_failure": "This job was stopped by a safety check. Have the document reviewed manually.",
    "step_skipped": "Re-run the analysis, or review manually the section that could not be produced.",
    "candidate_risks_present": "Discuss the unverified risk suggestions with a qualified person; they are not confirmed.",
    "contract_expiring": "Confirm the contract's expiry date and any renewal or notice requirements.",
    "ambiguous_date": "Confirm the intended date format with the counterparty; ambiguous dates were not interpreted.",
    "escalation_error": "The review check failed. Have this result reviewed manually.",
}


class Recommendation(BaseModel):
    code: str
    text: str
    source: Source
    ref: str | None = None  # the rule id for finding-based recommendations


def _non_compliant(name: str) -> str:
    return f"Discuss '{name}' with the counterparty or your legal team: the contract does not meet this rule."


def _needs_check(name: str) -> str:
    return f"Check the wording for '{name}' manually: it could not be judged automatically."


def _risk(name: str, category: str, severity: str) -> str:
    return f"Review the '{name}' risk ({category}, {severity} severity) with your legal team."


def recommend(state: "WorkflowState") -> list[Recommendation]:
    """Pure function of the finished state, in a stable order: review codes first, then findings."""
    out: list[Recommendation] = []

    for code in state.review_codes:
        text = ACTIONS.get(code)
        if text:
            out.append(Recommendation(code=code, text=text, source="review"))

    withheld = set(state.verification.withheld_finding_ids) if state.verification else set()

    if state.compliance is not None:
        for f in state.compliance.findings:
            if f"compliance:{f.rule_id}" in withheld:
                continue
            if f.status == "non_compliant":
                out.append(Recommendation(code="compliance:non_compliant", text=_non_compliant(f.rule_name), source="finding", ref=f.rule_id))
            elif f.status == "needs_review":
                out.append(Recommendation(code="compliance:needs_review", text=_needs_check(f.rule_name), source="finding", ref=f.rule_id))

    if state.risk is not None:
        for f in state.risk.findings:
            if f"risk:{f.rule_id}" in withheld:
                continue
            if f.outcome == "triggered":
                out.append(Recommendation(code="risk:triggered", text=_risk(f.rule_name, f.category, f.severity), source="finding", ref=f.rule_id))
            elif f.outcome == "needs_review":
                out.append(Recommendation(code="risk:needs_review", text=_needs_check(f.rule_name), source="finding", ref=f.rule_id))

    return out