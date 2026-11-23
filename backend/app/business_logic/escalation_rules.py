from datetime import date
from typing import TYPE_CHECKING

from pydantic import BaseModel

from app.business_logic.contract_rules import analyze_dates
from app.config import get_settings
from app.utils.time_utils import utc_now

if TYPE_CHECKING:  # annotation only: business_logic must not import orchestration at run time
    from app.orchestration.state import WorkflowState

# One fixed sentence per code. The order of this dict is the order reasons are reported in.
MESSAGES: dict[str, str] = {
    "verification_failed": "The automated verification of this analysis failed.",
    "items_withheld": "Some extracted items could not be verified and were withheld from the report.",
    "high_severity_noncompliance": "A high-severity compliance rule was not met.",
    "compliance_needs_review": "A compliance rule could not be judged automatically and needs a person to check it.",
    "high_risk_level": "The risk level from verified items is high or critical.",
    "low_risk_coverage": "Too few risk rules could be assessed, so the risk level may be understated.",
    "many_unresolved_rules": "Many compliance rules could not be resolved from the retrieved text.",
    "suspicious_document_content": "Passages of the document were withheld as suspicious or inconsistent and were not used.",
    "guardrail_intervention": "An automated safety check replaced or removed model-written text.",
    "guardrail_failure": "The job was stopped by an automated safety check.",
    "step_skipped": "A step of the analysis failed and was skipped, so part of the report is missing.",
    "candidate_risks_present": "Unverified risk suggestions were produced and need a person's judgement.",
    "contract_expiring": "The contract has expired or expires within 30 days.",
    "ambiguous_date": "A contract date is written in an ambiguous format and was not interpreted.",
    "escalation_error": "The review check could not run, so this result is marked for review.",
}

AMBIGUOUS_DATE_REASONS = frozenset({"effective_date_ambiguous", "term_ambiguous_date"})
INTERVENTION_CODES = frozenset({"text_replaced", "candidate_dropped"})


class EscalationDecision(BaseModel):
    needs_human_review: bool
    codes: list[str]
    reasons: list[str]


def _decision(codes: set[str]) -> EscalationDecision:
    ordered = [c for c in MESSAGES if c in codes]
    return EscalationDecision(
        needs_human_review=bool(ordered), codes=ordered, reasons=[MESSAGES[c] for c in ordered]
    )


def error_decision() -> EscalationDecision:
    return _decision({"escalation_error"})


def _risk_view(state: "WorkflowState") -> tuple[str, float] | None:
    """The risk level and coverage the user relies on: the summary's view if there is one."""
    summary = state.summary
    if summary is not None and summary.risk is not None:
        return summary.risk.level, summary.risk.coverage
    if state.risk is not None:
        return state.risk.score.level, state.risk.score.coverage
    return None


def evaluate_escalation(state: "WorkflowState", today: date | None = None) -> EscalationDecision:
    """Pure function of the finished state. Evaluates whatever data is present."""
    settings = get_settings()
    today = today or utc_now().date()
    codes: set[str] = set()

    v = state.verification
    if v is not None:
        if v.result == "fail":
            codes.add("verification_failed")
        if v.withheld_item_ids:
            codes.add("items_withheld")

    comp = state.compliance
    if comp is not None and comp.findings:
        if any(f.status == "non_compliant" and f.severity == "high" for f in comp.findings):
            codes.add("high_severity_noncompliance")
        if any(f.status == "needs_review" for f in comp.findings):
            codes.add("compliance_needs_review")
        unresolved = sum(1 for f in comp.findings if f.status in ("not_found", "needs_review"))
        if unresolved / len(comp.findings) >= settings.escalation_max_unresolved_share:
            codes.add("many_unresolved_rules")

    view = _risk_view(state)
    if view is not None:
        level, coverage = view
        if level in ("high", "critical"):
            codes.add("high_risk_level")
        if coverage < settings.escalation_min_risk_coverage:
            codes.add("low_risk_coverage")

    for f in state.guardrail_findings:
        if f.guardrail == "retrieval" and f.action == "block":
            codes.add("suspicious_document_content")
        if f.code in INTERVENTION_CODES:
            codes.add("guardrail_intervention")
    if state.failed_step is not None and state.failed_step.startswith("guardrail:"):
        codes.add("guardrail_failure")

    if state.failed_step is None and any(s.status == "failed" for s in state.steps):
        codes.add("step_skipped")  # a non-fatal agent failure: part of the report is missing

    if state.risk is not None and state.risk.candidates:
        codes.add("candidate_risks_present")

    if state.contract is not None and v is not None:
        verified = {i.item_id for i in v.items if i.verdict == "verified"}
        dates = analyze_dates(state.contract, today, verified)
        if dates.status in ("expired", "expiring_30"):
            codes.add("contract_expiring")
        if AMBIGUOUS_DATE_REASONS & set(dates.reasons):
            codes.add("ambiguous_date")

    return _decision(codes)