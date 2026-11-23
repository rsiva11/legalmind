from app.agents.compliance.schemas import ComplianceReport
from app.guardrails.output_guardrail import ACTION_FALLBACK, EXPLANATION_FALLBACK, screen_replace
from app.guardrails.result import GuardrailFinding

NAME = "compliance"


def screen_compliance_report(report: ComplianceReport) -> tuple[ComplianceReport, list[GuardrailFinding]]:
    findings: list[GuardrailFinding] = []
    screened = []
    for f in report.findings:
        source = " ".join([f.detail, *(e.quote for e in f.evidence)])
        explanation, fx, r1 = screen_replace(
            NAME, f"compliance.{f.rule_id}.explanation", f.explanation, EXPLANATION_FALLBACK, source
        )
        action, fy, r2 = screen_replace(
            NAME, f"compliance.{f.rule_id}.suggested_action", f.suggested_action, ACTION_FALLBACK, source
        )
        findings += fx + fy
        screened.append(
            f.model_copy(update={"explanation": explanation, "suggested_action": action}) if (r1 or r2) else f
        )
    return report.model_copy(update={"findings": screened}), findings