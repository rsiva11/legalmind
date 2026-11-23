from app.agents.risk.schemas import RiskReport
from app.guardrails.output_guardrail import (
    ACTION_FALLBACK,
    EXPLANATION_FALLBACK,
    check_text,
    screen_replace,
)
from app.guardrails.result import GuardrailFinding

NAME = "risk"


def screen_risk_report(report: RiskReport) -> tuple[RiskReport, list[GuardrailFinding]]:
    findings: list[GuardrailFinding] = []
    screened = []
    for f in report.findings:
        source = " ".join([f.detail, *(e.quote for e in f.evidence)])
        explanation, fx, r1 = screen_replace(
            NAME, f"risk.{f.rule_id}.explanation", f.explanation, EXPLANATION_FALLBACK, source
        )
        action, fy, r2 = screen_replace(
            NAME, f"risk.{f.rule_id}.suggested_action", f.suggested_action, ACTION_FALLBACK, source
        )
        findings += fx + fy
        screened.append(
            f.model_copy(update={"explanation": explanation, "suggested_action": action}) if (r1 or r2) else f
        )

    kept = []
    for i, c in enumerate(report.candidates):
        field = f"risk.candidates[{i}]"
        found = check_text(field, f"{c.title}. {c.description}", "commentary", " ".join(e.quote for e in c.evidence))
        blocking = sorted({x.code for x in found if x.action == "block"})
        if blocking:  # a suggestion cannot be "replaced": drop it and record why
            findings.append(
                GuardrailFinding(
                    guardrail=NAME, action="flag", code="candidate_dropped", field=field,
                    detail="Suggested risk was dropped; reason: " + ", ".join(blocking),
                )
            )
        else:
            kept.append(c)
            findings += [x.model_copy(update={"guardrail": NAME}) for x in found]
    return report.model_copy(update={"findings": screened, "candidates": kept}), findings