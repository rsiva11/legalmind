from dataclasses import dataclass

from app.guardrails.output_guardrail import NAME, Profile, check_text
from app.guardrails.result import GuardrailFinding, GuardrailResult
from app.orchestration.state import WorkflowState
from app.schemas.document import BuiltContext

from app.guardrails.input_guardrail import InputCheck, check_request

from app.guardrails.compliance_guardrail import screen_compliance_report
from app.guardrails.risk_guardrail import screen_risk_report

from app.guardrails.tool_guardrail import ToolGuardrail, ToolSession


@dataclass
class OutputItem:
    field: str
    text: str
    profile: Profile
    source: str = ""
    labels: set[str] | None = None


def collect_output_items(state: WorkflowState, context: BuiltContext | None) -> list[OutputItem]:
    """Gather the model-written free text. Fixed code-written strings and verified facts are not screened."""
    evidence = {e.label: e.text for e in context.evidence} if context else {}
    labels = set(evidence) if context else None
    items: list[OutputItem] = []

    la = state.legal_answer
    if la is not None and la.status in ("answered", "out_of_scope"):  # not_found and unverified are fixed strings
        passages = state.research.passages if state.research is not None else None
        source = " ".join(p.text for p in passages) if passages else ""
        k_labels = {p.label for p in passages} if passages is not None else None
        items.append(OutputItem("legal_answer.answer", la.answer, "general", source, k_labels))
        items += [
            OutputItem(f"legal_answer.key_points[{i}]", p, "general", source, k_labels)
            for i, p in enumerate(la.key_points)
        ]

    da = state.document_answer
    if da is not None and da.status == "answered":  # not_found and unverified are fixed strings
        cited = " ".join(evidence[c.label] for c in da.citations if c.label in evidence)
        items.append(
            OutputItem("document_answer.answer", da.answer, "document", cited or " ".join(evidence.values()), labels)
        )

    for prefix, report in (("compliance", state.compliance), ("risk", state.risk)):
        if report is None:
            continue
        for f in report.findings:
            src = " ".join([f.detail, *(e.quote for e in f.evidence)])
            for name, text in (("explanation", f.explanation), ("suggested_action", f.suggested_action)):
                if text:
                    items.append(OutputItem(f"{prefix}.{f.rule_id}.{name}", text, "commentary", src))

    if state.risk is not None:
        for i, c in enumerate(state.risk.candidates):
            src = " ".join(e.quote for e in c.evidence)
            items.append(OutputItem(f"risk.candidates[{i}]", f"{c.title}. {c.description}", "commentary", src))

    s = state.summary
    if s is not None and s.overview_source == "model":  # the template overview is code-written
        lines = [line.text for line in [*s.key_facts, *s.compliance, *(s.risk.lines if s.risk else [])]]
        items.append(
            OutputItem("summary.overview", s.overview, "commentary", " ".join([*evidence.values(), *lines]), labels)
        )
    return items


class GuardrailManager:
    """One entry point for the workflows. Input, document and tool checks are added here in later steps."""

    def check_outputs(self, state: WorkflowState, context: BuiltContext | None = None) -> GuardrailResult:
        findings: list[GuardrailFinding] = []
        for item in collect_output_items(state, context):
            findings += check_text(item.field, item.text, item.profile, item.source, item.labels)
        return GuardrailResult(guardrail=NAME, findings=findings)

    def __init__(self, tools: ToolGuardrail | None = None) -> None:
        self.tools = tools if tools is not None else ToolGuardrail()

    def tool_session(self, agent: str, document_id: str | None = None) -> ToolSession:
        return self.tools.session(agent, document_id)

    def check_input(self, question: str | None, document_id: str | None) -> InputCheck:
        return check_request(question, document_id)

    def screen_compliance(self, report):
        return screen_compliance_report(report)

    def screen_risk(self, report):
        return screen_risk_report(report)