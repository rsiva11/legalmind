import logging

from app.agents.base_agent import BaseAgent
from app.agents.compliance.prompts import build_messages
from app.agents.compliance.schemas import (
    ComplianceExplanationDraft,
    ComplianceInput,
    ComplianceReport,
    ReviewedFinding,
)
from app.business_logic.compliance_rules import evaluate, load_rules

logger = logging.getLogger(__name__)

EXPLAINABLE = ("non_compliant", "needs_review")
NOT_FOUND_ACTION = "Run a targeted search of the document or review it manually before drawing a conclusion."
MAX_OUTPUT_TOKENS = 2000


class ComplianceAgent(BaseAgent[ComplianceInput, ComplianceReport]):
    name = "compliance"
    description = "Evaluates a contract analysis against policy rules and explains the findings"
    input_schema = ComplianceInput
    output_schema = ComplianceReport

    async def execute(self, data: ComplianceInput) -> ComplianceReport:
        rules = data.rules if data.rules is not None else load_rules()
        findings = evaluate(data.analysis, rules)  # deterministic: statuses are final here

        to_explain = [f for f in findings if f.status in EXPLAINABLE]
        explanations = {}
        if to_explain:
            draft = await self.ask_structured(
                build_messages(to_explain),
                ComplianceExplanationDraft,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            allowed = {f.rule_id for f in to_explain}
            explanations = {e.rule_id: e for e in draft.explanations if e.rule_id in allowed}
            missing = allowed - explanations.keys()
            if missing:
                logger.warning("Compliance agent: %d findings got no explanation", len(missing))

        reviewed: list[ReviewedFinding] = []
        for f in findings:
            e = explanations.get(f.rule_id)
            action = e.suggested_action if e else None
            if f.status == "not_found":
                action = NOT_FOUND_ACTION
            reviewed.append(
                ReviewedFinding(
                    **f.model_dump(),
                    explanation=e.explanation if e else None,
                    suggested_action=action,
                )
            )

        counts: dict[str, int] = {}
        for f in reviewed:
            counts[f.status] = counts.get(f.status, 0) + 1

        return ComplianceReport(
            document_id=data.document_id,
            findings=reviewed,
            status_counts=counts,
            rules_evaluated=len(rules),
        )