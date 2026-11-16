import logging

from app.agents.base_agent import BaseAgent
from app.agents.risk.prompts import build_messages
from app.agents.risk.schemas import (
    CandidateRisk,
    ReviewedRisk,
    RiskDraft,
    RiskInput,
    RiskReport,
)
from app.business_logic.risk_rules import evaluate_risks, load_risk_rules
from app.business_logic.scoring_engine import score_risks
from app.schemas.document import BuiltContext
from app.utils.text_utils import good_refs, norm

logger = logging.getLogger(__name__)

EXPLAINABLE = ("triggered", "needs_review")
UNASSESSED_ACTION = "Not enough evidence to assess this risk. Run a targeted search or review the document manually."
MAX_OUTPUT_TOKENS = 2500


def _ground_candidates(
    candidates: list[CandidateRisk], context: BuiltContext
) -> tuple[list[CandidateRisk], int]:
    lookup = {e.label: norm(e.text) for e in context.evidence}
    kept: list[CandidateRisk] = []
    for c in candidates:
        refs = good_refs(c.evidence, lookup)
        if refs:
            kept.append(c.model_copy(update={"evidence": refs}))
    return kept, len(candidates) - len(kept)


class RiskAgent(BaseAgent[RiskInput, RiskReport]):
    name = "risk"
    description = "Scores contract risks with deterministic rules and explains them"
    input_schema = RiskInput
    output_schema = RiskReport

    async def execute(self, data: RiskInput) -> RiskReport:
        rules = data.rules if data.rules is not None else load_risk_rules()
        findings = evaluate_risks(data.analysis, rules)  # deterministic: final
        score = score_risks(findings)  # deterministic: final

        to_explain = [f for f in findings if f.outcome in EXPLAINABLE]
        ctx = data.context
        use_candidates = data.include_candidates and ctx is not None and bool(ctx.evidence)

        explanations = {}
        candidates: list[CandidateRisk] = []
        dropped = 0

        if to_explain or use_candidates:
            draft = await self.ask_structured(
                build_messages(to_explain, ctx.text if use_candidates else None),
                RiskDraft,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            allowed = {f.rule_id for f in to_explain}
            explanations = {e.rule_id: e for e in draft.explanations if e.rule_id in allowed}
            if allowed - explanations.keys():
                logger.warning("Risk agent: %d findings got no explanation", len(allowed - explanations.keys()))
            if use_candidates:
                candidates, dropped = _ground_candidates(draft.candidates, ctx)
                if dropped:
                    logger.warning("Risk agent dropped %d unverified candidates", dropped)

        reviewed = []
        for f in findings:
            e = explanations.get(f.rule_id)
            action = e.suggested_action if e else None
            if f.outcome == "unassessed":
                action = UNASSESSED_ACTION
            reviewed.append(
                ReviewedRisk(
                    **f.model_dump(),
                    explanation=e.explanation if e else None,
                    suggested_action=action,
                )
            )

        return RiskReport(
            document_id=data.document_id,
            findings=reviewed,
            score=score,
            candidates=candidates,
            candidates_dropped=dropped,
            rules_evaluated=len(rules),
        )