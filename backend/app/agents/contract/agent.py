import logging
import re
import unicodedata

from app.agents.base_agent import BaseAgent
from app.agents.contract.prompts import build_messages
from app.agents.contract.schemas import (
    ContractAnalysis,
    ContractAnalysisInput,
    ContractExtraction,
    EvidenceRef,
)
from app.schemas.document import BuiltContext

from app.utils.text_utils import good_refs as _good_refs, norm as _norm

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 3500


def _ground(extraction: ContractExtraction, context: BuiltContext) -> tuple[ContractExtraction, int]:
    lookup = {e.label: _norm(e.text) for e in context.evidence}
    dropped = 0

    def keep_fact(fact):
        nonlocal dropped
        if fact is None:
            return None
        refs = _good_refs(fact.evidence, lookup)
        if not refs:
            dropped += 1
            return None
        return fact.model_copy(update={"evidence": refs})

    def keep_items(items):
        nonlocal dropped
        kept = []
        for item in items:
            refs = _good_refs(item.evidence, lookup)
            if refs:
                kept.append(item.model_copy(update={"evidence": refs}))
            else:
                dropped += 1
        return kept

    grounded = extraction.model_copy(
        update={
            "parties": keep_items(extraction.parties),
            "effective_date": keep_fact(extraction.effective_date),
            "term_or_expiry": keep_fact(extraction.term_or_expiry),
            "termination_notice": keep_fact(extraction.termination_notice),
            "payment_terms": keep_fact(extraction.payment_terms),
            "governing_law": keep_fact(extraction.governing_law),
            "clauses": keep_items(extraction.clauses),
            "obligations": keep_items(extraction.obligations),
        }
    )
    return grounded, dropped


class ContractAgent(BaseAgent[ContractAnalysisInput, ContractAnalysis]):
    name = "contract"
    description = "Extracts parties, dates, clauses and obligations from retrieved contract evidence"
    input_schema = ContractAnalysisInput
    output_schema = ContractAnalysis

    async def execute(self, data: ContractAnalysisInput) -> ContractAnalysis:
        labels = [e.label for e in data.context.evidence]
        base = {
            "document_id": data.document_id,
            "evidence_labels": labels,
            "context_truncated": data.context.truncated,
        }
        if not labels:
            return ContractAnalysis(**base)  # nothing to analyse: skip the LLM entirely

        extraction = await self.ask_structured(
            build_messages(data.context.text, data.focus),
            ContractExtraction,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        grounded, dropped = _ground(extraction, data.context)

        if data.focus:
            grounded = grounded.model_copy(
                update={"clauses": [c for c in grounded.clauses if c.clause_type in data.focus]}
            )
        if dropped:
            logger.warning("Contract agent dropped %d unverified items", dropped)

        return ContractAnalysis(**grounded.model_dump(), **base, dropped_unverified=dropped)