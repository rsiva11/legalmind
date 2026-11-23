import logging
import re

from app.agents.base_agent import BaseAgent
from app.agents.legal_assistant.prompts import build_messages
from app.agents.legal_assistant.schemas import LegalAnswer, LegalAnswerDraft, LegalQuestionInput
from app.agents.research.schemas import ResearchResult
from app.schemas.knowledge import KnowledgeCitation
from app.utils.text_utils import norm, numbers_in

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is general legal information, not legal advice. "
    "For decisions about your own situation, consult a qualified lawyer."
)
NOT_FOUND_ANSWER = (
    "I could not find this in the legal knowledge base. "
    "It holds only a small set of Indian legal sources, so this does not mean there is no answer."
)
UNVERIFIED_ANSWER = (
    "I could not verify an answer against the legal knowledge base. "
    "Please consult a qualified lawyer or the official source."
)
MAX_OUTPUT_TOKENS = 2000
_LABEL = re.compile(r"\[(K\d+)\]")


def check_legal_answer(draft: LegalAnswerDraft, research: ResearchResult) -> LegalAnswer:
    """Code-only checks on the model's draft. Pure function, easy to test."""
    passages = {p.label: p for p in research.passages}
    common = {
        "evidence_labels": list(passages),
        "disclaimer": DISCLAIMER,
        "jurisdiction_dependent": draft.jurisdiction_dependent,
        "seek_professional_advice": draft.seek_professional_advice,
    }

    def fixed(status: str, text: str, reason: str | None = None) -> LegalAnswer:
        return LegalAnswer(status=status, answer=text, source="none", rejected_reason=reason, **common)

    if not draft.in_scope:
        return LegalAnswer(status="out_of_scope", in_scope=False, answer=draft.answer, source="none", **common)
    if not draft.answered_from_knowledge:
        return fixed("not_found", NOT_FOUND_ANSWER)

    texts = [draft.answer, *draft.key_points]
    cited: list[str] = []
    for text in texts:
        for label in _LABEL.findall(text):
            if label not in cited:
                cited.append(label)

    unknown = [label for label in cited if label not in passages]
    if unknown:
        return fixed("unverified", UNVERIFIED_ANSWER, "cites unknown evidence: " + ", ".join(unknown))
    if not cited:
        return fixed("unverified", UNVERIFIED_ANSWER, "answer has no citations")

    good = [
        q for q in draft.quotes
        if q.label in cited and norm(q.quote) in norm(passages[q.label].text)
    ]
    if not good:
        return fixed("unverified", UNVERIFIED_ANSWER, "no supporting quote could be verified")

    cited_text = " ".join(passages[label].text for label in cited)
    missing = numbers_in(_LABEL.sub("", " ".join(texts))) - numbers_in(cited_text)
    if missing:
        return fixed(
            "unverified", UNVERIFIED_ANSWER,
            "number(s) not in the cited passages: " + ", ".join(f"{n:g}" for n in sorted(missing)),
        )

    citations = [
        KnowledgeCitation(
            label=label, source_id=passages[label].source_id, source_title=passages[label].source_title,
            source_url=passages[label].source_url, version_note=passages[label].version_note,
            page_number=passages[label].page_number, chunk_index=passages[label].chunk_index,
        )
        for label in cited
    ]
    return LegalAnswer(
        status="answered", answer=draft.answer, key_points=draft.key_points, quotes=good,
        citations=citations, source="knowledge_base", **common,
    )


class LegalAssistantAgent(BaseAgent[LegalQuestionInput, LegalAnswer]):
    name = "legal_assistant"
    description = "Answers general legal-information questions using only retrieved knowledge passages"
    input_schema = LegalQuestionInput
    output_schema = LegalAnswer

    async def execute(self, data: LegalQuestionInput) -> LegalAnswer:
        knowledge = data.knowledge
        if knowledge is not None and knowledge.status == "unavailable":
            raise RuntimeError("legal knowledge base unavailable")
        if knowledge is None or not knowledge.passages:  # no evidence: no LLM call and no fallback to memory
            return LegalAnswer(
                status="not_found", answer=NOT_FOUND_ANSWER, source="none", disclaimer=DISCLAIMER
            )
        draft = await self.ask_structured(
            build_messages(data.question, knowledge.context), LegalAnswerDraft, max_tokens=MAX_OUTPUT_TOKENS
        )
        answer = check_legal_answer(draft, knowledge)
        logger.info("Legal assistant status=%s", answer.status)
        return answer