import logging
import re

from app.agents.base_agent import BaseAgent
from app.agents.document_qa.schemas import DocumentAnswer, DocumentQuestionInput
from app.knowledge.citation_manager import extract_citations
from app.llm.prompt_manager import NOT_FOUND_MESSAGE, build_document_qa_messages
from app.schemas.document import BuiltContext
from app.utils.text_utils import numbers_in

logger = logging.getLogger(__name__)

UNVERIFIED_MESSAGE = (
    "I could not verify an answer to this question against the document. "
    "Please review the document directly."
)
MAX_OUTPUT_TOKENS = 1500
_LABEL = re.compile(r"\[E\d+\]")


def _unverified(reason: str, base: dict) -> DocumentAnswer:
    return DocumentAnswer(
        status="unverified", answer=UNVERIFIED_MESSAGE, citations=[], rejected_reason=reason, **base
    )


def check_answer(text: str, ctx: BuiltContext, base: dict) -> DocumentAnswer:
    """Code-only checks on the model's reply. Pure function, easy to test."""
    if not text:
        return _unverified("empty answer", base)

    report = extract_citations(text, ctx)
    says_not_found = NOT_FOUND_MESSAGE in text

    if says_not_found and not report.has_citations and not report.invalid_labels:
        return DocumentAnswer(status="not_found", answer=NOT_FOUND_MESSAGE, citations=[], **base)
    if says_not_found:
        return _unverified("answer mixes 'not found' with cited content", base)
    if report.invalid_labels:
        return _unverified("cites unknown evidence: " + ", ".join(report.invalid_labels), base)
    if not report.has_citations:
        return _unverified("answer has no citations", base)

    cited = {c.label for c in report.citations}
    cited_text = " ".join(e.text for e in ctx.evidence if e.label in cited)
    missing = numbers_in(_LABEL.sub("", text)) - numbers_in(cited_text)
    if missing:
        return _unverified(
            "number(s) not in the cited evidence: " + ", ".join(f"{n:g}" for n in sorted(missing)), base
        )

    return DocumentAnswer(status="answered", answer=text, citations=report.citations, **base)


class DocumentQAAgent(BaseAgent[DocumentQuestionInput, DocumentAnswer]):
    name = "document_qa"
    description = "Answers a question about one document, using only retrieved evidence"
    input_schema = DocumentQuestionInput
    output_schema = DocumentAnswer

    async def execute(self, data: DocumentQuestionInput) -> DocumentAnswer:
        ctx = data.context
        base = {
            "document_id": data.document_id,
            "evidence_labels": [e.label for e in ctx.evidence],
            "context_truncated": ctx.truncated,
        }
        if not ctx.evidence:  # nothing retrieved: no LLM call, no cost
            return DocumentAnswer(status="not_found", answer=NOT_FOUND_MESSAGE, citations=[], **base)

        result = await self.ask(
            build_document_qa_messages(data.question, ctx), max_tokens=MAX_OUTPUT_TOKENS
        )
        answer = check_answer(result.text.strip(), ctx, base)
        logger.info("Document QA status=%s", answer.status)
        return answer