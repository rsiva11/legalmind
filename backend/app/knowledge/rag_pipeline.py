import logging

from app.knowledge.context_builder import build_context
from app.knowledge.retriever import retrieve
from app.schemas.document import RAGResult

logger = logging.getLogger(__name__)


async def run_rag(question: str, document_id: str, top_k: int | None = None) -> RAGResult:
    retrieval = await retrieve(question, document_id, top_k)
    context = build_context(retrieval)

    found = retrieval.found and bool(context.evidence)
    logger.info("RAG done found=%s evidence=%d truncated=%s", found, len(context.evidence), context.truncated)

    return RAGResult(
        question=question,
        document_id=document_id,
        found=found,
        context=context,
    )