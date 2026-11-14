import logging

from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.knowledge import vector_store
from app.knowledge.embedding_service import embed_query
from app.schemas.document import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


async def retrieve(
    question: str,
    document_id: str,
    top_k: int | None = None,
) -> RetrievalResult:
    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    # Embedding is CPU-bound: keep it off the event loop
    query_vector = await run_in_threadpool(embed_query, question)
    hits = await vector_store.search(query_vector, document_id, k)

    chunks = [
        RetrievedChunk(
            document_id=h["document_id"],
            chunk_index=h["chunk_index"],
            page_number=h["page_number"],
            text=h["text"],
            score=h["score"],
            clause_types=h.get("metadata", {}).get("clause_types", []),
        )
        for h in hits
        if h["score"] >= settings.retrieval_min_score
    ]

    logger.info("Retrieved %d/%d chunks (top_k=%d)", len(chunks), len(hits), k)
    return RetrievalResult(question=question, found=bool(chunks), chunks=chunks)