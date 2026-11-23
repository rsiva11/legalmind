import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.config import get_settings
from app.knowledge.context_builder import build_context
from app.knowledge.retriever import retrieve
from app.schemas.document import BuiltContext, RetrievalResult

logger = logging.getLogger(__name__)

# Keys here are checked against the contract schema in tests/, because knowledge/ must not import agents/
FACT_QUERIES = {
    "parties": "Who are the parties to this agreement and what are their roles?",
    "effective_date": "What is the effective date of the agreement?",
    "term_or_expiry": "What is the term or expiry date of the agreement?",
    "termination_notice": "What notice period is required to terminate the agreement?",
    "payment_terms": "What are the payment terms and when are invoices due?",
    "governing_law": "Which law governs the agreement and which courts have jurisdiction?",
}
CLAUSE_QUERIES = {
    "termination": "termination of the agreement, grounds and consequences",
    "confidentiality": "confidentiality and non-disclosure obligations",
    "payment": "fees, invoices, payment and late payment interest",
    "liability": "limitation of liability and damages cap",
    "indemnity": "indemnification and hold harmless obligations",
    "renewal": "automatic renewal and extension of the term",
    "governing_law": "governing law and jurisdiction",
    "dispute_resolution": "dispute resolution, arbitration or mediation",
    "intellectual_property": "intellectual property ownership and licence",
    "force_majeure": "force majeure events",
    "data_protection": "personal data protection and data processing",
}

RetrieveFn = Callable[..., Awaitable[RetrievalResult]]


async def retrieve_for_analysis(
    document_id: str,
    *,
    retrieve_fn: RetrieveFn = retrieve,
    top_k: int | None = None,
    concurrency: int = 4,
) -> RetrievalResult:
    k = top_k or get_settings().analysis_top_k_per_query
    queries = list(dict.fromkeys([*FACT_QUERIES.values(), *CLAUSE_QUERIES.values()]))

    sem = asyncio.Semaphore(concurrency)

    async def one(query: str) -> RetrievalResult:
        async with sem:
            return await retrieve_fn(query, document_id, k)

    results = await asyncio.gather(*(one(q) for q in queries))

    merged: dict[int, dict] = {}
    for res in results:
        for c in res.chunks:
            entry = merged.setdefault(c.chunk_index, {"chunk": c, "hits": 0, "best": c.score})
            entry["hits"] += 1
            if c.score > entry["best"]:
                entry["best"], entry["chunk"] = c.score, c

    # The same bad chunk can come back for several queries: report it once
    findings = list(
        {(f.code, f.field, f.action): f for res in results for f in res.guardrail_findings}.values()
    )

    # Most-matched chunks first, so budget cuts drop the least-matched ones
    ordered = sorted(merged.values(), key=lambda e: (-e["hits"], -e["best"], e["chunk"].chunk_index))
    chunks = [e["chunk"] for e in ordered]
    logger.info("Analysis retrieval: %d queries, %d unique chunks", len(queries), len(chunks))
    return RetrievalResult(
        question="full analysis", found=bool(chunks), chunks=chunks, guardrail_findings=findings
    )

async def build_analysis_context(document_id: str, **kwargs) -> BuiltContext:
    result = await retrieve_for_analysis(document_id, **kwargs)
    return build_context(result, max_chars=get_settings().analysis_context_max_chars)