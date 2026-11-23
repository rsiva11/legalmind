import logging
from collections import Counter
from collections.abc import Callable

from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.guardrails.document_guardrail import injection_finding
from app.knowledge.knowledge_manifest import LoadedManifest
from app.knowledge.knowledge_store import KnowledgeBackend
from app.schemas.knowledge import KnowledgeHit

logger = logging.getLogger(__name__)


class KnowledgeSearchResult(BaseModel):
    hits: list[KnowledgeHit]
    withheld: int = 0
    dropped: int = 0
    dropped_reasons: dict[str, int] = {}


async def search_knowledge(
    query: str,
    manifest: LoadedManifest,
    backend: KnowledgeBackend,
    embed_query_fn: Callable[[str], list[float]],
    *,
    top_k: int | None = None,
) -> KnowledgeSearchResult:
    settings = get_settings()
    k = top_k or settings.knowledge_top_k
    vector = await run_in_threadpool(embed_query_fn, query)
    raw = await backend.search(vector, k, manifest.source_ids)

    hits: list[KnowledgeHit] = []
    reasons: Counter[str] = Counter()
    withheld = 0
    for doc in raw[:k]:  # never trust the backend to honour the limit or the source filter
        try:
            source = manifest.get(doc["source_id"])
            if source is None:
                reasons["unregistered_source"] += 1
                continue
            if doc["source_sha256"] != source.sha256:
                reasons["stale_source"] += 1  # chunk is from an older version of the file
                continue
            if doc["score"] < settings.knowledge_min_score:
                reasons["below_min_score"] += 1
                continue
            finding = injection_finding("retrieval", doc["chunk_index"], doc["text"])
            if finding is not None and finding.action == "block":
                withheld += 1
                continue
            hits.append(
                KnowledgeHit(
                    source_id=source.id, chunk_index=doc["chunk_index"], page_number=doc["page_number"],
                    text=doc["text"], score=doc["score"],
                )
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            reasons["malformed"] += 1

    logger.info("Knowledge search: hits=%d withheld=%d dropped=%d", len(hits), withheld, sum(reasons.values()))
    return KnowledgeSearchResult(
        hits=hits, withheld=withheld, dropped=sum(reasons.values()), dropped_reasons=dict(reasons)
    )