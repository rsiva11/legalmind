import logging
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from app.ingestion.ingestion_pipeline import ingest_document
from app.knowledge.knowledge_manifest import LoadedManifest
from app.knowledge.knowledge_store import KnowledgeBackend

logger = logging.getLogger(__name__)


class SourceIngestResult(BaseModel):
    source_id: str
    status: Literal["ingested", "failed"]
    chunks: int = 0
    withheld: int = 0
    error: str | None = None


class KnowledgeIngestReport(BaseModel):
    sources: list[SourceIngestResult]
    unregistered_files: list[str] = []

    @property
    def ok(self) -> bool:
        return all(s.status == "ingested" for s in self.sources)


async def ingest_knowledge(
    manifest: LoadedManifest,
    backend: KnowledgeBackend,
    *,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> KnowledgeIngestReport:
    if embed_fn is None:
        from app.knowledge.embedding_service import embed_documents  # lazy: loads the model library

        embed_fn = embed_documents

    results: list[SourceIngestResult] = []
    for source in manifest.sources:
        try:
            ingested = ingest_document(manifest.path_of(source))  # same cleaning, chunking, injection scan
            vectors = embed_fn([c.text for c in ingested.chunks])
            docs = [
                {
                    "source_id": source.id,
                    "source_sha256": source.sha256,
                    "jurisdiction": source.jurisdiction,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "text": c.text,
                    "metadata": c.metadata.model_dump(),
                    "embedding": v,
                }
                for c, v in zip(ingested.chunks, vectors)
            ]
            # Replace only after everything succeeded, so a failure never deletes the old version
            await backend.replace_source(source.id, docs)
            results.append(
                SourceIngestResult(
                    source_id=source.id, status="ingested", chunks=len(docs), withheld=ingested.withheld_count
                )
            )
        except Exception as exc:
            logger.error("Knowledge ingestion failed for %s: %s", source.id, type(exc).__name__)
            results.append(
                SourceIngestResult(
                    source_id=source.id, status="failed", error=f"{type(exc).__name__}: {str(exc)[:200]}"
                )
            )
    return KnowledgeIngestReport(sources=results, unregistered_files=manifest.unregistered_files)