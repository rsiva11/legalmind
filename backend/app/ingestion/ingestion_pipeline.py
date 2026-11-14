import logging
from pathlib import Path

from app.ingestion.chunker import chunk_document
from app.ingestion.document_loader import load_document
from app.ingestion.metadata_extractor import enrich_chunks
from app.ingestion.text_cleaner import clean_document
from app.schemas.document import IngestionResult

logger = logging.getLogger(__name__)


def ingest_document(path: Path) -> IngestionResult:
    parsed = load_document(path)

    if parsed.needs_ocr:
        # OCR service is not built yet. Fail loudly instead of indexing empty text.
        raise ValueError("Document appears to be scanned and needs OCR")

    cleaned = clean_document(parsed)
    chunks = enrich_chunks(chunk_document(cleaned))

    if not chunks:
        raise ValueError("No text could be extracted from the document")

    logger.info("Ingested pages=%d chunks=%d", parsed.page_count, len(chunks))

    return IngestionResult(
        source_path=str(path),
        page_count=parsed.page_count,
        chunk_count=len(chunks),
        needs_ocr=parsed.needs_ocr,
        chunks=chunks,
    )