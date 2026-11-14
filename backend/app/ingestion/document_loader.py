import logging
from pathlib import Path

from app.ingestion.docx_parser import parse_docx
from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.text_parser import parse_text
from app.schemas.document import ParsedDocument

logger = logging.getLogger(__name__)

PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_text,
}

# Below this many characters per page we assume the PDF is scanned
MIN_CHARS_PER_PAGE = 50


def load_document(path: Path) -> ParsedDocument:
    ext = path.suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"No parser for {ext}")

    pages = parser(path)
    total_chars = sum(len(p.text.strip()) for p in pages)
    needs_ocr = ext == ".pdf" and total_chars < MIN_CHARS_PER_PAGE * max(len(pages), 1)

    logger.info("Parsed %s pages=%d chars=%d needs_ocr=%s", ext, len(pages), total_chars, needs_ocr)

    return ParsedDocument(
        source_path=str(path),
        extension=ext,
        page_count=len(pages),
        pages=pages,
        needs_ocr=needs_ocr,
    )