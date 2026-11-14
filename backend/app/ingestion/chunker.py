import re

from app.config import get_settings
from app.schemas.document import Chunk, ParsedDocument

# A new section starts at a line beginning with 1. / 1.1 / 1.1.1 / Article / Section
# Non-capturing groups (?: ) so re.split never injects fragments into the result
SECTION_START = re.compile(
    r"(?m)^(?=(?:\d+(?:\.\d+)*[.)]?\s|(?:ARTICLE|SECTION|Article|Section)\s+[\dIVXivx]+))"
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def _split_sections(text: str) -> list[str]:
    sections = [s.strip() for s in SECTION_START.split(text) if s and s.strip()]
    return sections or [text.strip()]


def _split_long(section: str, max_chars: int, overlap: int) -> list[str]:
    """Split an oversized section at sentence boundaries with overlap."""
    sentences = SENTENCE_END.split(section)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current.strip())
            current = current[-overlap:] + " " + sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        pieces.append(current.strip())

    # A single sentence longer than max_chars: hard split as a last resort
    final: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            final.append(piece[:max_chars])
            piece = piece[max_chars - overlap:]
        final.append(piece)
    return final


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    settings = get_settings()
    max_chars = settings.chunk_max_chars
    overlap = settings.chunk_overlap_chars

    chunks: list[Chunk] = []

    for page in doc.pages:
        if not page.text.strip():
            continue

        buffer = ""
        for section in _split_sections(page.text):
            if len(section) > max_chars:
                if buffer:
                    chunks.append(_make(len(chunks), page.page_number, buffer))
                    buffer = ""
                for piece in _split_long(section, max_chars, overlap):
                    chunks.append(_make(len(chunks), page.page_number, piece))
            elif len(buffer) + len(section) + 1 > max_chars:
                chunks.append(_make(len(chunks), page.page_number, buffer))
                buffer = section
            else:
                buffer = f"{buffer}\n{section}".strip()

        if buffer:
            chunks.append(_make(len(chunks), page.page_number, buffer))

    return chunks


def _make(index: int, page: int, text: str) -> Chunk:
    text = text.strip()
    return Chunk(chunk_index=index, page_number=page, text=text, char_count=len(text))