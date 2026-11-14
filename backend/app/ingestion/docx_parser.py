from pathlib import Path

from docx import Document

from app.schemas.document import ParsedPage


def parse_docx(path: Path) -> list[ParsedPage]:
    doc = Document(str(path))
    parts: list[str] = [p.text for p in doc.paragraphs if p.text.strip()]

    # Contracts often put key terms in tables
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return [ParsedPage(page_number=1, text="\n".join(parts))]