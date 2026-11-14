from pathlib import Path

from pypdf import PdfReader

from app.schemas.document import ParsedPage


def parse_pdf(path: Path) -> list[ParsedPage]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError("PDF is password-protected")

    pages: list[ParsedPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(ParsedPage(page_number=index, text=text))
    return pages