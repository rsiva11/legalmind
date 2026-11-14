from pathlib import Path

from app.schemas.document import ParsedPage


def parse_text(path: Path) -> list[ParsedPage]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return [ParsedPage(page_number=1, text=raw.decode(encoding))]
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode text file")