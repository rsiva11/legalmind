import re
import unicodedata
from collections import Counter

from app.schemas.document import ParsedDocument, ParsedPage

# A line repeated on at least this share of pages is treated as header/footer
REPEAT_THRESHOLD = 0.5
MIN_PAGES_FOR_REPEAT_DETECTION = 3
MAX_HEADER_FOOTER_LEN = 80


def _normalize_chars(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")                      # soft hyphen
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)  # zero-width chars
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _looks_like_content(line: str) -> bool:
    """Lines that are probably real contract text, never boilerplate."""
    return bool(
        re.match(r"(\d+(\.\d+)*[.)]?\s|\([a-zA-Z0-9]{1,4}\)\s)", line)
        or line.endswith("-")
    )


def _find_repeated_lines(pages: list[ParsedPage]) -> set[str]:
    """Lines that appear on many pages are almost always headers/footers."""
    if len(pages) < MIN_PAGES_FOR_REPEAT_DETECTION:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.text.split("\n") if ln.strip()]
        if not lines:
            continue
        edge = {lines[0], lines[-1]}
        counts.update(ln for ln in edge if not _looks_like_content(ln))

    min_pages = len(pages) * REPEAT_THRESHOLD
    return {
        line for line, n in counts.items()
        if n >= min_pages and len(line) <= MAX_HEADER_FOOTER_LEN
    }


def _strip_page_numbers(line: str) -> bool:
    return bool(re.fullmatch(r"(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?", line.strip(), re.I))


def _clean_page(text: str, repeated: set[str]) -> str:
    text = _normalize_chars(text)

    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in repeated or _strip_page_numbers(stripped):
            continue
        kept.append(stripped)
    text = "\n".join(kept)

    # Re-join words split by a line-end hyphen: "termi-\nnation" -> "termination"
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)

    # Re-join hard-wrapped lines inside a paragraph, but keep real structure
    text = re.sub(
        r"(?<![.:;!?\n])\n(?!\n|\d+(\.\d+)*[.)]?\s|\([a-zA-Z0-9]{1,4}\)\s|[•\-*]\s)",
        " ",
        text,
    )

    text = re.sub(r"[ \t]+", " ", text)      # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse blank lines
    return text.strip()


def clean_document(doc: ParsedDocument) -> ParsedDocument:
    repeated = _find_repeated_lines(doc.pages)
    cleaned_pages = [
        ParsedPage(page_number=p.page_number, text=_clean_page(p.text, repeated))
        for p in doc.pages
    ]
    return doc.model_copy(update={"pages": cleaned_pages})