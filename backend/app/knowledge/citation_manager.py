import re

from app.schemas.document import BuiltContext, Citation, CitationReport

LABEL_PATTERN = re.compile(r"\[(E\d+)\]")


def extract_citations(answer: str, context: BuiltContext) -> CitationReport:
    lookup = {e.label: e for e in context.evidence}

    seen: list[str] = []
    for label in LABEL_PATTERN.findall(answer):
        if label not in seen:
            seen.append(label)

    citations = [
        Citation(
            label=label,
            page_number=lookup[label].page_number,
            chunk_index=lookup[label].chunk_index,
            document_id=lookup[label].document_id,
        )
        for label in seen
        if label in lookup
    ]
    invalid = [label for label in seen if label not in lookup]

    return CitationReport(
        citations=citations,
        invalid_labels=invalid,
        has_citations=bool(citations),
    )