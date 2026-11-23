from app.config import get_settings
from app.schemas.document import BuiltContext, EvidenceItem, RetrievalResult


def build_context(result: RetrievalResult, max_chars: int | None = None) -> BuiltContext:
    budget = max_chars or get_settings().context_max_chars
    evidence: list[EvidenceItem] = []
    blocks: list[str] = []
    used = 0
    truncated = False

    for i, chunk in enumerate(result.chunks, start=1):
        label = f"E{i}"
        block = f"[{label}] (page {chunk.page_number})\n{chunk.text}"
        if used + len(block) > budget:
            truncated = True
            break
        blocks.append(block)
        used += len(block) + 2
        evidence.append(
            EvidenceItem(
                label=label,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                text=chunk.text,
                score=chunk.score,
            )
        )

    text = "<evidence>\n" + "\n\n".join(blocks) + "\n</evidence>" if blocks else ""
    return BuiltContext(
        text=text,
        evidence=evidence,
        truncated=truncated,
        guardrail_findings=list(result.guardrail_findings),
    )