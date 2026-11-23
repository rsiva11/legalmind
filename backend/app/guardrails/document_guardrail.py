from pydantic import BaseModel

from app.guardrails.prompt_injection import detect_injection
from app.guardrails.result import GuardrailFinding
from app.schemas.document import Chunk

NAME = "document"


def injection_finding(guardrail: str, index: int, text: str) -> GuardrailFinding | None:
    """One finding per chunk. Never contains the chunk's text, only pattern-group names."""
    hits = detect_injection(text)
    if not hits:
        return None
    names = ", ".join(sorted(h.code for h in hits))
    withheld = any(h.severity == "high" for h in hits)
    return GuardrailFinding(
        guardrail=guardrail,
        action="block" if withheld else "flag",
        code="suspicious_content",
        field=f"chunk[{index}]",
        detail=f"{'Withheld from the model' if withheld else 'Kept'}; matched: {names}",
    )


class DocumentScan(BaseModel):
    kept: list[Chunk]
    withheld: int
    needs_review: bool
    findings: list[GuardrailFinding]


def scan_chunks(chunks: list[Chunk]) -> DocumentScan:
    kept: list[Chunk] = []
    findings: list[GuardrailFinding] = []
    for c in chunks:
        finding = injection_finding(NAME, c.chunk_index, c.text)
        if finding is not None:
            findings.append(finding)
        if finding is None or finding.action != "block":
            kept.append(c)
    withheld = len(chunks) - len(kept)
    return DocumentScan(kept=kept, withheld=withheld, needs_review=withheld > 0, findings=findings)