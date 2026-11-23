from app.guardrails.document_guardrail import injection_finding
from app.guardrails.result import GuardrailFinding
from app.schemas.document import RetrievedChunk

NAME = "retrieval"


def filter_retrieved(
    chunks: list[RetrievedChunk], document_id: str
) -> tuple[list[RetrievedChunk], list[GuardrailFinding]]:
    kept: list[RetrievedChunk] = []
    findings: list[GuardrailFinding] = []
    for c in chunks:
        if c.document_id != document_id:  # data isolation: never return another document's text
            findings.append(
                GuardrailFinding(
                    guardrail=NAME, action="block", code="wrong_document", field=f"chunk[{c.chunk_index}]",
                    detail="Chunk belongs to a different document and was dropped",
                )
            )
            continue
        finding = injection_finding(NAME, c.chunk_index, c.text)
        if finding is not None:
            findings.append(finding)
            if finding.action == "block":
                continue
        kept.append(c)
    return kept, findings