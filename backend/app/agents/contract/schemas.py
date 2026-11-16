from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.document import BuiltContext

# Must match the keys of CLAUSE_KEYWORDS in ingestion/metadata_extractor.py
ClauseType = Literal[
    "termination",
    "confidentiality",
    "payment",
    "liability",
    "indemnity",
    "renewal",
    "governing_law",
    "dispute_resolution",
    "intellectual_property",
    "force_majeure",
    "data_protection",
]


class EvidenceRef(BaseModel):
    label: str = Field(pattern=r"^E\d+$", description="Evidence label, for example E1")
    quote: str = Field(
        min_length=3,
        max_length=300,
        description="Short exact quote (at most 25 words) copied from that evidence",
    )


class ExtractedFact(BaseModel):
    value: str = Field(max_length=300, description="The fact as written in the document")
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=3)


class Party(BaseModel):
    name: str = Field(max_length=200)
    role: str | None = Field(None, description="Role as written, e.g. Client or Disclosing Party")
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=3)


class ExtractedClause(BaseModel):
    clause_type: ClauseType
    summary: str = Field(max_length=500, description="Neutral summary using only the evidence")
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=3)


class Obligation(BaseModel):
    obligated_party: str = Field(max_length=200)
    action: str = Field(max_length=400)
    deadline: str | None = Field(None, description="Deadline as written, or null if none is stated")
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=3)


class ContractExtraction(BaseModel):
    """What the LLM produces. Every field is None or empty if not stated in the evidence."""

    parties: list[Party] = Field(default_factory=list, max_length=10)
    effective_date: ExtractedFact | None = None
    term_or_expiry: ExtractedFact | None = None
    termination_notice: ExtractedFact | None = None
    payment_terms: ExtractedFact | None = None
    governing_law: ExtractedFact | None = None
    clauses: list[ExtractedClause] = Field(default_factory=list, max_length=15)
    obligations: list[Obligation] = Field(default_factory=list, max_length=15)


class ContractAnalysisInput(BaseModel):
    document_id: str
    context: BuiltContext
    focus: list[ClauseType] | None = None  # None = extract everything supported by the evidence


class ContractAnalysis(ContractExtraction):
    """What the agent returns: the extraction plus fields set by code."""

    document_id: str
    evidence_labels: list[str]
    context_truncated: bool
    dropped_unverified: int = 0