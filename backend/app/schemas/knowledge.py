from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):
    source_id: str
    title: str
    publisher: str
    url: str
    jurisdiction: str
    version_note: str | None = None


class KnowledgePassage(BaseModel):
    label: str = Field(pattern=r"^K\d+$")
    source_id: str
    source_title: str
    source_url: str
    jurisdiction: str
    version_note: str | None = None
    chunk_index: int
    page_number: int
    text: str
    score: float


class KnowledgeCitation(BaseModel):
    label: str
    source_id: str
    source_title: str
    source_url: str
    version_note: str | None = None
    page_number: int
    chunk_index: int


class KnowledgeHit(BaseModel):
    source_id: str
    chunk_index: int
    page_number: int
    text: str
    score: float


class SearchKnowledgeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=500)
    top_k: int = Field(4, ge=1, le=8)


class SearchKnowledgeOutput(BaseModel):
    hits: list[KnowledgeHit] = Field(default_factory=list, max_length=8)
    withheld: int = 0
    dropped: int = 0