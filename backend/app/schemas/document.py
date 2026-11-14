from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: str
    original_filename: str
    extension: str
    size_bytes: int
    sha256: str
    status: str

class ParsedPage(BaseModel):
    page_number: int
    text: str


class ParsedDocument(BaseModel):
    source_path: str
    extension: str
    page_count: int
    pages: list[ParsedPage]
    needs_ocr: bool = False

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

class ChunkMetadata(BaseModel):
    clause_types: list[str] = []
    dates: list[str] = []
    money: list[str] = []
    durations: list[str] = []


class Chunk(BaseModel):
    chunk_index: int
    page_number: int
    text: str
    char_count: int
    metadata: ChunkMetadata = ChunkMetadata()

class IngestionResult(BaseModel):
    source_path: str
    page_count: int
    chunk_count: int
    needs_ocr: bool
    chunks: list[Chunk]

class RetrievedChunk(BaseModel):
    document_id: str
    chunk_index: int
    page_number: int
    text: str
    score: float
    clause_types: list[str] = []


class RetrievalResult(BaseModel):
    question: str
    found: bool
    chunks: list[RetrievedChunk]

class EvidenceItem(BaseModel):
    label: str
    document_id: str
    chunk_index: int
    page_number: int
    text: str
    score: float


class BuiltContext(BaseModel):
    text: str
    evidence: list[EvidenceItem]
    truncated: bool

class Citation(BaseModel):
    label: str
    page_number: int
    chunk_index: int
    document_id: str


class CitationReport(BaseModel):
    citations: list[Citation]
    invalid_labels: list[str]
    has_citations: bool

class RAGResult(BaseModel):
    question: str
    document_id: str
    found: bool
    context: BuiltContext