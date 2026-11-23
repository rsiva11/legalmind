from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LegalMind"
    app_env: str = "development"
    log_level: str = "INFO"

    mongodb_uri: str = ""
    mongodb_db_name: str = "legalmind"

    groq_api_key: str = ""
    jwt_secret_key: str = ""

    upload_dir: str = "uploads"
    max_upload_mb: int = 10
    allowed_extensions: list[str] = [".pdf", ".docx", ".txt"]

    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 32

    chunks_collection: str = "chunks"
    vector_index_name: str = "chunk_vector_index"

    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.0

    context_max_chars: int = 6000

    groq_model: str = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = 30.0
    groq_max_retries: int = 2
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    request_token_budget: int = 20000

    policies_dir: str = "../data/company_policies"

    risk_rules_dir: str = "../data/risk_rules"
    risk_threshold_medium: int = 20
    risk_threshold_high: int = 40
    risk_threshold_critical: int = 70

    analysis_top_k_per_query: int = 3
    analysis_context_max_chars: int = 12000
    analysis_token_budget: int = 40000

    jobs_collection: str = "jobs"
    workflow_timeout_seconds: float = 300.0
    max_concurrent_jobs: int = 2

    tool_max_calls_per_request: int = 5
    tool_timeout_seconds: float = 10.0

    escalation_min_risk_coverage: float = 0.5
    escalation_max_unresolved_share: float = 0.5

    knowledge_dir: str = "../data/legal_knowledge"
    knowledge_collection: str = "knowledge_chunks"
    knowledge_vector_index_name: str = "knowledge_vector_index"
    knowledge_allowed_jurisdictions: list[str] = ["India"]
    knowledge_top_k: int = 4
    knowledge_min_score: float = 0.0
    knowledge_context_max_chars: int = 6000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()