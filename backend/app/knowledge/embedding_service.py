import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

# BGE models are trained to expect this prefix on *queries* (not on documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _get_model() -> SentenceTransformer:
    name = get_settings().embedding_model
    logger.info("Loading embedding model %s", name)
    return SentenceTransformer(name)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts for storage."""
    if not texts:
        return []
    vectors = _get_model().encode(
        texts,
        batch_size=get_settings().embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(question: str) -> list[float]:
    """Embed a user question for search."""
    vector = _get_model().encode(
        QUERY_PREFIX + question,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()