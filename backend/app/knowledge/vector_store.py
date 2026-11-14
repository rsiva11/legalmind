import logging

from pymongo.operations import SearchIndexModel

from app.config import get_settings
from app.database.mongodb import get_database
from app.schemas.document import Chunk

logger = logging.getLogger(__name__)


def _collection():
    return get_database()[get_settings().chunks_collection]


async def ensure_vector_index() -> None:
    """Create the collection and Atlas vector index if they do not exist yet."""
    settings = get_settings()
    db = get_database()

    # Collections are created lazily; a search index needs the collection to exist
    if settings.chunks_collection not in await db.list_collection_names():
        await db.create_collection(settings.chunks_collection)
        logger.info("Created collection %s", settings.chunks_collection)

    col = _collection()

    existing = [i["name"] async for i in await col.list_search_indexes()]
    if settings.vector_index_name in existing:
        return

    model = SearchIndexModel(
        name=settings.vector_index_name,
        type="vectorSearch",
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": settings.embedding_dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "document_id"},
                {"type": "filter", "path": "metadata.clause_types"},
            ]
        },
    )
    await col.create_search_index(model)
    logger.info("Created vector index %s (builds in background)", settings.vector_index_name)


async def add_chunks(document_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must be the same length")

    docs = [
        {
            "document_id": document_id,
            "chunk_index": c.chunk_index,
            "page_number": c.page_number,
            "text": c.text,
            "metadata": c.metadata.model_dump(),
            "embedding": v,
        }
        for c, v in zip(chunks, vectors)
    ]
    result = await _collection().insert_many(docs)
    return len(result.inserted_ids)


async def search(query_vector: list[float], document_id: str, limit: int = 5) -> list[dict]:
    settings = get_settings()
    pipeline = [
        {
            "$vectorSearch": {
                "index": settings.vector_index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(limit * 20, 100),
                "limit": limit,
                "filter": {"document_id": document_id},
            }
        },
        {
            "$project": {
                "_id": 0,
                "document_id": 1,
                "chunk_index": 1,
                "page_number": 1,
                "text": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    cursor = await _collection().aggregate(pipeline)
    return [doc async for doc in cursor]


async def delete_document(document_id: str) -> int:
    result = await _collection().delete_many({"document_id": document_id})
    return result.deleted_count