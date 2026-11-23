import logging
from typing import Protocol

from pymongo.operations import SearchIndexModel

from app.config import get_settings
from app.database.mongodb import get_database

logger = logging.getLogger(__name__)


class KnowledgeBackend(Protocol):
    async def replace_source(self, source_id: str, docs: list[dict]) -> int: ...
    async def search(self, vector: list[float], limit: int, source_ids: list[str]) -> list[dict]: ...


class MongoKnowledgeBackend:
    """Legal knowledge lives in its own collection and index, never beside contract chunks."""

    def _col(self):
        return get_database()[get_settings().knowledge_collection]

    async def ensure_index(self) -> None:
        s = get_settings()
        db = get_database()
        if s.knowledge_collection not in await db.list_collection_names():
            await db.create_collection(s.knowledge_collection)
        col = self._col()
        existing = [i["name"] async for i in await col.list_search_indexes()]
        if s.knowledge_vector_index_name in existing:
            return
        model = SearchIndexModel(
            name=s.knowledge_vector_index_name,
            type="vectorSearch",
            definition={
                "fields": [
                    {"type": "vector", "path": "embedding", "numDimensions": s.embedding_dimensions, "similarity": "cosine"},
                    {"type": "filter", "path": "source_id"},
                    {"type": "filter", "path": "jurisdiction"},
                ]
            },
        )
        await col.create_search_index(model)
        logger.info("Created knowledge index %s (builds in background)", s.knowledge_vector_index_name)

    async def replace_source(self, source_id: str, docs: list[dict]) -> int:
        col = self._col()
        await col.delete_many({"source_id": source_id})
        if not docs:
            return 0
        result = await col.insert_many(docs)
        return len(result.inserted_ids)

    async def search(self, vector: list[float], limit: int, source_ids: list[str]) -> list[dict]:
        s = get_settings()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": s.knowledge_vector_index_name,
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": max(limit * 20, 100),
                    "limit": limit,
                    "filter": {"source_id": {"$in": source_ids}},
                }
            },
            {
                "$project": {
                    "_id": 0, "source_id": 1, "source_sha256": 1, "chunk_index": 1,
                    "page_number": 1, "text": 1, "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        cursor = await self._col().aggregate(pipeline)
        return [doc async for doc in cursor]