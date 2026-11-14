import logging

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: AsyncMongoClient | None = None


async def connect_to_mongo() -> None:
    """Create the shared client and verify the database is reachable."""
    global _client
    settings = get_settings()

    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is not set. Check backend/.env")

    _client = AsyncMongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,
    )
    await _client.admin.command("ping")  # fail fast if unreachable
    logger.info("Connected to MongoDB (db=%s)", settings.mongodb_db_name)


async def close_mongo_connection() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
        logger.info("MongoDB connection closed")


def get_database() -> AsyncDatabase:
    """Used by repositories to access collections."""
    if _client is None:
        raise RuntimeError("MongoDB client is not initialised")
    return _client[get_settings().mongodb_db_name]


async def ping_database() -> bool:
    """Used by /health. Returns False instead of raising."""
    if _client is None:
        return False
    try:
        await _client.admin.command("ping")
        return True
    except PyMongoError:
        logger.exception("MongoDB ping failed")
        return False