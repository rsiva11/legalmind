from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.api.v1 import health
from app.config import get_settings
from app.database.mongodb import close_mongo_connection, connect_to_mongo

from app.core.logging import setup_logging

setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()


app = FastAPI(
    title=settings.app_name,
    description="Multi-Agent Contract Intelligence & Compliance System",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)      # /health (unversioned, for infra)
app.include_router(api_router)         # /api/v1/... (business endpoints)