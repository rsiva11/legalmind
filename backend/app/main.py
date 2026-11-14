from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database.mongodb import (
    close_mongo_connection,
    connect_to_mongo,
    ping_database,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()      # runs on startup
    yield
    await close_mongo_connection()  # runs on shutdown


app = FastAPI(
    title=settings.app_name,
    description="Multi-Agent Contract Intelligence & Compliance System",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
async def health():
    db_ok = await ping_database()
    body = {
        "status": "ok" if db_ok else "degraded",
        "app": settings.app_name,
        "environment": settings.app_env,
        "database": "up" if db_ok else "down",
    }
    return JSONResponse(body, status_code=200 if db_ok else 503)