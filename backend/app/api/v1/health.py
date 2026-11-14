from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database.mongodb import ping_database

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> JSONResponse:
    settings = get_settings()
    db_ok = await ping_database()
    body = {
        "status": "ok" if db_ok else "degraded",
        "app": settings.app_name,
        "environment": settings.app_env,
        "database": "up" if db_ok else "down",
    }
    return JSONResponse(body, status_code=200 if db_ok else 503)