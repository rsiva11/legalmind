from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Multi-Agent Contract Intelligence & Compliance System",
    version="0.1.0",
)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
    }