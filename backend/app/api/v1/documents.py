from fastapi import APIRouter, File, UploadFile

from app.schemas.document import DocumentUploadResponse
from app.services.document_service import save_upload

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    return await save_upload(file)