import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import get_settings
from app.schemas.document import DocumentUploadResponse
from app.utils.file_utils import (
    clean_display_name,
    get_extension,
    safe_stored_name,
    sha256_of,
)

logger = logging.getLogger(__name__)

# Magic bytes: the first bytes of a real file of that type
PDF_MAGIC = b"%PDF-"
DOCX_MAGIC = b"PK\x03\x04"  # DOCX is a zip archive


async def save_upload(file: UploadFile) -> DocumentUploadResponse:
    settings = get_settings()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = get_extension(file.filename)
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {settings.allowed_extensions}",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)  # read one extra byte to detect oversize
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_mb} MB limit",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Check content matches the claimed extension (extension alone can be faked)
    if ext == ".pdf" and not data.startswith(PDF_MAGIC):
        raise HTTPException(status_code=400, detail="File is not a valid PDF")
    if ext == ".docx" and not data.startswith(DOCX_MAGIC):
        raise HTTPException(status_code=400, detail="File is not a valid DOCX")

    stored_name = safe_stored_name(file.filename)
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / stored_name).write_bytes(data)

    logger.info("Stored upload id=%s size=%d", stored_name, len(data))  # no filename/content in logs

    return DocumentUploadResponse(
        document_id=Path(stored_name).stem,
        original_filename=clean_display_name(file.filename),
        extension=ext,
        size_bytes=len(data),
        sha256=sha256_of(data),
        status="uploaded",
    )