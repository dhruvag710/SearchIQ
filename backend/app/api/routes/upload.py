import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.upload import DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPE = "application/pdf"
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
CHUNK_SIZE_BYTES = 1024 * 1024

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAW_UPLOAD_DIR = PROJECT_ROOT / "data" / "raw"


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload (max 50 MB)"),
) -> DocumentUploadResponse:
    """Accept a PDF file, validate it, and persist it under ``data/raw/``."""
    if file.content_type != ALLOWED_CONTENT_TYPE:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted. Received content type: {file.content_type!r}",
        )

    original_filename = file.filename or "unknown.pdf"
    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only files with a .pdf extension are accepted.",
        )

    document_id = str(uuid.uuid4())
    stored_filename = f"{document_id}.pdf"
    destination = RAW_UPLOAD_DIR / stored_filename

    RAW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file_size_bytes = 0
    try:
        with destination.open("wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE_BYTES):
                file_size_bytes += len(chunk)
                if file_size_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="File exceeds the maximum allowed size of 50 MB.",
                    )
                buffer.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc
    finally:
        await file.close()

    if file_size_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    return DocumentUploadResponse(
        document_id=document_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_size_bytes=file_size_bytes,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        status="uploaded",
    )
