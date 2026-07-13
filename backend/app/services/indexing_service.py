import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.chunking.chunker import chunk_document
from app.chunking.models import Chunk as ChunkDTO
from app.embeddings.service import embed_chunks
from app.ingestion.models import Document as ParsedDocument
from app.ingestion.parser import parse_pdf
from app.models.chunk import Chunk as ChunkRecord
from app.models.document import Document as DocumentRecord


@dataclass(frozen=True)
class IndexingSummary:
    """Result returned after a document has been indexed."""

    document_id: str
    total_pages: int
    total_chunks: int
    status: str


def index_document(
    file_path: Path,
    db: Session,
    *,
    document_id: str | uuid.UUID | None = None,
) -> IndexingSummary:
    """Parse, chunk, embed, and persist a document end to end."""
    resolved_document_id = _resolve_document_id(document_id)
    parsed_document, file_type = _parse_document(file_path)

    chunk_dtos = chunk_document(
        parsed_document,
        document_id=str(resolved_document_id),
    )
    embedded_chunks = embed_chunks(chunk_dtos)

    try:
        _persist_document(
            db=db,
            document_id=resolved_document_id,
            parsed_document=parsed_document,
            file_type=file_type,
        )
        _persist_chunks(
            db=db,
            document_id=resolved_document_id,
            embedded_chunks=embedded_chunks,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    return IndexingSummary(
        document_id=str(resolved_document_id),
        total_pages=parsed_document.total_pages,
        total_chunks=len(embedded_chunks),
        status="indexed",
    )


def _resolve_document_id(document_id: str | uuid.UUID | None) -> uuid.UUID:
    if document_id is None:
        return uuid.uuid4()
    if isinstance(document_id, uuid.UUID):
        return document_id
    return uuid.UUID(document_id)


def _parse_document(file_path: Path) -> tuple[ParsedDocument, str]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path), "pdf"
    raise ValueError(f"Unsupported file type for indexing: {suffix!r}")


def _persist_document(
    *,
    db: Session,
    document_id: uuid.UUID,
    parsed_document: ParsedDocument,
    file_type: str,
) -> None:
    db.add(
        DocumentRecord(
            id=document_id,
            title=parsed_document.document_title,
            file_type=file_type,
            total_pages=parsed_document.total_pages,
        )
    )
    db.flush()


def _persist_chunks(
    *,
    db: Session,
    document_id: uuid.UUID,
    embedded_chunks: list[tuple[ChunkDTO, list[float]]],
) -> None:
    for chunk_dto, embedding in embedded_chunks:
        db.add(
            ChunkRecord(
                id=uuid.UUID(chunk_dto.chunk_id),
                document_id=document_id,
                page_number=chunk_dto.page_number,
                chunk_index=chunk_dto.chunk_index,
                text=chunk_dto.text,
                embedding=embedding,
            )
        )
