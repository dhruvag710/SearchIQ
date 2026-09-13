import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.chunking.chunker import chunk_document
from app.chunking.models import Chunk as ChunkDTO
from app.embeddings.embedder import BGEEmbedder
from app.embeddings.service import _get_default_embedder, embed_chunks
from app.generation.vlm_service import VisualCaptioner
from app.ingestion.docx_parser import parse_docx
from app.ingestion.markdown_parser import parse_markdown
from app.ingestion.models import Document as ParsedDocument
from app.ingestion.multimodal_parser import MultimodalPdfParser
from app.ingestion.parser import parse_pdf
from app.ingestion.txt_parser import parse_txt
from app.models.chunk import Chunk as ChunkRecord
from app.models.document import Document as DocumentRecord
from app.models.document_image import DocumentImage as DocumentImageRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingSummary:
    """Result returned after a document has been indexed."""

    document_id: str
    total_pages: int
    total_chunks: int
    status: str
    total_images: int = 0


def index_document(
    file_path: Path,
    db: Session,
    *,
    document_id: str | uuid.UUID | None = None,
    enable_multimodal: bool | None = None,
    captioner: VisualCaptioner | None = None,
    embedder: BGEEmbedder | None = None,
) -> IndexingSummary:
    """Parse, chunk, embed, and persist a document end to end."""
    if enable_multimodal is None:
        enable_multimodal = file_path.suffix.lower() == ".pdf"

    resolved_document_id = _resolve_document_id(document_id)
    parsed_document, file_type = _parse_document(
        file_path,
        document_id=str(resolved_document_id),
        enable_multimodal=enable_multimodal,
    )


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
        total_images = 0
        if enable_multimodal:
            total_images = _persist_images(
                db=db,
                document_id=resolved_document_id,
                parsed_document=parsed_document,
                captioner=captioner,
                embedder=embedder,
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
        total_images=total_images,
    )


def _resolve_document_id(document_id: str | uuid.UUID | None) -> uuid.UUID:
    if document_id is None:
        return uuid.uuid4()
    if isinstance(document_id, uuid.UUID):
        return document_id
    return uuid.UUID(document_id)


def _parse_document(
    file_path: Path,
    document_id: str | None = None,
    enable_multimodal: bool = False,
) -> tuple[ParsedDocument, str]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        if enable_multimodal:
            return MultimodalPdfParser().parse(file_path, document_id=document_id), "pdf"
        return parse_pdf(file_path), "pdf"
    if suffix == ".docx":
        return parse_docx(file_path), "docx"
    if suffix == ".txt":
        return parse_txt(file_path), "txt"
    if suffix == ".md":
        return parse_markdown(file_path), "md"
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


def _persist_images(
    *,
    db: Session,
    document_id: uuid.UUID,
    parsed_document: ParsedDocument,
    captioner: VisualCaptioner | None = None,
    embedder: BGEEmbedder | None = None,
) -> int:
    active_captioner = captioner or VisualCaptioner()
    active_embedder = embedder or _get_default_embedder()

    total_images = 0
    for page in parsed_document.pages:
        for image_dto in page.images:
            caption = image_dto.caption
            embedding = None

            # Attempt VLM captioning + embedding for each image individually
            if caption is None:
                try:
                    caption = active_captioner.caption_image(image_dto.image_path)
                except Exception:
                    logger.warning(
                        "VLM captioning failed for image %s on page %d; "
                        "persisting without caption.",
                        image_dto.image_id,
                        image_dto.page_number,
                        exc_info=True,
                    )

            if caption:
                try:
                    embedding = active_embedder.embed(caption)
                except Exception:
                    logger.warning(
                        "Embedding failed for caption of image %s; "
                        "persisting without embedding.",
                        image_dto.image_id,
                        exc_info=True,
                    )

            db.add(
                DocumentImageRecord(
                    id=uuid.UUID(image_dto.image_id),
                    document_id=document_id,
                    page_number=image_dto.page_number,
                    image_index=image_dto.image_index,
                    image_path=image_dto.image_path,
                    bbox=image_dto.bbox,
                    caption=caption,
                    embedding=embedding,
                )
            )
            total_images += 1
    return total_images

