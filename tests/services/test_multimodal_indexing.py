import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database.base import Base
from app.generation.vlm_service import VLMGenerationError
from app.models.document import Document as DocumentRecord
from app.models.document_image import DocumentImage as DocumentImageRecord
from app.services.indexing_service import index_document

FAKE_CAPTION = "A 60x60 white square used as a sample test image."
FAKE_EMBEDDING = [0.1] * 384


@pytest.fixture
def sqlite_db_session():
    """Create an in-memory SQLite database session for unit testing."""
    engine = create_engine("sqlite:///:memory:")
    # Create tables (handling pgvector column by fallback in sqlite if needed)
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    yield session
    session.close()


@pytest.fixture
def temp_pdf_with_image(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample_multimodal.pdf"
    doc = fitz.open()

    page = doc.new_page()
    page.insert_text((50, 50), "Multimodal PDF header text")

    # Insert a sample 60x60 RGB image into the PDF page
    pix = fitz.Pixmap(fitz.csRGB, fitz.Rect(0, 0, 60, 60), False)
    pix.clear_with(255)
    page.insert_image(fitz.Rect(50, 100, 150, 200), pixmap=pix)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def mock_captioner() -> MagicMock:
    """A mock VisualCaptioner that returns a canned caption."""
    captioner = MagicMock()
    captioner.caption_image.return_value = FAKE_CAPTION
    return captioner


@pytest.fixture
def mock_embedder() -> MagicMock:
    """A mock BGEEmbedder that returns a canned 384-d embedding."""
    embedder = MagicMock()
    embedder.embed.return_value = FAKE_EMBEDDING
    return embedder


def test_document_image_model_instantiation():
    doc_id = uuid.uuid4()
    img_id = uuid.uuid4()
    record = DocumentImageRecord(
        id=img_id,
        document_id=doc_id,
        page_number=1,
        image_index=0,
        image_path="data/media/documents/test/page_1_img_0.png",
        bbox=[50.0, 100.0, 150.0, 200.0],
        caption=None,
        embedding=None,
    )
    assert record.id == img_id
    assert record.document_id == doc_id
    assert record.page_number == 1
    assert record.image_index == 0
    assert record.image_path == "data/media/documents/test/page_1_img_0.png"
    assert record.bbox == [50.0, 100.0, 150.0, 200.0]
    assert record.caption is None
    assert record.embedding is None


def test_index_document_multimodal_disabled(temp_pdf_with_image: Path, sqlite_db_session: Session):
    summary = index_document(
        temp_pdf_with_image,
        db=sqlite_db_session,
        enable_multimodal=False,
    )

    assert summary.status == "indexed"
    assert summary.total_pages == 1
    assert summary.total_images == 0

    # Ensure no DocumentImage records were created
    images = sqlite_db_session.query(DocumentImageRecord).all()
    assert len(images) == 0


def test_index_document_multimodal_enabled(
    temp_pdf_with_image: Path,
    sqlite_db_session: Session,
    mock_captioner: MagicMock,
    mock_embedder: MagicMock,
):
    doc_id = uuid.uuid4()
    summary = index_document(
        temp_pdf_with_image,
        db=sqlite_db_session,
        document_id=doc_id,
        enable_multimodal=True,
        captioner=mock_captioner,
        embedder=mock_embedder,
    )

    assert summary.status == "indexed"
    assert summary.total_pages == 1
    assert summary.total_images == 1

    # Verify DocumentRecord and DocumentImageRecord in DB
    doc_record = sqlite_db_session.query(DocumentRecord).filter_by(id=doc_id).first()
    assert doc_record is not None
    assert len(doc_record.images) == 1

    img_record = doc_record.images[0]
    assert img_record.document_id == doc_id
    assert img_record.page_number == 1
    assert img_record.image_index == 0
    assert Path(img_record.image_path).exists()
    assert img_record.caption == FAKE_CAPTION
    assert img_record.embedding == FAKE_EMBEDDING


# ---------------------------------------------------------------------------
# New integration tests for VLM captioning + embedding pipeline
# ---------------------------------------------------------------------------


def test_multimodal_disabled_no_vlm_calls(
    temp_pdf_with_image: Path,
    sqlite_db_session: Session,
    mock_captioner: MagicMock,
    mock_embedder: MagicMock,
):
    """When enable_multimodal=False, the captioner and embedder should never be called."""
    index_document(
        temp_pdf_with_image,
        db=sqlite_db_session,
        enable_multimodal=False,
        captioner=mock_captioner,
        embedder=mock_embedder,
    )

    mock_captioner.caption_image.assert_not_called()
    mock_embedder.embed.assert_not_called()


def test_caption_persisted_on_image_record(
    temp_pdf_with_image: Path,
    sqlite_db_session: Session,
    mock_captioner: MagicMock,
    mock_embedder: MagicMock,
):
    """The VLM-generated caption should be stored in DocumentImage.caption."""
    doc_id = uuid.uuid4()
    index_document(
        temp_pdf_with_image,
        db=sqlite_db_session,
        document_id=doc_id,
        enable_multimodal=True,
        captioner=mock_captioner,
        embedder=mock_embedder,
    )

    img_record = sqlite_db_session.query(DocumentImageRecord).filter_by(document_id=doc_id).first()
    assert img_record is not None
    assert img_record.caption == FAKE_CAPTION
    mock_captioner.caption_image.assert_called_once()


def test_embedding_generated_and_persisted(
    temp_pdf_with_image: Path,
    sqlite_db_session: Session,
    mock_captioner: MagicMock,
    mock_embedder: MagicMock,
):
    """The embedding for the caption should be stored in DocumentImage.embedding."""
    doc_id = uuid.uuid4()
    index_document(
        temp_pdf_with_image,
        db=sqlite_db_session,
        document_id=doc_id,
        enable_multimodal=True,
        captioner=mock_captioner,
        embedder=mock_embedder,
    )

    img_record = sqlite_db_session.query(DocumentImageRecord).filter_by(document_id=doc_id).first()
    assert img_record is not None
    assert img_record.embedding == FAKE_EMBEDDING
    mock_embedder.embed.assert_called_once_with(FAKE_CAPTION)


def test_vlm_failure_does_not_break_text_indexing(
    temp_pdf_with_image: Path,
    sqlite_db_session: Session,
    mock_embedder: MagicMock,
):
    """If VLM captioning fails, the document and text chunks should still be indexed."""
    failing_captioner = MagicMock()
    failing_captioner.caption_image.side_effect = VLMGenerationError("service unavailable")

    doc_id = uuid.uuid4()
    summary = index_document(
        temp_pdf_with_image,
        db=sqlite_db_session,
        document_id=doc_id,
        enable_multimodal=True,
        captioner=failing_captioner,
        embedder=mock_embedder,
    )

    # Document is still indexed successfully
    assert summary.status == "indexed"
    assert summary.total_pages == 1
    assert summary.total_chunks > 0

    # Image record is persisted but without caption/embedding
    img_record = sqlite_db_session.query(DocumentImageRecord).filter_by(document_id=doc_id).first()
    assert img_record is not None
    assert img_record.caption is None
    assert img_record.embedding is None

    # Embedder should NOT be called since there is no caption to embed
    mock_embedder.embed.assert_not_called()


def test_embedding_failure_preserves_caption(
    temp_pdf_with_image: Path,
    sqlite_db_session: Session,
    mock_captioner: MagicMock,
):
    """If embedding fails, the caption should still be persisted on the image record."""
    failing_embedder = MagicMock()
    failing_embedder.embed.side_effect = RuntimeError("model crashed")

    doc_id = uuid.uuid4()
    summary = index_document(
        temp_pdf_with_image,
        db=sqlite_db_session,
        document_id=doc_id,
        enable_multimodal=True,
        captioner=mock_captioner,
        embedder=failing_embedder,
    )

    assert summary.status == "indexed"
    assert summary.total_images == 1

    img_record = sqlite_db_session.query(DocumentImageRecord).filter_by(document_id=doc_id).first()
    assert img_record is not None
    assert img_record.caption == FAKE_CAPTION
    assert img_record.embedding is None


# ---------------------------------------------------------------------------
# Auto-detection: PDF → multimodal, other formats → text-only
# ---------------------------------------------------------------------------


def test_pdf_auto_enables_multimodal(
    temp_pdf_with_image: Path,
    sqlite_db_session: Session,
    mock_captioner: MagicMock,
    mock_embedder: MagicMock,
):
    """PDF uploads should automatically trigger multimodal processing without explicit flag."""
    doc_id = uuid.uuid4()
    summary = index_document(
        temp_pdf_with_image,
        db=sqlite_db_session,
        document_id=doc_id,
        # NOTE: enable_multimodal is NOT passed — auto-detection kicks in
        captioner=mock_captioner,
        embedder=mock_embedder,
    )

    assert summary.status == "indexed"
    assert summary.total_images == 1

    img_record = sqlite_db_session.query(DocumentImageRecord).filter_by(document_id=doc_id).first()
    assert img_record is not None
    assert img_record.caption == FAKE_CAPTION
    assert img_record.embedding == FAKE_EMBEDDING
    mock_captioner.caption_image.assert_called_once()


def test_txt_does_not_trigger_multimodal(
    tmp_path: Path,
    sqlite_db_session: Session,
    mock_captioner: MagicMock,
    mock_embedder: MagicMock,
):
    """TXT uploads should remain text-only — no multimodal processing."""
    txt_file = tmp_path / "plain.txt"
    txt_file.write_text("Just plain text content.", encoding="utf-8")

    summary = index_document(
        txt_file,
        db=sqlite_db_session,
        # NOTE: enable_multimodal is NOT passed
        captioner=mock_captioner,
        embedder=mock_embedder,
    )

    assert summary.status == "indexed"
    assert summary.total_images == 0
    mock_captioner.caption_image.assert_not_called()
    mock_embedder.embed.assert_not_called()


def test_docx_does_not_trigger_multimodal(
    tmp_path: Path,
    sqlite_db_session: Session,
    mock_captioner: MagicMock,
    mock_embedder: MagicMock,
):
    """DOCX uploads should remain text-only — no multimodal processing."""
    from docx import Document as DocxDocument

    docx_file = tmp_path / "sample.docx"
    doc = DocxDocument()
    doc.add_paragraph("Word document content.")
    doc.save(str(docx_file))

    summary = index_document(
        docx_file,
        db=sqlite_db_session,
        # NOTE: enable_multimodal is NOT passed
        captioner=mock_captioner,
        embedder=mock_embedder,
    )

    assert summary.status == "indexed"
    assert summary.total_images == 0
    mock_captioner.caption_image.assert_not_called()
    mock_embedder.embed.assert_not_called()

