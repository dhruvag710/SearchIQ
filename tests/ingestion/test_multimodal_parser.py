import sys
from pathlib import Path

import fitz
import pytest

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ingestion.models import ExtractedImage, Page
from app.ingestion.multimodal_parser import MultimodalPdfParser, parse_multimodal_pdf


@pytest.fixture
def sample_pdf_with_image(tmp_path: Path) -> Path:
    """Create a temporary PDF file containing text and a sample image."""
    pdf_path = tmp_path / "multimodal_sample.pdf"
    doc = fitz.open()

    # Create page 1 with text and an image
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Page 1 sample header text.")

    # Create a simple 100x100 RGB image pixmap and insert into page 1
    pix = fitz.Pixmap(fitz.csRGB, fitz.Rect(0, 0, 100, 100), False)
    pix.clear_with(255)
    page1.insert_image(fitz.Rect(100, 100, 200, 200), pixmap=pix)

    # Create page 2 with text only
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page 2 text only content.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_multimodal_pdf_parser_extraction(sample_pdf_with_image: Path, tmp_path: Path):
    media_dir = tmp_path / "media"
    parser = MultimodalPdfParser(output_media_dir=media_dir, min_image_dimension=20)

    document = parser.parse(sample_pdf_with_image, document_id="doc_123")

    assert document.document_title == "multimodal_sample"
    assert document.total_pages == 2
    assert len(document.pages) == 2

    # Check page 1 text & visual extraction
    page1 = document.pages[0]
    assert page1.page_number == 1
    assert "Page 1 sample header text." in page1.text
    assert len(page1.images) == 1

    img: ExtractedImage = page1.images[0]
    assert img.page_number == 1
    assert img.image_index == 0
    assert Path(img.image_path).exists()
    assert img.image_path.endswith("page_1_img_0.png")
    assert img.bbox is not None
    assert len(img.bbox) == 4

    # Check page 2 (text only)
    page2 = document.pages[1]
    assert page2.page_number == 2
    assert "Page 2 text only content." in page2.text
    assert len(page2.images) == 0


def test_parse_multimodal_pdf_helper(sample_pdf_with_image: Path):
    document = parse_multimodal_pdf(sample_pdf_with_image, document_id="test_doc_456")
    assert document.total_pages == 2
    assert len(document.pages[0].images) == 1
