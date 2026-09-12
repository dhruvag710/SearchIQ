import uuid
from pathlib import Path

import fitz

from app.ingestion.base import DocumentParser
from app.ingestion.models import Document, ExtractedImage
from app.ingestion.parser import PdfParser

MIN_IMAGE_DIMENSION = 50  # Filter out tiny icon / bullet images smaller than 50x50


class MultimodalPdfParser(DocumentParser):
    """Extract structured text and visual assets (images/charts) from PDF documents."""

    def __init__(
        self,
        base_parser: PdfParser | None = None,
        output_media_dir: Path | str = "data/media/documents",
        min_image_dimension: int = MIN_IMAGE_DIMENSION,
    ) -> None:
        self._base_parser = base_parser or PdfParser()
        self._output_media_dir = Path(output_media_dir)
        self._min_image_dimension = min_image_dimension

    def parse(self, path: Path, *, document_id: str | None = None) -> Document:
        """Parse text using PdfParser, then extract and store page images."""
        doc_model = self._base_parser.parse(path)

        resolved_doc_id = document_id or str(uuid.uuid4())
        doc_media_dir = self._output_media_dir / resolved_doc_id

        with fitz.open(path) as pdf_doc:
            for page_idx, page_model in enumerate(doc_model.pages):
                page = pdf_doc.load_page(page_idx)
                extracted_images = self._extract_page_images(
                    pdf_doc=pdf_doc,
                    page=page,
                    page_number=page_model.page_number,
                    doc_media_dir=doc_media_dir,
                )
                page_model.images = extracted_images

        return doc_model

    def _extract_page_images(
        self,
        pdf_doc: fitz.Document,
        page: fitz.Page,
        page_number: int,
        doc_media_dir: Path,
    ) -> list[ExtractedImage]:
        image_list = page.get_images(full=True)
        if not image_list:
            return []

        extracted_images: list[ExtractedImage] = []
        image_index = 0

        for img_info in image_list:
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(pdf_doc, xref)
            except Exception:
                continue

            if pix.width < self._min_image_dimension or pix.height < self._min_image_dimension:
                pix = None
                continue

            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            doc_media_dir.mkdir(parents=True, exist_ok=True)
            image_filename = f"page_{page_number}_img_{image_index}.png"
            image_path = doc_media_dir / image_filename

            pix.save(str(image_path))
            pix = None

            bbox: list[float] | None = None
            try:
                rects = page.get_image_rects(xref)
                if rects:
                    rect = rects[0]
                    bbox = [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]
            except Exception:
                bbox = None

            extracted_images.append(
                ExtractedImage(
                    image_id=str(uuid.uuid4()),
                    page_number=page_number,
                    image_index=image_index,
                    image_path=str(image_path).replace("\\", "/"),
                    bbox=bbox,
                )
            )
            image_index += 1

        return extracted_images


def parse_multimodal_pdf(pdf_path: Path, document_id: str | None = None) -> Document:
    """Read a PDF page by page and return structured text and extracted images."""
    return MultimodalPdfParser().parse(pdf_path, document_id=document_id)
