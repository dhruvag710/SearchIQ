from pathlib import Path

import fitz

from app.ingestion.cleaner import clean_text
from app.ingestion.models import Document, Page


def _resolve_document_title(doc: fitz.Document, pdf_path: Path) -> str:
    """Return PDF metadata title when present, otherwise the file stem."""
    metadata = doc.metadata or {}
    title = (metadata.get("title") or "").strip()
    if title:
        return title
    return pdf_path.stem


def parse_pdf(pdf_path: Path) -> Document:
    """Read a PDF page by page and return structured text for every page."""
    with fitz.open(pdf_path) as doc:
        document_title = _resolve_document_title(doc, pdf_path)
        total_pages = doc.page_count
        pages: list[Page] = []

        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            raw_text = page.get_text()
            pages.append(
                Page(
                    page_number=page_index + 1,
                    text=clean_text(raw_text),
                )
            )

        return Document(
            document_title=document_title,
            total_pages=total_pages,
            pages=pages,
        )
