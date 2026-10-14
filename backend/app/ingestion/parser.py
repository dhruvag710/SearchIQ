from pathlib import Path

import fitz

from app.ingestion.base import DocumentParser
from app.ingestion.cleaner import clean_text
from app.ingestion.models import Document, Page


def _resolve_document_title(doc: fitz.Document, pdf_path: Path) -> str:
    """Return PDF metadata title when present, otherwise the file stem."""
    metadata = doc.metadata or {}
    title = (metadata.get("title") or "").strip()
    if title:
        return title
    return pdf_path.stem


def _extract_page_content(page: fitz.Page) -> str:
    """Extract page text with tables formatted as Markdown in natural reading order.

    If PyMuPDF table detection is supported and finds tables on the page, non-table
    text blocks outside table bounding boxes are combined with Markdown representations
    of the tables in top-to-bottom reading order. If no tables are detected or table
    extraction fails, standard page text extraction is used as a fallback.
    """
    if not hasattr(page, "find_tables"):
        return page.get_text()

    try:
        tabs = page.find_tables()
    except Exception:
        return page.get_text()

    if not tabs or not getattr(tabs, "tables", None):
        return page.get_text()

    tab_rects = [fitz.Rect(tab.bbox) for tab in tabs.tables]

    elements: list[tuple[float, float, str]] = []
    for tab in tabs.tables:
        try:
            markdown = tab.to_markdown()
            if markdown and markdown.strip():
                elements.append((tab.bbox[1], tab.bbox[0], markdown.strip()))
        except Exception:
            pass

    # Extract non-table prose at line level to avoid discarding surrounding prose
    # that happens to share or intersect a block with a table
    try:
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])
        for block in blocks:
            if block.get("type") != 0:
                continue

            current_prose_lines: list[str] = []
            first_line_pos: tuple[float, float] | None = None

            for line in block.get("lines", []):
                line_rect = fitz.Rect(line.get("bbox", (0, 0, 0, 0)))
                line_center = fitz.Point(
                    (line_rect.x0 + line_rect.x1) / 2,
                    (line_rect.y0 + line_rect.y1) / 2,
                )
                inside_table = any(line_center in t_rect for t_rect in tab_rects)
                line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()

                if not line_text:
                    continue

                if inside_table:
                    if current_prose_lines and first_line_pos:
                        elements.append((first_line_pos[0], first_line_pos[1], "\n".join(current_prose_lines)))
                        current_prose_lines = []
                        first_line_pos = None
                else:
                    if not current_prose_lines:
                        first_line_pos = (line_rect.y0, line_rect.x0)
                    current_prose_lines.append(line_text)

            if current_prose_lines and first_line_pos:
                elements.append((first_line_pos[0], first_line_pos[1], "\n".join(current_prose_lines)))
    except Exception:
        return page.get_text()

    if not elements:
        return page.get_text()

    elements.sort(key=lambda item: (item[0], item[1]))
    return "\n\n".join(item[2] for item in elements)


class PdfParser(DocumentParser):
    """Extract structured text from PDF documents."""

    def parse(self, path: Path) -> Document:
        """Read a PDF page by page and return structured text for every page."""
        with fitz.open(path) as doc:
            document_title = _resolve_document_title(doc, path)
            total_pages = doc.page_count
            pages: list[Page] = []

            for page_index in range(total_pages):
                page = doc.load_page(page_index)
                raw_text = _extract_page_content(page)
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


def parse_pdf(pdf_path: Path) -> Document:
    """Read a PDF page by page and return structured text for every page."""
    return PdfParser().parse(pdf_path)
