from pathlib import Path

from app.ingestion.base import DocumentParser
from app.ingestion.cleaner import clean_text
from app.ingestion.models import Document, Page


class MarkdownParser(DocumentParser):
    """Extract structured text from Markdown (.md) documents."""

    def parse(self, path: Path) -> Document:
        """Read a Markdown file and return structured text for the document."""
        raw_text = path.read_text(encoding="utf-8")
        return Document(
            document_title=path.stem,
            total_pages=1,
            pages=[
                Page(
                    page_number=1,
                    text=clean_text(raw_text),
                )
            ],
        )


def parse_markdown(path: Path) -> Document:
    """Read a UTF-8 encoded Markdown file and return structured text for the document."""
    return MarkdownParser().parse(path)
