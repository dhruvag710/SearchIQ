from pathlib import Path

from app.ingestion.base import DocumentParser
from app.ingestion.models import Document


class MarkdownParser(DocumentParser):
    """Parser for Markdown (.md) documents."""

    def parse(self, path: Path) -> Document:
        raise NotImplementedError("Markdown parsing is not yet implemented.")
