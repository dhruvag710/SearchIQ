from pathlib import Path

from app.ingestion.base import DocumentParser
from app.ingestion.models import Document


class DocxParser(DocumentParser):
    """Parser for Microsoft Word (.docx) documents."""

    def parse(self, path: Path) -> Document:
        raise NotImplementedError("DOCX parsing is not yet implemented.")
