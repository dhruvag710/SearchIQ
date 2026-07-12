from pathlib import Path

from app.ingestion.base import DocumentParser
from app.ingestion.models import Document


class TxtParser(DocumentParser):
    """Parser for plain text (.txt) documents."""

    def parse(self, path: Path) -> Document:
        raise NotImplementedError("TXT parsing is not yet implemented.")
