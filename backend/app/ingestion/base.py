from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.models import Document


class DocumentParser(ABC):
    """Base interface for document format parsers."""

    @abstractmethod
    def parse(self, path: Path) -> Document:
        """Extract structured text content from a document file."""
