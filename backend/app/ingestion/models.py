from pydantic import BaseModel, Field


class Page(BaseModel):
    """Text content extracted from a single PDF page."""

    page_number: int = Field(description="1-based page index")
    text: str = Field(description="Extracted text content for the page")


class Document(BaseModel):
    """Structured representation of a parsed PDF document."""

    document_title: str = Field(description="Title of the document")
    total_pages: int = Field(description="Total number of pages in the document")
    pages: list[Page] = Field(description="Extracted content for each page")
