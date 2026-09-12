from pydantic import BaseModel, Field


class ExtractedImage(BaseModel):
    """Visual asset extracted from a PDF page."""

    image_id: str = Field(description="Unique identifier for the extracted image")
    page_number: int = Field(description="1-based page index")
    image_index: int = Field(description="0-based index of the image on the page")
    image_path: str = Field(description="Path to the saved image file on disk")
    bbox: list[float] | None = Field(default=None, description="Bounding box [x0, y0, x1, y1] on page if available")
    caption: str | None = Field(default=None, description="Optional description/caption of the image")


class Page(BaseModel):
    """Text content extracted from a single PDF page, with optional visual assets."""

    page_number: int = Field(description="1-based page index")
    text: str = Field(description="Extracted text content for the page")
    images: list[ExtractedImage] = Field(default_factory=list, description="Extracted visual assets for the page")


class Document(BaseModel):
    """Structured representation of a parsed PDF document."""

    document_title: str = Field(description="Title of the document")
    total_pages: int = Field(description="Total number of pages in the document")
    pages: list[Page] = Field(description="Extracted content for each page")

