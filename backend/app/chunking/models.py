from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text segment derived from a parsed document page."""

    chunk_id: str = Field(description="Unique identifier for the chunk")
    document_id: str = Field(description="Identifier of the source document")
    document_title: str = Field(description="Title of the source document")
    page_number: int = Field(description="1-based page index the chunk was derived from")
    chunk_index: int = Field(description="0-based position of the chunk within the document")
    text: str = Field(description="Chunk text content")
    char_count: int = Field(description="Number of characters in the chunk text")
