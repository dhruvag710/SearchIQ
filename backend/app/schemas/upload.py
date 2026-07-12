from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned after a successful PDF upload."""

    document_id: str = Field(description="Unique identifier assigned to the uploaded document")
    original_filename: str = Field(description="Original filename provided by the client")
    stored_filename: str = Field(description="UUID-based filename used on disk")
    file_size_bytes: int = Field(description="Size of the saved file in bytes")
    uploaded_at: str = Field(description="Upload timestamp in ISO 8601 format")
    status: str = Field(description="Current processing status of the document")
