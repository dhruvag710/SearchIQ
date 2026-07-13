from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    question: str = Field(description="User question to answer from indexed documents")


class ChatSourceResponse(BaseModel):
    """Source reference included in a chat response."""

    document_id: str
    page_number: int
    chunk_index: int


class ChatResponse(BaseModel):
    """Response returned after a chat question is answered."""

    answer: str
    sources: list[ChatSourceResponse]
