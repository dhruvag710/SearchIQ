from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ChatSourceResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Answer a question using indexed document context."""
    result = ChatService().chat(db, request.question)

    return ChatResponse(
        answer=result.answer,
        sources=[
            ChatSourceResponse(
                document_id=source.document_id,
                page_number=source.page_number,
                chunk_index=source.chunk_index,
            )
            for source in result.sources
        ],
    )
