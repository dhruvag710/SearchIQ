from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ChatSourceResponse
from app.retrieval.visual_search import VisualSearchService
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Answer a question using indexed document context."""
    service = ChatService(visual_search=VisualSearchService())
    result = service.chat(db, request.question)

    return ChatResponse(
        answer=result.answer,
        sources=[
            ChatSourceResponse(
                document_id=source.document_id,
                page_number=source.page_number,
                chunk_index=source.chunk_index,
                is_visual=source.is_visual,
                image_path=source.image_path,
            )
            for source in result.sources
        ],
    )
