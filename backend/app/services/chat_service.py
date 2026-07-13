from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.generation.llm_service import LLMService
from app.prompting.prompt_builder import PromptBuilder
from app.retrieval.hybrid_search import HybridSearchService
from app.retrieval.reranker import CrossEncoderReranker, RerankedResult


@dataclass(frozen=True)
class ChatSource:
    """A document passage used to generate the chat answer."""

    document_id: str
    page_number: int
    chunk_index: int


@dataclass(frozen=True)
class ChatResponse:
    """Structured response returned by the chat orchestration service."""

    answer: str
    sources: list[ChatSource]


class ChatService:
    """Orchestrate retrieval, reranking, prompt construction, and answer generation."""

    def __init__(
        self,
        hybrid_search: HybridSearchService | None = None,
        reranker: CrossEncoderReranker | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self._hybrid_search = hybrid_search or HybridSearchService()
        self._reranker = reranker or CrossEncoderReranker()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._llm_service = llm_service or LLMService()

    def chat(
        self,
        db: Session,
        question: str,
        top_k: int = 5,
    ) -> ChatResponse:
        """Run the full RAG pipeline and return an answer with source references."""
        hybrid_results = self._hybrid_search.search(db, question, top_k=top_k)
        reranked_results = self._reranker.rerank(question, hybrid_results, top_k=top_k)
        prompt = self._prompt_builder.build(question, reranked_results)
        answer = self._llm_service.generate(prompt)

        return ChatResponse(
            answer=answer,
            sources=_build_sources(reranked_results),
        )


def _build_sources(results: list[RerankedResult]) -> list[ChatSource]:
    """Map reranked retrieval results to response source references."""
    return [
        ChatSource(
            document_id=result.document_id,
            page_number=result.page_number,
            chunk_index=result.chunk_index,
        )
        for result in results
    ]
