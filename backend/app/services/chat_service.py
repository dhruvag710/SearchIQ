from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.generation.llm_service import LLMService
from app.prompting.prompt_builder import PromptBuilder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.hybrid_search import HybridSearchResult
from app.retrieval.reranker import CrossEncoderReranker, RerankedResult
from app.retrieval.visual_search import (
    RetrievalCandidate,
    VisualSearchService,
)


@dataclass(frozen=True)
class ChatSource:
    """A document passage or visual asset used to generate the chat answer."""

    document_id: str
    page_number: int
    chunk_index: int
    is_visual: bool = False
    image_path: str | None = None


@dataclass(frozen=True)
class ChatResponse:
    """Structured response returned by the chat orchestration service."""

    answer: str
    sources: list[ChatSource]


class ChatService:
    """Orchestrate hybrid retrieval, reranking, prompt construction, and answer generation."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_service: LLMService | None = None,
        visual_search: VisualSearchService | None = None,
        *,
        hybrid_search: Any | None = None,
    ) -> None:
        self._hybrid_retriever = hybrid_retriever or hybrid_search or HybridRetriever()
        self._reranker = reranker or CrossEncoderReranker()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._llm_service = llm_service or LLMService()
        self._visual_search = visual_search

    def chat(
        self,
        db: Session,
        question: str,
        top_k: int = 5,
        candidate_k: int = 25,
        visual_k: int = 10,
    ) -> ChatResponse:
        """Run the full RAG pipeline and return an answer with source references."""
        retrieve_k = max(top_k, candidate_k)
        chunks = self._hybrid_retriever.retrieve(db, question, top_k=retrieve_k)

        if self._visual_search is not None:
            visual_results = self._visual_search.search(db, question, top_k=visual_k)

            # Wrap both text and visual results in a unified candidate
            candidates: list[Any] = [RetrievalCandidate.from_chunk(c) for c in chunks]
            candidates += [RetrievalCandidate.from_visual_result(v) for v in visual_results]

            # Build lookup for recovering visual metadata after reranking
            candidate_map: dict[str, RetrievalCandidate] = {
                str(c.id): c for c in candidates
            }
            visual_ids: set[str] = {str(c.id) for c in candidates if c.is_visual}
        else:
            # Text-only: pass raw chunks to the reranker for full backward compat
            candidates = chunks
            candidate_map = {}
            visual_ids = set()

        reranked_results = self._reranker.rerank(question, candidates, top_k=top_k)
        prompt = self._prompt_builder.build(
            question, reranked_results, visual_ids=visual_ids,
        )
        answer = self._llm_service.generate(prompt)

        return ChatResponse(
            answer=answer,
            sources=_build_sources(reranked_results, candidate_map),
        )


def _build_sources(
    results: list[RerankedResult],
    candidate_map: dict[str, RetrievalCandidate] | None = None,
) -> list[ChatSource]:
    """Map reranked retrieval results to response source references."""
    sources: list[ChatSource] = []
    for result in results:
        candidate = candidate_map.get(result.chunk_id) if candidate_map else None
        is_visual = candidate.is_visual if candidate else False
        image_path = candidate.image_path if candidate and candidate.is_visual else None
        sources.append(
            ChatSource(
                document_id=result.document_id,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                is_visual=is_visual,
                image_path=image_path,
            )
        )
    return sources

