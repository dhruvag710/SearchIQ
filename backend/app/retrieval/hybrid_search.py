from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.retrieval.keyword_search import KeywordSearchService
from app.retrieval.vector_search import VectorSearchService


@dataclass(frozen=True)
class HybridSearchResult:
    """A chunk ranked by combined vector and keyword retrieval scores."""

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    vector_score: float
    keyword_score: float
    final_score: float


class HybridSearchService:
    """Combine vector similarity and keyword relevance into a single ranked result set."""

    def __init__(
        self,
        vector_search: VectorSearchService | None = None,
        keyword_search: KeywordSearchService | None = None,
    ) -> None:
        self._vector_search = vector_search or VectorSearchService()
        self._keyword_search = keyword_search or KeywordSearchService()

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 5,
    ) -> list[HybridSearchResult]:
        """Retrieve candidates from both services, merge by chunk_id, and rank by final score."""
        vector_results = self._vector_search.search(db, query, top_k=top_k)
        keyword_results = self._keyword_search.search(db, query, top_k=top_k)

        normalized_vector_scores = _min_max_normalize(
            {result.chunk_id: result.similarity_score for result in vector_results}
        )
        normalized_keyword_scores = _min_max_normalize(
            {result.chunk_id: result.relevance_score for result in keyword_results}
        )

        merged: dict[str, HybridSearchResult] = {}

        for result in vector_results:
            merged[result.chunk_id] = HybridSearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                text=result.text,
                vector_score=normalized_vector_scores[result.chunk_id],
                keyword_score=0.0,
                final_score=0.0,
            )

        for result in keyword_results:
            keyword_score = normalized_keyword_scores[result.chunk_id]
            existing = merged.get(result.chunk_id)
            if existing is not None:
                merged[result.chunk_id] = HybridSearchResult(
                    chunk_id=existing.chunk_id,
                    document_id=existing.document_id,
                    page_number=existing.page_number,
                    chunk_index=existing.chunk_index,
                    text=existing.text,
                    vector_score=existing.vector_score,
                    keyword_score=keyword_score,
                    final_score=0.0,
                )
            else:
                merged[result.chunk_id] = HybridSearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    page_number=result.page_number,
                    chunk_index=result.chunk_index,
                    text=result.text,
                    vector_score=0.0,
                    keyword_score=keyword_score,
                    final_score=0.0,
                )

        ranked_results = [
            HybridSearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                text=result.text,
                vector_score=result.vector_score,
                keyword_score=result.keyword_score,
                final_score=(result.vector_score + result.keyword_score) / 2,
            )
            for result in merged.values()
        ]

        ranked_results.sort(key=lambda result: result.final_score, reverse=True)
        return ranked_results[:top_k]


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to [0, 1] independently within a single retrieval result set."""
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())
    if min_score == max_score:
        return {chunk_id: 1.0 for chunk_id in scores}

    score_range = max_score - min_score
    return {
        chunk_id: (score - min_score) / score_range
        for chunk_id, score in scores.items()
    }
