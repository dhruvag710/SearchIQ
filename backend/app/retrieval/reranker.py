from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.retrieval.hybrid_search import HybridSearchResult

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_default_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    """Return a process-wide cross-encoder instance, loading the model on first use."""
    global _default_cross_encoder
    if _default_cross_encoder is None:
        _default_cross_encoder = CrossEncoder(MODEL_NAME)
    return _default_cross_encoder


@dataclass(frozen=True)
class RerankedResult:
    """A chunk reordered by cross-encoder relevance scoring."""

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    hybrid_score: float
    reranker_score: float


class CrossEncoderReranker:
    """Rerank hybrid retrieval candidates with a cross-encoder relevance model."""

    def __init__(self, cross_encoder: CrossEncoder | None = None) -> None:
        self._cross_encoder = cross_encoder

    def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """Score query-chunk pairs and return the top-k results by reranker score."""
        if not results:
            return []

        pairs = [(query, result.text) for result in results]
        scores = self._get_model().predict(pairs, show_progress_bar=False)

        reranked_results = [
            RerankedResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                text=result.text,
                hybrid_score=result.final_score,
                reranker_score=float(score),
            )
            for result, score in zip(results, scores, strict=True)
        ]

        reranked_results.sort(key=lambda result: result.reranker_score, reverse=True)
        return reranked_results[:top_k]

    def _get_model(self) -> CrossEncoder:
        """Return the injected cross-encoder or the shared lazy-loaded instance."""
        if self._cross_encoder is not None:
            return self._cross_encoder
        return _get_cross_encoder()
