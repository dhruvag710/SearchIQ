from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.embedder import BGEEmbedder
from app.models.chunk import Chunk

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_default_embedder: BGEEmbedder | None = None


def _get_default_embedder() -> BGEEmbedder:
    """Return a process-wide embedder instance, loading the model on first use."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = BGEEmbedder()
    return _default_embedder


@dataclass(frozen=True)
class VectorSearchResult:
    """A chunk matched by vector similarity search."""

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    similarity_score: float


class VectorSearchService:
    """Retrieve the most similar stored chunks for a natural-language query."""

    def __init__(self, embedder: BGEEmbedder | None = None) -> None:
        self._embedder = embedder or _get_default_embedder()

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Embed the query and return the top-k chunks by cosine similarity."""
        query_embedding = self._embed_query(query)
        distance = Chunk.embedding.cosine_distance(query_embedding)
        similarity_score = (1 - distance).label("similarity_score")

        statement = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.page_number,
                Chunk.chunk_index,
                Chunk.text,
                similarity_score,
            )
            .where(Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(top_k)
        )

        rows = db.execute(statement).all()
        return [
            VectorSearchResult(
                chunk_id=str(row.id),
                document_id=str(row.document_id),
                page_number=row.page_number,
                chunk_index=row.chunk_index,
                text=row.text,
                similarity_score=float(row.similarity_score),
            )
            for row in rows
        ]

    def _embed_query(self, query: str) -> list[float]:
        """Generate a BGE query embedding with the model's retrieval prefix."""
        prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
        return self._embedder.embed(prefixed_query)
