from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.embedder import BGEEmbedder
from app.models.document_image import DocumentImage
from app.retrieval.vector_search import BGE_QUERY_PREFIX

_default_embedder: BGEEmbedder | None = None


def _get_default_embedder() -> BGEEmbedder:
    """Return a process-wide embedder instance, loading the model on first use."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = BGEEmbedder()
    return _default_embedder


@dataclass(frozen=True)
class VisualSearchResult:
    """A document image matched by vector similarity search."""

    id: str
    document_id: str
    page_number: int
    image_index: int
    image_path: str
    caption: str
    similarity_score: float


@dataclass
class RetrievalCandidate:
    """Unified wrapper so text chunks and visual results share a common interface for the reranker.

    The existing ``CrossEncoderReranker`` accesses ``.id``, ``.document_id``,
    ``.page_number``, ``.chunk_index``, and ``.text`` via ``getattr`` — this
    dataclass satisfies that contract for both text and visual candidates.
    """

    id: object
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    is_visual: bool = False
    image_path: str | None = None

    @classmethod
    def from_chunk(cls, chunk: Any) -> "RetrievalCandidate":
        """Wrap a ``Chunk`` ORM object."""
        return cls(
            id=chunk.id,
            document_id=str(chunk.document_id),
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            is_visual=False,
            image_path=None,
        )

    @classmethod
    def from_visual_result(cls, result: "VisualSearchResult") -> "RetrievalCandidate":
        """Wrap a ``VisualSearchResult``."""
        return cls(
            id=result.id,
            document_id=result.document_id,
            page_number=result.page_number,
            chunk_index=result.image_index,
            text=result.caption,
            is_visual=True,
            image_path=result.image_path,
        )


class VisualSearchService:
    """Retrieve the most similar document images for a natural-language query."""

    def __init__(self, embedder: BGEEmbedder | None = None) -> None:
        self._embedder = embedder or _get_default_embedder()

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 10,
    ) -> list[VisualSearchResult]:
        """Embed the query and return the top-k images by cosine similarity."""
        query_embedding = self._embed_query(query)
        distance = DocumentImage.embedding.cosine_distance(query_embedding)
        similarity_score = (1 - distance).label("similarity_score")

        statement = (
            select(
                DocumentImage.id,
                DocumentImage.document_id,
                DocumentImage.page_number,
                DocumentImage.image_index,
                DocumentImage.image_path,
                DocumentImage.caption,
                similarity_score,
            )
            .where(DocumentImage.embedding.is_not(None))
            .order_by(distance)
            .limit(top_k)
        )

        rows = db.execute(statement).all()
        return [
            VisualSearchResult(
                id=str(row.id),
                document_id=str(row.document_id),
                page_number=row.page_number,
                image_index=row.image_index,
                image_path=row.image_path,
                caption=row.caption or "",
                similarity_score=float(row.similarity_score),
            )
            for row in rows
        ]

    def _embed_query(self, query: str) -> list[float]:
        """Generate a BGE query embedding with the model's retrieval prefix."""
        prefixed_query = f"{BGE_QUERY_PREFIX}{query}"
        return self._embedder.embed(prefixed_query)
