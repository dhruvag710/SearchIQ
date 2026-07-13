from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk

FTS_CONFIG = "english"


@dataclass(frozen=True)
class KeywordSearchResult:
    """A chunk matched by full-text keyword search."""

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    relevance_score: float


class KeywordSearchService:
    """Retrieve the most relevant stored chunks for a keyword query."""

    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 5,
    ) -> list[KeywordSearchResult]:
        """Run PostgreSQL full-text search and return the top-k ranked chunks."""
        tsvector = func.to_tsvector(FTS_CONFIG, Chunk.text)
        tsquery = func.websearch_to_tsquery(FTS_CONFIG, query)
        relevance_score = func.ts_rank(tsvector, tsquery).label("relevance_score")

        statement = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.page_number,
                Chunk.chunk_index,
                Chunk.text,
                relevance_score,
            )
            .where(tsvector.bool_op("@@")(tsquery))
            .order_by(relevance_score.desc())
            .limit(top_k)
        )

        rows = db.execute(statement).all()
        return [
            KeywordSearchResult(
                chunk_id=str(row.id),
                document_id=str(row.document_id),
                page_number=row.page_number,
                chunk_index=row.chunk_index,
                text=row.text,
                relevance_score=float(row.relevance_score),
            )
            for row in rows
        ]
