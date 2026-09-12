from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.vector_search import VectorSearchService


class HybridRetriever:
    """Coordinate dense vector search and BM25 lexical search merged via RRF."""

    def __init__(
        self,
        dense_retriever: VectorSearchService | Any | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever or VectorSearchService()
        self.bm25_retriever = bm25_retriever

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int = 20,
    ) -> list[Chunk]:
        """Retrieve top-k chunks by combining dense and lexical search with RRF."""
        if not query or not query.strip() or top_k <= 0:
            return []

        # 1. Dense retrieval using existing VectorSearchService
        dense_chunks: list[Chunk] = []
        if hasattr(self.dense_retriever, "retrieve"):
            dense_chunks = self.dense_retriever.retrieve(db, query, top_k=top_k)
        elif hasattr(self.dense_retriever, "search"):
            vector_results = self.dense_retriever.search(db, query, top_k=top_k)
            chunk_ids = [res.chunk_id for res in vector_results]
            if chunk_ids:
                fetched_chunks = db.scalars(
                    select(Chunk).where(Chunk.id.in_(chunk_ids))
                ).all()
                chunk_map = {str(c.id): c for c in fetched_chunks}
                dense_chunks = [
                    chunk_map[cid] for cid in chunk_ids if cid in chunk_map
                ]

        # 2. Build/use BM25Retriever
        bm25_retriever = self.bm25_retriever
        if bm25_retriever is None:
            bm25_retriever = BM25Retriever()
            bm25_retriever.build_index(db)

        # 3. Retrieve BM25 candidates
        bm25_chunks = bm25_retriever.retrieve(query, top_k=top_k)

        # 4. Fuse both lists using reciprocal_rank_fusion()
        fused_results = reciprocal_rank_fusion(dense_chunks, bm25_chunks)

        # 5. Return the fused ranking
        return fused_results[:top_k]
