from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from app.models.chunk import Chunk


def _tokenize(text: str) -> list[str]:
    """Tokenize text using simple whitespace splitting after lowercasing."""
    if not text:
        return []
    return text.lower().split()


class BM25Retriever:
    """Lexical retriever using BM25Okapi over stored document chunks."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build_index(self, db: Session) -> None:
        """Fetch all stored document chunks and construct the BM25 index."""
        chunks = db.query(Chunk).all()
        self._chunks = chunks

        if not chunks:
            self._bm25 = None
            return

        corpus = [_tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[Chunk]:
        """Retrieve top-k Chunk objects matching the query sorted by relevance."""
        if self._bm25 is None or not self._chunks or top_k <= 0:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        scored_chunks = list(zip(scores, self._chunks))
        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        return [chunk for _, chunk in scored_chunks[:top_k]]
