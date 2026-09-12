import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy.orm import Session

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.chunk import Chunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever


def create_chunk(text: str = "sample text") -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        text=text,
    )


def test_both_retrievers_invoked_and_fused():
    c1 = create_chunk("Chunk 1")
    c2 = create_chunk("Chunk 2")
    c3 = create_chunk("Chunk 3")

    mock_db = MagicMock(spec=Session)

    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = [c1, c2]

    mock_bm25 = MagicMock(spec=BM25Retriever)
    mock_bm25.retrieve.return_value = [c2, c3]

    retriever = HybridRetriever(dense_retriever=mock_dense, bm25_retriever=mock_bm25)
    results = retriever.retrieve(mock_db, "python query", top_k=20)

    # Verify both retrievers were called
    mock_dense.retrieve.assert_called_once_with(mock_db, "python query", top_k=20)
    mock_bm25.retrieve.assert_called_once_with("python query", top_k=20)

    # c2 is in both dense and BM25, so it should rank highest via RRF
    assert results[0] == c2
    assert set(results) == {c1, c2, c3}


def test_empty_retrieval():
    mock_db = MagicMock(spec=Session)

    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = []

    mock_bm25 = MagicMock(spec=BM25Retriever)
    mock_bm25.retrieve.return_value = []

    retriever = HybridRetriever(dense_retriever=mock_dense, bm25_retriever=mock_bm25)
    results = retriever.retrieve(mock_db, "empty query", top_k=20)

    assert results == []


def test_top_k_respected():
    chunks_dense = [create_chunk(f"Dense {i}") for i in range(5)]
    chunks_bm25 = [create_chunk(f"BM25 {i}") for i in range(5)]

    mock_db = MagicMock(spec=Session)

    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = chunks_dense

    mock_bm25 = MagicMock(spec=BM25Retriever)
    mock_bm25.retrieve.return_value = chunks_bm25

    retriever = HybridRetriever(dense_retriever=mock_dense, bm25_retriever=mock_bm25)
    results = retriever.retrieve(mock_db, "query", top_k=3)

    assert len(results) == 3


def test_default_bm25_build_index():
    mock_db = MagicMock(spec=Session)
    mock_db.query.return_value.all.return_value = []

    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = []

    # When bm25_retriever is None, HybridRetriever builds BM25 index using db
    retriever = HybridRetriever(dense_retriever=mock_dense, bm25_retriever=None)
    results = retriever.retrieve(mock_db, "query", top_k=5)

    assert results == []
    mock_db.query.assert_called()


if __name__ == "__main__":
    import unittest

    class TestHybridRetrieverUnittest(unittest.TestCase):
        def test_both_retrievers_invoked_and_fused(self):
            test_both_retrievers_invoked_and_fused()

        def test_empty_retrieval(self):
            test_empty_retrieval()

        def test_top_k_respected(self):
            test_top_k_respected()

        def test_default_bm25_build_index(self):
            test_default_bm25_build_index()

    unittest.main()
