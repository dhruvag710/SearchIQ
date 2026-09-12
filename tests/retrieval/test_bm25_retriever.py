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


def create_mock_chunk(text: str, page_number: int = 1, chunk_index: int = 0) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
    )


def test_empty_database():
    mock_db = MagicMock(spec=Session)
    mock_db.query.return_value.all.return_value = []

    retriever = BM25Retriever()
    retriever.build_index(mock_db)

    results = retriever.retrieve("python")
    assert results == []


def test_index_creation_and_retrieval_ordering():
    chunk1 = create_mock_chunk("Python programming language for data science and machine learning")
    chunk2 = create_mock_chunk("Baking delicious chocolate chip cookies at home")
    chunk3 = create_mock_chunk("Python async programming tutorial and web APIs")

    mock_db = MagicMock(spec=Session)
    mock_db.query.return_value.all.return_value = [chunk1, chunk2, chunk3]

    retriever = BM25Retriever()
    retriever.build_index(mock_db)

    results = retriever.retrieve("python programming", top_k=3)
    assert len(results) == 3

    # Chunks mentioning python programming should rank above baking cookies
    assert results[0] in [chunk1, chunk3]
    assert results[1] in [chunk1, chunk3]
    assert results[2] == chunk2


def test_top_k_behaviour():
    chunks = [create_mock_chunk(f"Python document chunk number {i}") for i in range(10)]

    mock_db = MagicMock(spec=Session)
    mock_db.query.return_value.all.return_value = chunks

    retriever = BM25Retriever()
    retriever.build_index(mock_db)

    results_k2 = retriever.retrieve("python", top_k=2)
    assert len(results_k2) == 2

    results_k5 = retriever.retrieve("python", top_k=5)
    assert len(results_k5) == 5

    results_k20 = retriever.retrieve("python", top_k=20)
    assert len(results_k20) == 10

    results_k0 = retriever.retrieve("python", top_k=0)
    assert results_k0 == []


def test_empty_query():
    chunk1 = create_mock_chunk("Python programming")
    mock_db = MagicMock(spec=Session)
    mock_db.query.return_value.all.return_value = [chunk1]

    retriever = BM25Retriever()
    retriever.build_index(mock_db)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


if __name__ == "__main__":
    import unittest

    class TestBM25RetrieverUnittest(unittest.TestCase):
        def test_empty_database(self):
            test_empty_database()

        def test_index_creation_and_retrieval_ordering(self):
            test_index_creation_and_retrieval_ordering()

        def test_top_k_behaviour(self):
            test_top_k_behaviour()

        def test_empty_query(self):
            test_empty_query()

    unittest.main()
