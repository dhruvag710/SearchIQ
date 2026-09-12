import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.chunk import Chunk
from app.retrieval.hybrid_search import HybridSearchResult
from app.retrieval.reranker import CrossEncoderReranker


def create_chunk(text: str = "sample passage") -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        text=text,
    )


def test_reranker_accepts_chunk_objects():
    chunk1 = create_chunk("Python programming")
    chunk2 = create_chunk("Baking cookies")
    chunks = [chunk1, chunk2]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.1]

    reranker = CrossEncoderReranker(cross_encoder=mock_model)
    results = reranker.rerank("python", chunks, top_k=2)

    mock_model.predict.assert_called_once_with(
        [("python", "Python programming"), ("python", "Baking cookies")],
        show_progress_bar=False,
    )
    assert len(results) == 2
    assert results[0].chunk_id == str(chunk1.id)
    assert results[0].reranker_score == 0.9
    assert results[1].chunk_id == str(chunk2.id)
    assert results[1].reranker_score == 0.1


def test_reranker_accepts_hybrid_search_results():
    hs1 = HybridSearchResult(
        chunk_id="11111111-1111-1111-1111-111111111111",
        document_id="22222222-2222-2222-2222-222222222222",
        page_number=1,
        chunk_index=0,
        text="Sample text",
        vector_score=0.8,
        keyword_score=0.6,
        final_score=0.7,
    )

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.95]

    reranker = CrossEncoderReranker(cross_encoder=mock_model)
    results = reranker.rerank("sample", [hs1], top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "11111111-1111-1111-1111-111111111111"
    assert results[0].hybrid_score == 0.7
    assert results[0].reranker_score == 0.95


def test_reranker_top_k_limiting():
    chunks = [create_chunk(f"Passage {i}") for i in range(10)]
    mock_model = MagicMock()
    mock_model.predict.return_value = [float(i) for i in range(10)]

    reranker = CrossEncoderReranker(cross_encoder=mock_model)
    results = reranker.rerank("query", chunks, top_k=3)

    assert len(results) == 3
    # Highest score (9.0) should rank first
    assert results[0].reranker_score == 9.0


if __name__ == "__main__":
    import unittest

    class TestRerankerUnittest(unittest.TestCase):
        def test_reranker_accepts_chunk_objects(self):
            test_reranker_accepts_chunk_objects()

        def test_reranker_accepts_hybrid_search_results(self):
            test_reranker_accepts_hybrid_search_results()

        def test_reranker_top_k_limiting(self):
            test_reranker_top_k_limiting()

    unittest.main()
