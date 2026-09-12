import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.chunk import Chunk
from app.retrieval.fusion import reciprocal_rank_fusion


def create_chunk(text: str = "sample text") -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        text=text,
    )


def test_empty_lists():
    c1 = create_chunk("Chunk 1")

    assert reciprocal_rank_fusion([], []) == []
    assert reciprocal_rank_fusion([c1], []) == [c1]
    assert reciprocal_rank_fusion([], [c1]) == [c1]


def test_overlapping_chunks():
    c1 = create_chunk("Chunk 1")
    c2 = create_chunk("Chunk 2")
    c3 = create_chunk("Chunk 3")

    # c1 is rank 1 in dense, rank 2 in BM25
    # c2 is rank 2 in dense, rank 1 in BM25
    # c3 is rank 3 in dense, absent in BM25
    dense_results = [c1, c2, c3]
    bm25_results = [c2, c1]

    results = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

    assert len(results) == 3
    # Overlapping chunks (c1 and c2) must rank above non-overlapping (c3)
    assert set(results[:2]) == {c1, c2}
    assert results[2] == c3


def test_unique_chunks():
    c1 = create_chunk("Chunk 1")
    c2 = create_chunk("Chunk 2")
    c3 = create_chunk("Chunk 3")
    c4 = create_chunk("Chunk 4")

    dense_results = [c1, c2]
    bm25_results = [c3, c4]

    results = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

    assert len(results) == 4
    assert set(results) == {c1, c2, c3, c4}


def test_identical_rankings():
    c1 = create_chunk("Chunk 1")
    c2 = create_chunk("Chunk 2")
    c3 = create_chunk("Chunk 3")

    dense_results = [c1, c2, c3]
    bm25_results = [c1, c2, c3]

    results = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

    assert results == [c1, c2, c3]


def test_different_rankings():
    c1 = create_chunk("Chunk 1")
    c2 = create_chunk("Chunk 2")
    c3 = create_chunk("Chunk 3")

    # c1: dense rank 1 (1/61), BM25 rank 3 (1/63) -> sum = 0.032266
    # c2: dense rank 2 (1/62), BM25 rank 1 (1/61) -> sum = 0.032786
    # c3: dense absent, BM25 rank 2 (1/62) -> sum = 0.016129
    dense_results = [c1, c2]
    bm25_results = [c2, c3, c1]

    results = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

    assert results == [c2, c1, c3]


if __name__ == "__main__":
    import unittest

    class TestFusionUnittest(unittest.TestCase):
        def test_empty_lists(self):
            test_empty_lists()

        def test_overlapping_chunks(self):
            test_overlapping_chunks()

        def test_unique_chunks(self):
            test_unique_chunks()

        def test_identical_rankings(self):
            test_identical_rankings()

        def test_different_rankings(self):
            test_different_rankings()

    unittest.main()
