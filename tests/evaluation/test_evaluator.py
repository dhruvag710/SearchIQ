import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy.orm import Session

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.evaluation.dataset import EvaluationTestCase, GoldenDataset
from app.evaluation.evaluator import RetrievalEvaluator, extract_chunk_ids
from app.evaluation.reporter import EvaluationReporter
from app.models.chunk import Chunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, RerankedResult
from app.retrieval.vector_search import VectorSearchResult, VectorSearchService


def create_mock_chunk(chunk_id_str: str) -> Chunk:
    return Chunk(
        id=uuid.UUID(chunk_id_str),
        document_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        text="Sample text",
    )


def test_chunk_id_normalization():
    chunk_uuid = uuid.uuid4()
    orm_chunk = Chunk(
        id=chunk_uuid,
        document_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        text="Text",
    )
    vs_res = VectorSearchResult(
        chunk_id="vs-123",
        document_id="doc-123",
        page_number=1,
        chunk_index=0,
        text="Text",
        similarity_score=0.9,
    )
    rerank_res = RerankedResult(
        chunk_id="rr-456",
        document_id="doc-123",
        page_number=1,
        chunk_index=0,
        text="Text",
        hybrid_score=0.5,
        reranker_score=0.95,
    )

    ids = extract_chunk_ids([orm_chunk, vs_res, rerank_res])
    assert ids == [str(chunk_uuid), "vs-123", "rr-456"]


def test_evaluator_orchestration_and_mocks():
    c1 = create_mock_chunk("11111111-1111-1111-1111-111111111111")
    c2 = create_mock_chunk("22222222-2222-2222-2222-222222222222")

    mock_db = MagicMock(spec=Session)

    mock_vector_search = MagicMock(spec=VectorSearchService)
    mock_vector_search.search.return_value = [
        VectorSearchResult(
            chunk_id=str(c1.id),
            document_id=str(c1.document_id),
            page_number=1,
            chunk_index=0,
            text=c1.text,
            similarity_score=0.9,
        )
    ]

    mock_bm25 = MagicMock(spec=BM25Retriever)
    mock_bm25.retrieve.return_value = [c2]

    mock_hybrid = MagicMock(spec=HybridRetriever)
    mock_hybrid.retrieve.return_value = [c1, c2]

    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    mock_reranker.rerank.return_value = [
        RerankedResult(
            chunk_id=str(c1.id),
            document_id=str(c1.document_id),
            page_number=1,
            chunk_index=0,
            text=c1.text,
            hybrid_score=0.5,
            reranker_score=0.99,
        )
    ]

    evaluator = RetrievalEvaluator(
        vector_search=mock_vector_search,
        bm25_retriever=mock_bm25,
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
    )

    dataset = GoldenDataset(
        dataset_name="test_dataset",
        description="Unit test evaluation dataset",
        test_cases=[
            EvaluationTestCase(
                id="q1",
                question="What is SearchIQ?",
                relevant_chunk_ids=[str(c1.id)],
            )
        ],
    )

    report = evaluator.evaluate_dataset(mock_db, dataset)

    # 1. Verify BM25 index built
    mock_bm25.build_index.assert_called_once_with(mock_db)

    # 2. Verify all strategies received the exact same question
    mock_vector_search.search.assert_called_once_with(mock_db, "What is SearchIQ?", top_k=25)
    mock_bm25.retrieve.assert_called_once_with("What is SearchIQ?", top_k=25)
    mock_hybrid.retrieve.assert_called_once_with(mock_db, "What is SearchIQ?", top_k=25)
    mock_reranker.rerank.assert_called_once_with("What is SearchIQ?", [c1, c2], top_k=5)

    # 3. Verify report structure
    assert report.total_test_cases == 1
    assert len(report.strategies) == 4

    # 4. Verify Strategy D (Reranked) has recall_at_25 = None
    reranked_strat = report.strategies[3]
    assert reranked_strat.strategy_name == "D. Hybrid + Reranker"
    assert reranked_strat.metrics.recall_at_25 is None
    assert reranked_strat.metrics.recall_at_5 == 1.0

    # 5. Verify reporter formats cleanly
    markdown = EvaluationReporter.format_report(report)
    assert "| **D. Hybrid + Reranker** |" in markdown
    assert "N/A" in markdown

    console = EvaluationReporter.format_console(report)
    assert "SEARCHIQ RETRIEVAL BENCHMARK REPORT" in console


if __name__ == "__main__":
    import unittest

    class TestEvaluatorUnittest(unittest.TestCase):
        def test_chunk_id_normalization(self):
            test_chunk_id_normalization()

        def test_evaluator_orchestration_and_mocks(self):
            test_evaluator_orchestration_and_mocks()

    unittest.main()
