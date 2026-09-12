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
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, RerankedResult
from app.services.chat_service import ChatResponse, ChatService, ChatSource


def create_chunk(text: str = "sample passage") -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        text=text,
    )


def test_chat_service_hybrid_pipeline():
    chunk1 = create_chunk("Passage 1 content")
    chunk2 = create_chunk("Passage 2 content")

    mock_db = MagicMock(spec=Session)

    mock_hybrid_retriever = MagicMock(spec=HybridRetriever)
    mock_hybrid_retriever.retrieve.return_value = [chunk1, chunk2]

    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    mock_reranker.rerank.return_value = [
        RerankedResult(
            chunk_id=str(chunk1.id),
            document_id=str(chunk1.document_id),
            page_number=chunk1.page_number,
            chunk_index=chunk1.chunk_index,
            text=chunk1.text,
            hybrid_score=0.0,
            reranker_score=0.95,
        )
    ]

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build.return_value = "Formatted Prompt"

    mock_llm_service = MagicMock()
    mock_llm_service.generate.return_value = "Generated answer from LLM."

    service = ChatService(
        hybrid_retriever=mock_hybrid_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    response = service.chat(mock_db, "What is SearchIQ?", top_k=5)

    # 1. Verify HybridRetriever was invoked with candidate_k pool size 25
    mock_hybrid_retriever.retrieve.assert_called_once_with(mock_db, "What is SearchIQ?", top_k=25)

    # 2. Verify Cross-Encoder Reranker was invoked with chunks and top_k=5
    mock_reranker.rerank.assert_called_once_with("What is SearchIQ?", [chunk1, chunk2], top_k=5)

    # 3. Verify PromptBuilder was invoked
    mock_prompt_builder.build.assert_called_once()

    # 4. Verify LLM Service was invoked
    mock_llm_service.generate.assert_called_once_with("Formatted Prompt")

    # 5. Verify response shape
    assert isinstance(response, ChatResponse)
    assert response.answer == "Generated answer from LLM."
    assert len(response.sources) == 1
    assert response.sources[0] == ChatSource(
        document_id=str(chunk1.document_id),
        page_number=1,
        chunk_index=0,
    )


def test_chat_service_candidate_pool_and_top_k_limiting():
    # 30 candidate chunks
    chunks = [create_chunk(f"Candidate passage {i}") for i in range(30)]

    mock_db = MagicMock(spec=Session)

    mock_hybrid_retriever = MagicMock(spec=HybridRetriever)
    mock_hybrid_retriever.retrieve.return_value = chunks

    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    # Reranker trims candidates to top 5
    mock_reranker.rerank.return_value = [
        RerankedResult(
            chunk_id=str(c.id),
            document_id=str(c.document_id),
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            text=c.text,
            hybrid_score=0.0,
            reranker_score=0.9 - (i * 0.01),
        )
        for i, c in enumerate(chunks[:5])
    ]

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build.return_value = "Formatted Prompt"

    mock_llm_service = MagicMock()
    mock_llm_service.generate.return_value = "Answer"

    service = ChatService(
        hybrid_retriever=mock_hybrid_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    response = service.chat(mock_db, "Query", top_k=5)

    # Verify retrieval requested 25 candidates
    mock_hybrid_retriever.retrieve.assert_called_once_with(mock_db, "Query", top_k=25)

    # Verify reranker received all 30 candidate chunks and top_k=5 limit
    mock_reranker.rerank.assert_called_once_with("Query", chunks, top_k=5)

    # Verify response sources are limited to 5
    assert len(response.sources) == 5


if __name__ == "__main__":
    import unittest

    class TestChatServiceUnittest(unittest.TestCase):
        def test_chat_service_hybrid_pipeline(self):
            test_chat_service_hybrid_pipeline()

        def test_chat_service_candidate_pool_and_top_k_limiting(self):
            test_chat_service_candidate_pool_and_top_k_limiting()

    unittest.main()
