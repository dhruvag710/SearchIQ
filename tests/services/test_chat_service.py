import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from sqlalchemy.orm import Session

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.chunk import Chunk
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, RerankedResult
from app.retrieval.visual_search import (
    RetrievalCandidate,
    VisualSearchResult,
    VisualSearchService,
)
from app.services.chat_service import ChatResponse, ChatService, ChatSource


def create_chunk(text: str = "sample passage") -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        page_number=1,
        chunk_index=0,
        text=text,
    )


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Multimodal integration tests
# ---------------------------------------------------------------------------


def _make_visual_search_result(
    *,
    img_id: str = "img-001",
    doc_id: str = "doc-001",
    page_number: int = 2,
    image_index: int = 0,
    image_path: str = "media/page_2_img_0.png",
    caption: str = "A bar chart showing Q3 revenue.",
    similarity_score: float = 0.87,
) -> VisualSearchResult:
    return VisualSearchResult(
        id=img_id,
        document_id=doc_id,
        page_number=page_number,
        image_index=image_index,
        image_path=image_path,
        caption=caption,
        similarity_score=similarity_score,
    )


def test_chat_service_with_visual_search():
    """Visual retrieval is invoked, candidates are merged, and source metadata survives."""
    chunk = create_chunk("Text passage about revenue")
    visual = _make_visual_search_result(
        img_id="visual-uuid-1",
        doc_id="doc-visual-1",
        page_number=3,
        image_index=1,
        image_path="media/page_3_img_1.png",
        caption="Revenue growth chart",
    )

    mock_db = MagicMock(spec=Session)

    mock_hybrid = MagicMock(spec=HybridRetriever)
    mock_hybrid.retrieve.return_value = [chunk]

    mock_visual = MagicMock(spec=VisualSearchService)
    mock_visual.search.return_value = [visual]

    # Reranker returns one text result and one visual result
    text_reranked = RerankedResult(
        chunk_id=str(chunk.id),
        document_id=str(chunk.document_id),
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        hybrid_score=0.0,
        reranker_score=0.90,
    )
    visual_reranked = RerankedResult(
        chunk_id="visual-uuid-1",
        document_id="doc-visual-1",
        page_number=3,
        chunk_index=1,
        text="Revenue growth chart",
        hybrid_score=0.0,
        reranker_score=0.85,
    )

    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    mock_reranker.rerank.return_value = [text_reranked, visual_reranked]

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build.return_value = "Prompt"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Answer with visual"

    service = ChatService(
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm,
        visual_search=mock_visual,
    )

    response = service.chat(mock_db, "Q3 revenue chart", top_k=5, visual_k=5)

    # Visual search must be invoked
    mock_visual.search.assert_called_once_with(mock_db, "Q3 revenue chart", top_k=5)

    # Reranker must receive unified candidates (text + visual)
    rerank_call_args = mock_reranker.rerank.call_args
    candidates_sent = rerank_call_args[0][1]
    assert len(candidates_sent) == 2  # 1 text + 1 visual
    assert all(isinstance(c, RetrievalCandidate) for c in candidates_sent)

    # The visual candidate's text should be the caption
    visual_candidate = [c for c in candidates_sent if c.is_visual][0]
    assert visual_candidate.text == "Revenue growth chart"
    assert visual_candidate.image_path == "media/page_3_img_1.png"

    # Text candidate should not be visual
    text_candidate = [c for c in candidates_sent if not c.is_visual][0]
    assert text_candidate.text == "Text passage about revenue"
    assert text_candidate.image_path is None

    # Sources must carry visual metadata
    assert len(response.sources) == 2
    text_source = response.sources[0]
    visual_source = response.sources[1]

    assert text_source.is_visual is False
    assert text_source.image_path is None

    assert visual_source.is_visual is True
    assert visual_source.image_path == "media/page_3_img_1.png"
    assert visual_source.document_id == "doc-visual-1"
    assert visual_source.page_number == 3
    assert visual_source.chunk_index == 1


def test_chat_service_without_visual_search_regression():
    """When visual_search is None, raw chunks reach the reranker directly (backward compat)."""
    chunk = create_chunk("Plain text chunk")

    mock_db = MagicMock(spec=Session)

    mock_hybrid = MagicMock(spec=HybridRetriever)
    mock_hybrid.retrieve.return_value = [chunk]

    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    mock_reranker.rerank.return_value = [
        RerankedResult(
            chunk_id=str(chunk.id),
            document_id=str(chunk.document_id),
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            hybrid_score=0.0,
            reranker_score=0.9,
        )
    ]

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build.return_value = "Prompt"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Answer"

    service = ChatService(
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm,
        # visual_search intentionally omitted
    )

    response = service.chat(mock_db, "question", top_k=5)

    # Reranker should receive raw Chunk objects, NOT RetrievalCandidate wrappers
    rerank_candidates = mock_reranker.rerank.call_args[0][1]
    assert rerank_candidates == [chunk]
    assert isinstance(rerank_candidates[0], Chunk)

    # Source should be backward-compatible (is_visual=False, image_path=None)
    assert response.sources[0].is_visual is False
    assert response.sources[0].image_path is None


def test_chat_service_visual_ids_passed_to_prompt_builder():
    """Verify that visual_ids kwarg is forwarded to PromptBuilder.build()."""
    chunk = create_chunk("Some text")
    visual = _make_visual_search_result(img_id="vis-id-42")

    mock_db = MagicMock(spec=Session)

    mock_hybrid = MagicMock(spec=HybridRetriever)
    mock_hybrid.retrieve.return_value = [chunk]

    mock_visual = MagicMock(spec=VisualSearchService)
    mock_visual.search.return_value = [visual]

    mock_reranker = MagicMock(spec=CrossEncoderReranker)
    mock_reranker.rerank.return_value = []

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build.return_value = "Prompt"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Answer"

    service = ChatService(
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm,
        visual_search=mock_visual,
    )

    service.chat(mock_db, "query", top_k=5)

    # PromptBuilder.build must receive visual_ids as a keyword argument
    build_call = mock_prompt_builder.build.call_args
    assert "visual_ids" in build_call.kwargs
    assert "vis-id-42" in build_call.kwargs["visual_ids"]
