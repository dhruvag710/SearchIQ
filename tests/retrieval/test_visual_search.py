import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.retrieval.visual_search import (
    RetrievalCandidate,
    VisualSearchResult,
    VisualSearchService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_embedder(embedding: list[float] | None = None) -> MagicMock:
    """Return a mock BGEEmbedder that produces a fixed 384-d vector."""
    embedder = MagicMock()
    embedder.embed.return_value = embedding or [0.01] * 384
    return embedder


def _make_db_row(
    *,
    img_id: uuid.UUID | None = None,
    doc_id: uuid.UUID | None = None,
    page_number: int = 1,
    image_index: int = 0,
    image_path: str = "media/page_1_img_0.png",
    caption: str = "A bar chart showing Q3 revenue.",
    similarity_score: float = 0.87,
) -> SimpleNamespace:
    """Simulate a SQLAlchemy result row from the document_images cosine query."""
    return SimpleNamespace(
        id=img_id or uuid.uuid4(),
        document_id=doc_id or uuid.uuid4(),
        page_number=page_number,
        image_index=image_index,
        image_path=image_path,
        caption=caption,
        similarity_score=similarity_score,
    )


# ---------------------------------------------------------------------------
# VisualSearchService — query embedding
# ---------------------------------------------------------------------------

class TestVisualSearchQueryEmbedding:

    def test_embed_query_uses_bge_prefix(self) -> None:
        mock_embedder = _make_mock_embedder()
        service = VisualSearchService(embedder=mock_embedder)

        service._embed_query("revenue chart")

        call_args = mock_embedder.embed.call_args[0][0]
        assert call_args.startswith("Represent this sentence for searching relevant passages: ")
        assert "revenue chart" in call_args

    def test_search_calls_embed_and_db(self) -> None:
        mock_embedder = _make_mock_embedder()
        mock_db = MagicMock()
        mock_db.execute.return_value.all.return_value = []

        service = VisualSearchService(embedder=mock_embedder)
        results = service.search(mock_db, "test query", top_k=5)

        mock_embedder.embed.assert_called_once()
        assert results == []


# ---------------------------------------------------------------------------
# VisualSearchService — result metadata
# ---------------------------------------------------------------------------

class TestVisualSearchResults:

    def test_returns_correct_metadata(self) -> None:
        img_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        row = _make_db_row(
            img_id=img_id,
            doc_id=doc_id,
            page_number=3,
            image_index=2,
            image_path="media/page_3_img_2.png",
            caption="Flowchart of data pipeline",
            similarity_score=0.92,
        )

        mock_embedder = _make_mock_embedder()
        mock_db = MagicMock()
        mock_db.execute.return_value.all.return_value = [row]

        service = VisualSearchService(embedder=mock_embedder)
        results = service.search(mock_db, "data pipeline", top_k=5)

        assert len(results) == 1
        result = results[0]
        assert result.id == str(img_id)
        assert result.document_id == str(doc_id)
        assert result.page_number == 3
        assert result.image_index == 2
        assert result.image_path == "media/page_3_img_2.png"
        assert result.caption == "Flowchart of data pipeline"
        assert result.similarity_score == pytest.approx(0.92)

    def test_null_caption_becomes_empty_string(self) -> None:
        row = _make_db_row(caption=None)

        mock_embedder = _make_mock_embedder()
        mock_db = MagicMock()
        mock_db.execute.return_value.all.return_value = [row]

        service = VisualSearchService(embedder=mock_embedder)
        results = service.search(mock_db, "anything", top_k=5)

        assert results[0].caption == ""

    def test_top_k_respected(self) -> None:
        rows = [_make_db_row(similarity_score=0.9 - i * 0.05) for i in range(5)]

        mock_embedder = _make_mock_embedder()
        mock_db = MagicMock()
        mock_db.execute.return_value.all.return_value = rows

        service = VisualSearchService(embedder=mock_embedder)
        results = service.search(mock_db, "query", top_k=5)

        assert len(results) == 5


# ---------------------------------------------------------------------------
# RetrievalCandidate — from_chunk
# ---------------------------------------------------------------------------

class TestRetrievalCandidateFromChunk:

    def test_text_candidate_fields(self) -> None:
        chunk_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk = SimpleNamespace(
            id=chunk_id,
            document_id=doc_id,
            page_number=2,
            chunk_index=5,
            text="Some chunk text.",
        )

        candidate = RetrievalCandidate.from_chunk(chunk)

        assert candidate.id == chunk_id
        assert candidate.document_id == str(doc_id)
        assert candidate.page_number == 2
        assert candidate.chunk_index == 5
        assert candidate.text == "Some chunk text."
        assert candidate.is_visual is False
        assert candidate.image_path is None


# ---------------------------------------------------------------------------
# RetrievalCandidate — from_visual_result
# ---------------------------------------------------------------------------

class TestRetrievalCandidateFromVisualResult:

    def test_visual_candidate_fields(self) -> None:
        vsr = VisualSearchResult(
            id="img-uuid-123",
            document_id="doc-uuid-456",
            page_number=4,
            image_index=1,
            image_path="media/page_4_img_1.png",
            caption="Table of quarterly earnings.",
            similarity_score=0.88,
        )

        candidate = RetrievalCandidate.from_visual_result(vsr)

        assert candidate.id == "img-uuid-123"
        assert candidate.document_id == "doc-uuid-456"
        assert candidate.page_number == 4
        assert candidate.chunk_index == 1  # mapped from image_index
        assert candidate.text == "Table of quarterly earnings."  # mapped from caption
        assert candidate.is_visual is True
        assert candidate.image_path == "media/page_4_img_1.png"
