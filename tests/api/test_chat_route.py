"""Tests for the /chat API route — verifies VisualSearchService wiring and response metadata."""
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.chat_service import ChatResponse, ChatSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_source(
    doc_id: str = "doc-1", page: int = 1, chunk_idx: int = 0
) -> ChatSource:
    return ChatSource(
        document_id=doc_id,
        page_number=page,
        chunk_index=chunk_idx,
        is_visual=False,
        image_path=None,
    )


def _visual_source(
    doc_id: str = "doc-2",
    page: int = 3,
    chunk_idx: int = 1,
    image_path: str = "media/page_3_img_1.png",
) -> ChatSource:
    return ChatSource(
        document_id=doc_id,
        page_number=page,
        chunk_index=chunk_idx,
        is_visual=True,
        image_path=image_path,
    )


# ---------------------------------------------------------------------------
# Route wiring — VisualSearchService is injected
# ---------------------------------------------------------------------------


@patch("app.api.routes.chat.ChatService")
@patch("app.api.routes.chat.VisualSearchService")
def test_route_injects_visual_search_service(mock_vs_cls, mock_cs_cls):
    """The route must construct ChatService with a VisualSearchService instance."""
    from fastapi.testclient import TestClient
    from app.main import app

    mock_vs_instance = MagicMock()
    mock_vs_cls.return_value = mock_vs_instance

    mock_service = MagicMock()
    mock_service.chat.return_value = ChatResponse(answer="Answer", sources=[])
    mock_cs_cls.return_value = mock_service

    client = TestClient(app)
    resp = client.post("/chat", json={"question": "test?"})

    assert resp.status_code == 200

    # VisualSearchService must be instantiated
    mock_vs_cls.assert_called_once()

    # ChatService must receive visual_search= keyword argument
    mock_cs_cls.assert_called_once_with(visual_search=mock_vs_instance)


# ---------------------------------------------------------------------------
# Visual retrieval is invoked through the route
# ---------------------------------------------------------------------------


@patch("app.api.routes.chat.ChatService")
@patch("app.api.routes.chat.VisualSearchService")
def test_visual_retrieval_invoked_via_route(mock_vs_cls, mock_cs_cls):
    """ChatService.chat() is called with the DB session and the user's question."""
    from fastapi.testclient import TestClient
    from app.main import app

    mock_service = MagicMock()
    mock_service.chat.return_value = ChatResponse(answer="Yes", sources=[])
    mock_cs_cls.return_value = mock_service

    client = TestClient(app)
    resp = client.post("/chat", json={"question": "what about the graph?"})

    assert resp.status_code == 200
    mock_service.chat.assert_called_once()
    call_args = mock_service.chat.call_args
    assert call_args[0][1] == "what about the graph?"


# ---------------------------------------------------------------------------
# Visual source metadata reaches JSON response
# ---------------------------------------------------------------------------


@patch("app.api.routes.chat.ChatService")
@patch("app.api.routes.chat.VisualSearchService")
def test_visual_source_metadata_in_response(mock_vs_cls, mock_cs_cls):
    """is_visual and image_path must appear in the response JSON for visual sources."""
    from fastapi.testclient import TestClient
    from app.main import app

    mock_service = MagicMock()
    mock_service.chat.return_value = ChatResponse(
        answer="Revenue increased",
        sources=[
            _text_source(doc_id="doc-txt"),
            _visual_source(doc_id="doc-vis", page=6, image_path="media/page_6_img_0.png"),
        ],
    )
    mock_cs_cls.return_value = mock_service

    client = TestClient(app)
    resp = client.post("/chat", json={"question": "revenue chart?"})

    assert resp.status_code == 200
    data = resp.json()

    assert len(data["sources"]) == 2

    text_src = data["sources"][0]
    assert text_src["document_id"] == "doc-txt"
    assert text_src["is_visual"] is False
    assert text_src["image_path"] is None

    visual_src = data["sources"][1]
    assert visual_src["document_id"] == "doc-vis"
    assert visual_src["page_number"] == 6
    assert visual_src["is_visual"] is True
    assert visual_src["image_path"] == "media/page_6_img_0.png"


# ---------------------------------------------------------------------------
# Text-only sources still work (backward compat)
# ---------------------------------------------------------------------------


@patch("app.api.routes.chat.ChatService")
@patch("app.api.routes.chat.VisualSearchService")
def test_text_only_sources_backward_compatible(mock_vs_cls, mock_cs_cls):
    """When only text sources are returned, is_visual defaults to False and image_path to None."""
    from fastapi.testclient import TestClient
    from app.main import app

    mock_service = MagicMock()
    mock_service.chat.return_value = ChatResponse(
        answer="Text answer",
        sources=[_text_source(doc_id="doc-1", page=2, chunk_idx=3)],
    )
    mock_cs_cls.return_value = mock_service

    client = TestClient(app)
    resp = client.post("/chat", json={"question": "text only question"})

    assert resp.status_code == 200
    data = resp.json()

    assert len(data["sources"]) == 1
    src = data["sources"][0]
    assert src["document_id"] == "doc-1"
    assert src["page_number"] == 2
    assert src["chunk_index"] == 3
    assert src["is_visual"] is False
    assert src["image_path"] is None


# ---------------------------------------------------------------------------
# Empty sources
# ---------------------------------------------------------------------------


@patch("app.api.routes.chat.ChatService")
@patch("app.api.routes.chat.VisualSearchService")
def test_empty_sources_response(mock_vs_cls, mock_cs_cls):
    """Route handles empty sources list gracefully."""
    from fastapi.testclient import TestClient
    from app.main import app

    mock_service = MagicMock()
    mock_service.chat.return_value = ChatResponse(answer="No context", sources=[])
    mock_cs_cls.return_value = mock_service

    client = TestClient(app)
    resp = client.post("/chat", json={"question": "unknown topic"})

    assert resp.status_code == 200
    assert resp.json()["sources"] == []
