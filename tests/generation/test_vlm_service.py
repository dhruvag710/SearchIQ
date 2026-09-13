import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.generation.vlm_service import (
    CAPTION_PROMPT,
    MissingOpenRouterApiKeyError,
    VisualCaptioner,
    VLMGenerationError,
    VLMImageNotFoundError,
    VLMService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(content: str = "A chart showing revenue growth.") -> MagicMock:
    """Build a mock OpenAI client whose chat.completions.create returns *content*."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _create_test_image(tmp_path: Path, name: str = "test_image.png") -> Path:
    """Write a tiny valid PNG file to *tmp_path* and return its path."""
    # Minimal 1x1 white PNG (67 bytes)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
        b"\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path = tmp_path / name
    img_path.write_bytes(png_bytes)
    return img_path


# ---------------------------------------------------------------------------
# Tests — Successful caption generation
# ---------------------------------------------------------------------------

class TestSuccessfulCaption:
    """Verify that a well-formed image produces a caption string."""

    def test_caption_returned(self, tmp_path: Path) -> None:
        expected = "A bar chart showing quarterly revenue growth from 2023 to 2025."
        mock_client = _make_mock_client(content=expected)
        captioner = VisualCaptioner(client=mock_client)

        img = _create_test_image(tmp_path)
        result = captioner.caption_image(img)

        assert result == expected
        mock_client.chat.completions.create.assert_called_once()

    def test_caption_strips_whitespace(self, tmp_path: Path) -> None:
        mock_client = _make_mock_client(content="  padded caption  \n")
        captioner = VisualCaptioner(client=mock_client)

        img = _create_test_image(tmp_path)
        result = captioner.caption_image(img)

        assert result == "padded caption"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        mock_client = _make_mock_client(content="description")
        captioner = VisualCaptioner(client=mock_client)

        img = _create_test_image(tmp_path)
        result = captioner.caption_image(str(img))  # pass as str, not Path

        assert result == "description"

    def test_custom_prompt_forwarded(self, tmp_path: Path) -> None:
        custom = "Describe the diagram for accessibility purposes."
        mock_client = _make_mock_client(content="ok")
        captioner = VisualCaptioner(client=mock_client)

        img = _create_test_image(tmp_path)
        captioner.caption_image(img, prompt=custom)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        content_blocks = messages[0]["content"]
        text_block = next(b for b in content_blocks if b["type"] == "text")
        assert text_block["text"] == custom


# ---------------------------------------------------------------------------
# Tests — Base64 payload construction
# ---------------------------------------------------------------------------

class TestBase64Payload:
    """Ensure the image is correctly read, base64-encoded, and sent as a data URL."""

    def test_data_url_contains_base64(self, tmp_path: Path) -> None:
        img = _create_test_image(tmp_path)
        raw_bytes = img.read_bytes()
        expected_b64 = base64.b64encode(raw_bytes).decode("utf-8")

        mock_client = _make_mock_client()
        captioner = VisualCaptioner(client=mock_client)
        captioner.caption_image(img)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        content_blocks = messages[0]["content"]
        image_block = next(b for b in content_blocks if b["type"] == "image_url")
        data_url = image_block["image_url"]["url"]

        assert data_url.startswith("data:image/png;base64,")
        assert expected_b64 in data_url

    def test_jpeg_mime_type(self, tmp_path: Path) -> None:
        """A .jpg file should produce a data:image/jpeg data URL."""
        img = _create_test_image(tmp_path, name="photo.jpg")

        mock_client = _make_mock_client()
        captioner = VisualCaptioner(client=mock_client)
        captioner.caption_image(img)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        content_blocks = messages[0]["content"]
        image_block = next(b for b in content_blocks if b["type"] == "image_url")
        data_url = image_block["image_url"]["url"]

        assert data_url.startswith("data:image/jpeg;base64,")

    def test_default_prompt_used(self, tmp_path: Path) -> None:
        """When no custom prompt is given, the module-level CAPTION_PROMPT is used."""
        mock_client = _make_mock_client()
        captioner = VisualCaptioner(client=mock_client)

        img = _create_test_image(tmp_path)
        captioner.caption_image(img)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        content_blocks = messages[0]["content"]
        text_block = next(b for b in content_blocks if b["type"] == "text")
        assert text_block["text"] == CAPTION_PROMPT


# ---------------------------------------------------------------------------
# Tests — Missing image file
# ---------------------------------------------------------------------------

class TestMissingImage:

    def test_nonexistent_path_raises(self) -> None:
        captioner = VisualCaptioner(client=_make_mock_client())
        with pytest.raises(VLMImageNotFoundError, match="Image file not found"):
            captioner.caption_image("/does/not/exist/img.png")

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        """A directory (not a file) should be rejected."""
        captioner = VisualCaptioner(client=_make_mock_client())
        with pytest.raises(VLMImageNotFoundError, match="Image file not found"):
            captioner.caption_image(tmp_path)


# ---------------------------------------------------------------------------
# Tests — API / client failure
# ---------------------------------------------------------------------------

class TestAPIFailure:

    def test_openai_exception_wrapped(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("connection refused")

        captioner = VisualCaptioner(client=mock_client)
        img = _create_test_image(tmp_path)

        with pytest.raises(VLMGenerationError, match="OpenRouter VLM request failed"):
            captioner.caption_image(img)

    def test_timeout_exception_wrapped(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("request timed out")

        captioner = VisualCaptioner(client=mock_client)
        img = _create_test_image(tmp_path)

        with pytest.raises(VLMGenerationError, match="OpenRouter VLM request failed"):
            captioner.caption_image(img)

    def test_original_exception_chained(self, tmp_path: Path) -> None:
        original = ValueError("bad request payload")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = original

        captioner = VisualCaptioner(client=mock_client)
        img = _create_test_image(tmp_path)

        with pytest.raises(VLMGenerationError) as exc_info:
            captioner.caption_image(img)

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests — Empty / invalid model response
# ---------------------------------------------------------------------------

class TestEmptyOrInvalidResponse:

    def test_none_content_raises(self, tmp_path: Path) -> None:
        mock_client = _make_mock_client(content=None)
        captioner = VisualCaptioner(client=mock_client)
        img = _create_test_image(tmp_path)

        with pytest.raises(VLMGenerationError, match="empty response"):
            captioner.caption_image(img)

    def test_empty_string_raises(self, tmp_path: Path) -> None:
        mock_client = _make_mock_client(content="")
        captioner = VisualCaptioner(client=mock_client)
        img = _create_test_image(tmp_path)

        with pytest.raises(VLMGenerationError, match="empty response"):
            captioner.caption_image(img)

    def test_whitespace_only_raises(self, tmp_path: Path) -> None:
        mock_client = _make_mock_client(content="   \n\t  ")
        captioner = VisualCaptioner(client=mock_client)
        img = _create_test_image(tmp_path)

        with pytest.raises(VLMGenerationError, match="empty response"):
            captioner.caption_image(img)

    def test_no_choices_raises(self, tmp_path: Path) -> None:
        """Response with empty choices list should raise."""
        mock_response = MagicMock()
        mock_response.choices = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        captioner = VisualCaptioner(client=mock_client)
        img = _create_test_image(tmp_path)

        with pytest.raises((VLMGenerationError, IndexError)):
            captioner.caption_image(img)


# ---------------------------------------------------------------------------
# Tests — VLMService alias
# ---------------------------------------------------------------------------

class TestVLMServiceAlias:

    def test_alias_is_visual_captioner(self) -> None:
        assert VLMService is VisualCaptioner

    def test_alias_instantiation(self, tmp_path: Path) -> None:
        mock_client = _make_mock_client(content="alias works")
        svc = VLMService(client=mock_client)
        img = _create_test_image(tmp_path)

        assert svc.caption_image(img) == "alias works"
