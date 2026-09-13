import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
MODEL_ENV_VAR = "OPENROUTER_MODEL"
DEFAULT_MODEL_NAME = "google/gemini-2.5-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

CAPTION_PROMPT = (
    "Analyze this image extracted from a document and provide a detailed, factual description "
    "for downstream semantic search indexing.\n"
    "Prioritize:\n"
    "1. What the image, figure, table, chart, or diagram represents.\n"
    "2. Important entities, text labels, numeric values, key relationships, and trends shown.\n"
    "3. Useful searchable terminology relevant to the content.\n"
    "4. Factual description based strictly on what is visible rather than speculation."
)

_client: OpenAI | None = None


class MissingOpenRouterApiKeyError(RuntimeError):
    """Raised when the OpenRouter API key environment variable is not set."""


class VLMImageNotFoundError(FileNotFoundError):
    """Raised when the target image file does not exist on disk."""


class VLMGenerationError(RuntimeError):
    """Raised when the VLM service fails to generate a usable caption."""


class VisualCaptioner:
    """Generate detailed textual descriptions for document images using an OpenRouter VLM."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client

    def caption_image(self, image_path: Path | str, prompt: str | None = None) -> str:
        """Read a local image file, encode it as a base64 Data URL, and request a visual caption."""
        path = Path(image_path)
        if not path.is_file():
            raise VLMImageNotFoundError(f"Image file not found: {path}")

        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"

        try:
            image_bytes = path.read_bytes()
        except Exception as exc:
            raise VLMGenerationError(f"Failed to read image file {path}: {exc}") from exc

        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_encoded}"

        caption_prompt = prompt or CAPTION_PROMPT

        try:
            response = self._get_client().chat.completions.create(
                model=_get_model_name(),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": caption_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
            )
        except MissingOpenRouterApiKeyError:
            raise
        except Exception as exc:
            raise VLMGenerationError(f"OpenRouter VLM request failed: {exc}") from exc

        message = None
        if response and getattr(response, "choices", None):
            first_choice = response.choices[0]
            if getattr(first_choice, "message", None):
                message = first_choice.message.content

        if message is None or not message.strip():
            raise VLMGenerationError("OpenRouter VLM returned an empty response.")

        return message.strip()

    def _get_client(self) -> OpenAI:
        """Return the injected client or the shared lazy-loaded OpenRouter client."""
        if self._client is not None:
            return self._client
        return _get_default_client()


VLMService = VisualCaptioner


def _get_model_name() -> str:
    """Read the configured OpenRouter model name from the environment."""
    return os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_NAME)


def _get_default_client() -> OpenAI:
    """Create the process-wide OpenRouter client on first use."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise MissingOpenRouterApiKeyError(
            f"Missing OpenRouter API key. Set the {API_KEY_ENV_VAR} environment variable."
        )

    _client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )
    return _client
