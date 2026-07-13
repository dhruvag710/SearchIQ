import os

from google import genai

API_KEY_ENV_VAR = "GEMINI_API_KEY"
MODEL_ENV_VAR = "GEMINI_MODEL"
DEFAULT_MODEL_NAME = "gemini-2.5-flash"

_client: genai.Client | None = None


class MissingGeminiApiKeyError(RuntimeError):
    """Raised when the Gemini API key environment variable is not set."""


class GeminiGenerationError(RuntimeError):
    """Raised when Gemini fails to generate a usable response."""


class LLMService:
    """Generate answers from a complete prompt using Google Gemini."""

    def __init__(self, client: genai.Client | None = None) -> None:
        self._client = client

    def generate(self, prompt: str) -> str:
        """Send a prompt to Gemini and return the generated answer text."""
        try:
            response = self._get_client().models.generate_content(
                model=_get_model_name(),
                contents=prompt,
            )
        except MissingGeminiApiKeyError:
            raise
        except Exception as exc:
            raise GeminiGenerationError(f"Gemini request failed: {exc}") from exc

        if response.text is None or not response.text.strip():
            raise GeminiGenerationError("Gemini returned an empty response.")

        return response.text.strip()

    def _get_client(self) -> genai.Client:
        """Return the injected client or the shared lazy-loaded Gemini client."""
        if self._client is not None:
            return self._client
        return _get_default_client()


def _get_model_name() -> str:
    """Read the configured Gemini model name from the environment."""
    return os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_NAME)


def _get_default_client() -> genai.Client:
    """Create the process-wide Gemini client on first use."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise MissingGeminiApiKeyError(
            f"Missing Gemini API key. Set the {API_KEY_ENV_VAR} environment variable."
        )

    _client = genai.Client(api_key=api_key)
    return _client
