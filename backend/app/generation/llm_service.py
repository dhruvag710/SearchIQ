import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
MODEL_ENV_VAR = "OPENROUTER_MODEL"
DEFAULT_MODEL_NAME = "google/gemini-2.5-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client: OpenAI | None = None


class MissingOpenRouterApiKeyError(RuntimeError):
    """Raised when the OpenRouter API key environment variable is not set."""


class OpenRouterGenerationError(RuntimeError):
    """Raised when OpenRouter fails to generate a usable response."""


class LLMService:
    """Generate answers from a complete prompt using OpenRouter."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self._client = client

    def generate(self, prompt: str) -> str:
        """Send a prompt to OpenRouter and return the generated answer text."""
        try:
            response = self._get_client().chat.completions.create(
                model=_get_model_name(),
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        except MissingOpenRouterApiKeyError:
            raise
        except Exception as exc:
            raise OpenRouterGenerationError(f"OpenRouter request failed: {exc}") from exc

        message = response.choices[0].message.content
        if message is None or not message.strip():
            raise OpenRouterGenerationError("OpenRouter returned an empty response.")

        return message.strip()

    def _get_client(self) -> OpenAI:
        """Return the injected client or the shared lazy-loaded OpenRouter client."""
        if self._client is not None:
            return self._client
        return _get_default_client()


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
