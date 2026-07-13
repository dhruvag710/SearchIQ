from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class BGEEmbedder:
    """Sentence-transformer wrapper for BAAI/bge-small-en-v1.5 embeddings."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Generate a normalized embedding for a single string."""
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embeddings for multiple strings."""
        if not texts:
            return []

        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]
