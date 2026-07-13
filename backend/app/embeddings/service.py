from app.chunking.models import Chunk
from app.embeddings.embedder import BGEEmbedder

_default_embedder: BGEEmbedder | None = None


def _get_default_embedder() -> BGEEmbedder:
    """Return a process-wide embedder instance, loading the model on first use."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = BGEEmbedder()
    return _default_embedder


def embed_chunks(
    chunks: list[Chunk],
    embedder: BGEEmbedder | None = None,
) -> list[tuple[Chunk, list[float]]]:
    """Generate embeddings for each chunk and return (chunk, embedding) pairs."""
    if not chunks:
        return []

    active_embedder = embedder or _get_default_embedder()
    texts = [chunk.text for chunk in chunks]
    embeddings = active_embedder.embed_many(texts)
    return list(zip(chunks, embeddings, strict=True))
