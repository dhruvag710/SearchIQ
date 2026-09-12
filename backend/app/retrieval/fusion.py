from app.models.chunk import Chunk


def reciprocal_rank_fusion(
    dense_results: list[Chunk],
    bm25_results: list[Chunk],
    *,
    k: int = 60,
) -> list[Chunk]:
    """Combine dense and BM25 search results using Reciprocal Rank Fusion (RRF).

    RRF Score for a document chunk = sum(1 / (k + rank_i)) across result lists.
    """
    scores: dict[object, float] = {}
    chunk_map: dict[object, Chunk] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        chunk_key = chunk.id
        chunk_map[chunk_key] = chunk
        scores[chunk_key] = scores.get(chunk_key, 0.0) + (1.0 / (k + rank))

    for rank, chunk in enumerate(bm25_results, start=1):
        chunk_key = chunk.id
        chunk_map[chunk_key] = chunk
        scores[chunk_key] = scores.get(chunk_key, 0.0) + (1.0 / (k + rank))

    sorted_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [chunk_map[key] for key in sorted_keys]
