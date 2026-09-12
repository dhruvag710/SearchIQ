from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    """Calculated metric scores for a retrieval evaluation run."""

    recall_at_5: float
    recall_at_25: float | None
    mrr_at_5: float
    precision_at_5: float


def calculate_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Calculate Recall@K: proportion of relevant chunks retrieved in the top-K results.

    Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|
    """
    if not relevant_ids or k <= 0:
        return 0.0

    retrieved_top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_top_k.intersection(relevant_set))

    return hits / len(relevant_set)


def calculate_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Calculate Precision@K: proportion of top-K retrieved chunks that are relevant.

    Precision@K = |Retrieved@K ∩ Relevant| / K
    """
    if not relevant_ids or k <= 0:
        return 0.0

    retrieved_top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_top_k.intersection(relevant_set))

    return hits / k


def calculate_mrr_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """Calculate Mean Reciprocal Rank (MRR@K) for the first relevant chunk in top-K results.

    MRR@K = 1 / rank of first relevant result if rank <= K else 0.0
    """
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0

    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0
