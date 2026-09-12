import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.evaluation.dataset import EvaluationTestCase, GoldenDataset
from app.evaluation.metrics import (
    EvaluationMetrics,
    calculate_mrr_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_search import VectorSearchService


@dataclass(frozen=True)
class StrategyResult:
    """Evaluation summary for a single retrieval strategy across all test cases."""

    strategy_name: str
    description: str
    metrics: EvaluationMetrics
    avg_latency_ms: float
    total_test_cases: int


@dataclass(frozen=True)
class EvaluationReport:
    """Complete evaluation report containing benchmark results across strategies."""

    dataset_name: str
    total_test_cases: int
    strategies: list[StrategyResult]


def extract_chunk_ids(results: list[Any]) -> list[str]:
    """Normalize any retrieval result item (ORM Chunk or result dataclass) to string chunk ID."""
    chunk_ids: list[str] = []
    for item in results:
        cid = str(getattr(item, "id", getattr(item, "chunk_id", "")))
        if cid:
            chunk_ids.append(cid)
    return chunk_ids


class RetrievalEvaluator:
    """Evaluate SearchIQ retrieval strategies against a golden dataset."""

    def __init__(
        self,
        vector_search: VectorSearchService | None = None,
        bm25_retriever: BM25Retriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.vector_search = vector_search or VectorSearchService()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            dense_retriever=self.vector_search,
            bm25_retriever=self.bm25_retriever,
        )
        self.reranker = reranker or CrossEncoderReranker()

    def evaluate_dataset(
        self,
        db: Session,
        dataset: GoldenDataset,
    ) -> EvaluationReport:
        """Run all test cases through all 4 retrieval strategies and compute metrics."""
        if not dataset.test_cases:
            raise ValueError("Evaluation dataset contains zero test cases.")

        # Ensure BM25 index is built if using default retriever
        self.bm25_retriever.build_index(db)

        # Strategy accumulators: strategy_key -> list of (metrics_dict, latency_ms)
        strategy_data: dict[str, dict[str, Any]] = {
            "dense": {
                "name": "A. Dense Retrieval",
                "desc": "BGE vector similarity search (pgvector)",
                "recalls_5": [],
                "recalls_25": [],
                "mrrs_5": [],
                "precisions_5": [],
                "latencies_ms": [],
            },
            "bm25": {
                "name": "B. BM25 Retrieval",
                "desc": "In-memory lexical BM25 search",
                "recalls_5": [],
                "recalls_25": [],
                "mrrs_5": [],
                "precisions_5": [],
                "latencies_ms": [],
            },
            "hybrid": {
                "name": "C. Hybrid (RRF)",
                "desc": "Dense + BM25 merged via Reciprocal Rank Fusion",
                "recalls_5": [],
                "recalls_25": [],
                "mrrs_5": [],
                "precisions_5": [],
                "latencies_ms": [],
            },
            "reranked": {
                "name": "D. Hybrid + Reranker",
                "desc": "Hybrid RRF (top 25) rescored by Cross-Encoder (top 5)",
                "recalls_5": [],
                "recalls_25": [],
                "mrrs_5": [],
                "precisions_5": [],
                "latencies_ms": [],
            },
        }

        for test_case in dataset.test_cases:
            relevant_ids = test_case.relevant_chunk_ids
            question = test_case.question

            # Strategy A: Dense Retrieval (top 25)
            t0 = time.perf_counter()
            dense_res = self.vector_search.search(db, question, top_k=25)
            t_dense = (time.perf_counter() - t0) * 1000
            dense_ids = extract_chunk_ids(dense_res)

            strategy_data["dense"]["recalls_5"].append(calculate_recall_at_k(dense_ids, relevant_ids, k=5))
            strategy_data["dense"]["recalls_25"].append(calculate_recall_at_k(dense_ids, relevant_ids, k=25))
            strategy_data["dense"]["mrrs_5"].append(calculate_mrr_at_k(dense_ids, relevant_ids, k=5))
            strategy_data["dense"]["precisions_5"].append(calculate_precision_at_k(dense_ids, relevant_ids, k=5))
            strategy_data["dense"]["latencies_ms"].append(t_dense)

            # Strategy B: BM25 Retrieval (top 25)
            t0 = time.perf_counter()
            bm25_res = self.bm25_retriever.retrieve(question, top_k=25)
            t_bm25 = (time.perf_counter() - t0) * 1000
            bm25_ids = extract_chunk_ids(bm25_res)

            strategy_data["bm25"]["recalls_5"].append(calculate_recall_at_k(bm25_ids, relevant_ids, k=5))
            strategy_data["bm25"]["recalls_25"].append(calculate_recall_at_k(bm25_ids, relevant_ids, k=25))
            strategy_data["bm25"]["mrrs_5"].append(calculate_mrr_at_k(bm25_ids, relevant_ids, k=5))
            strategy_data["bm25"]["precisions_5"].append(calculate_precision_at_k(bm25_ids, relevant_ids, k=5))
            strategy_data["bm25"]["latencies_ms"].append(t_bm25)

            # Strategy C: Hybrid RRF (top 25)
            t0 = time.perf_counter()
            hybrid_res = self.hybrid_retriever.retrieve(db, question, top_k=25)
            t_hybrid = (time.perf_counter() - t0) * 1000
            hybrid_ids = extract_chunk_ids(hybrid_res)

            strategy_data["hybrid"]["recalls_5"].append(calculate_recall_at_k(hybrid_ids, relevant_ids, k=5))
            strategy_data["hybrid"]["recalls_25"].append(calculate_recall_at_k(hybrid_ids, relevant_ids, k=25))
            strategy_data["hybrid"]["mrrs_5"].append(calculate_mrr_at_k(hybrid_ids, relevant_ids, k=5))
            strategy_data["hybrid"]["precisions_5"].append(calculate_precision_at_k(hybrid_ids, relevant_ids, k=5))
            strategy_data["hybrid"]["latencies_ms"].append(t_hybrid)

            # Strategy D: Hybrid + Reranker (25 candidates -> top 5)
            t0 = time.perf_counter()
            reranked_res = self.reranker.rerank(question, hybrid_res, top_k=5)
            t_reranked = t_hybrid + ((time.perf_counter() - t0) * 1000)
            reranked_ids = extract_chunk_ids(reranked_res)

            strategy_data["reranked"]["recalls_5"].append(calculate_recall_at_k(reranked_ids, relevant_ids, k=5))
            strategy_data["reranked"]["mrrs_5"].append(calculate_mrr_at_k(reranked_ids, relevant_ids, k=5))
            strategy_data["reranked"]["precisions_5"].append(calculate_precision_at_k(reranked_ids, relevant_ids, k=5))
            strategy_data["reranked"]["latencies_ms"].append(t_reranked)

        total_cases = len(dataset.test_cases)
        results: list[StrategyResult] = []

        for key in ["dense", "bm25", "hybrid", "reranked"]:
            data = strategy_data[key]
            r5 = sum(data["recalls_5"]) / total_cases
            r25 = (sum(data["recalls_25"]) / total_cases) if data["recalls_25"] else None
            mrr5 = sum(data["mrrs_5"]) / total_cases
            p5 = sum(data["precisions_5"]) / total_cases
            avg_lat = sum(data["latencies_ms"]) / total_cases

            results.append(
                StrategyResult(
                    strategy_name=data["name"],
                    description=data["desc"],
                    metrics=EvaluationMetrics(
                        recall_at_5=r5,
                        recall_at_25=r25,
                        mrr_at_5=mrr5,
                        precision_at_5=p5,
                    ),
                    avg_latency_ms=avg_lat,
                    total_test_cases=total_cases,
                )
            )

        return EvaluationReport(
            dataset_name=dataset.dataset_name,
            total_test_cases=total_cases,
            strategies=results,
        )
