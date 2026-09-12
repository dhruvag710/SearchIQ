import sys
from pathlib import Path

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.evaluation.metrics import (
    calculate_mrr_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
)


def test_recall_perfect():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = ["c1", "c2"]
    assert calculate_recall_at_k(retrieved, relevant, k=5) == 1.0


def test_recall_partial():
    retrieved = ["c1", "x", "y", "z", "w"]
    relevant = ["c1", "c2"]
    # 1 hit out of 2 relevant = 0.5
    assert calculate_recall_at_k(retrieved, relevant, k=5) == 0.5


def test_recall_zero():
    retrieved = ["a", "b", "c"]
    relevant = ["x", "y"]
    assert calculate_recall_at_k(retrieved, relevant, k=5) == 0.0


def test_recall_empty_retrieved():
    retrieved = []
    relevant = ["c1", "c2"]
    assert calculate_recall_at_k(retrieved, relevant, k=5) == 0.0


def test_recall_empty_relevant():
    retrieved = ["c1", "c2"]
    relevant = []
    assert calculate_recall_at_k(retrieved, relevant, k=5) == 0.0


def test_recall_multiple_relevant_beyond_k():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = ["c1", "c2", "c5", "c6"]
    # Top 5 retrieved contains c1, c2, c5 (3 hits out of 4 relevant) = 0.75
    assert calculate_recall_at_k(retrieved, relevant, k=5) == 0.75


def test_precision_calculations():
    retrieved = ["c1", "c2", "x", "y", "z"]
    relevant = ["c1", "c2", "c3"]
    # 2 hits in top 5 = 2 / 5 = 0.4
    assert calculate_precision_at_k(retrieved, relevant, k=5) == 0.4


def test_precision_empty_retrieved():
    retrieved = []
    relevant = ["c1"]
    assert calculate_precision_at_k(retrieved, relevant, k=5) == 0.0


def test_mrr_first_rank():
    retrieved = ["c1", "x", "y"]
    relevant = ["c1", "c2"]
    assert calculate_mrr_at_k(retrieved, relevant, k=5) == 1.0  # 1 / 1


def test_mrr_later_rank():
    retrieved = ["x", "y", "c2", "z"]
    relevant = ["c1", "c2"]
    assert calculate_mrr_at_k(retrieved, relevant, k=5) == 1.0 / 3  # 1 / 3


def test_mrr_no_relevant():
    retrieved = ["x", "y", "z"]
    relevant = ["c1", "c2"]
    assert calculate_mrr_at_k(retrieved, relevant, k=5) == 0.0


def test_mrr_relevant_outside_k():
    retrieved = ["x", "y", "z", "w", "a", "c1"]
    relevant = ["c1"]
    # c1 is at rank 6, but k=5, so MRR@5 is 0.0
    assert calculate_mrr_at_k(retrieved, relevant, k=5) == 0.0


if __name__ == "__main__":
    import unittest

    class TestMetricsUnittest(unittest.TestCase):
        def test_recall_perfect(self):
            test_recall_perfect()

        def test_recall_partial(self):
            test_recall_partial()

        def test_recall_zero(self):
            test_recall_zero()

        def test_recall_empty_retrieved(self):
            test_recall_empty_retrieved()

        def test_recall_empty_relevant(self):
            test_recall_empty_relevant()

        def test_recall_multiple_relevant_beyond_k(self):
            test_recall_multiple_relevant_beyond_k()

        def test_precision_calculations(self):
            test_precision_calculations()

        def test_precision_empty_retrieved(self):
            test_precision_empty_retrieved()

        def test_mrr_first_rank(self):
            test_mrr_first_rank()

        def test_mrr_later_rank(self):
            test_mrr_later_rank()

        def test_mrr_no_relevant(self):
            test_mrr_no_relevant()

        def test_mrr_relevant_outside_k(self):
            test_mrr_relevant_outside_k()

    unittest.main()
