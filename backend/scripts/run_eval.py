import argparse
import json
import sys
from pathlib import Path

# Add backend directory to sys.path if not present
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database.session import SessionLocal
from app.evaluation.dataset import GoldenDataset
from app.evaluation.evaluator import RetrievalEvaluator
from app.evaluation.reporter import EvaluationReporter


def main() -> None:
    """Run SearchIQ retrieval evaluation benchmark from CLI."""
    parser = argparse.ArgumentParser(
        description="Run offline evaluation benchmark for SearchIQ retrieval strategies."
    )
    default_dataset_path = PROJECT_ROOT / "data" / "eval" / "golden_dataset.json"
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_path,
        help=f"Path to golden dataset JSON file (default: {default_dataset_path})",
    )
    args = parser.parse_args()

    dataset_path: Path = args.dataset.resolve()
    if not dataset_path.exists():
        print(f"Error: Golden dataset file not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    try:
        raw_json = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset = GoldenDataset.model_validate(raw_json)
    except Exception as exc:
        print(f"Error: Failed to parse golden dataset at {dataset_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not dataset.test_cases:
        print(
            f"Error: Golden dataset at {dataset_path} contains 0 test cases.\n"
            f"Please populate 'test_cases' with annotated evaluation queries before running benchmarks.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded golden dataset '{dataset.dataset_name}' with {len(dataset.test_cases)} test case(s).")
    print("Running offline retrieval evaluation across 4 strategies...")

    db = SessionLocal()
    try:
        evaluator = RetrievalEvaluator()
        report = evaluator.evaluate_dataset(db, dataset)
        print("\n" + EvaluationReporter.format_console(report))
    finally:
        db.close()


if __name__ == "__main__":
    main()
