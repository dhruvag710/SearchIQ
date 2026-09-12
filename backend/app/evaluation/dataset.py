from pydantic import BaseModel, Field


class EvaluationTestCase(BaseModel):
    """Ground truth evaluation test case mapping a question to relevant chunk IDs."""

    id: str = Field(description="Unique identifier for the test case")
    question: str = Field(description="Natural language query string")
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="List of ground-truth relevant chunk UUID strings",
    )
    query_type: str = Field(
        default="general",
        description="Categorization of the test query (e.g. keyword, semantic, complex)",
    )
    notes: str | None = Field(
        default=None,
        description="Optional human-readable notes or description for the test case",
    )


class GoldenDataset(BaseModel):
    """Collection of evaluation test cases for benchmarking retrieval strategies."""

    dataset_name: str = Field(description="Name of the golden dataset")
    description: str = Field(description="Summary of the dataset purpose and scope")
    test_cases: list[EvaluationTestCase] = Field(
        default_factory=list,
        description="List of evaluation test cases",
    )
