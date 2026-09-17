import sys
from pathlib import Path

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.prompting.prompt_builder import PromptBuilder
from app.retrieval.reranker import RerankedResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reranked(
    *,
    chunk_id: str = "chunk-1",
    page_number: int = 1,
    text: str = "Some passage.",
    reranker_score: float = 0.9,
) -> RerankedResult:
    return RerankedResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        page_number=page_number,
        chunk_index=0,
        text=text,
        hybrid_score=0.0,
        reranker_score=reranker_score,
    )


# ---------------------------------------------------------------------------
# Tests — text-only context (backward compat)
# ---------------------------------------------------------------------------


class TestTextOnlyContext:

    def test_page_label_format(self) -> None:
        result = _make_reranked(page_number=5, text="Revenue was up 20%.")
        builder = PromptBuilder()
        prompt = builder.build("revenue?", [result])

        assert "[Page 5]" in prompt
        assert "Revenue was up 20%." in prompt
        assert "[Image" not in prompt

    def test_no_results_fallback(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build("anything?", [])

        assert "No relevant context was retrieved." in prompt


# ---------------------------------------------------------------------------
# Tests — visual context labeling
# ---------------------------------------------------------------------------


class TestVisualContextLabeling:

    def test_visual_result_labeled_as_image(self) -> None:
        result = _make_reranked(
            chunk_id="vis-1",
            page_number=3,
            text="Bar chart showing Q3 earnings.",
        )
        builder = PromptBuilder()
        prompt = builder.build("earnings?", [result], visual_ids={"vis-1"})

        assert "[Image, Page 3]" in prompt
        assert "Bar chart showing Q3 earnings." in prompt

    def test_text_result_not_labeled_as_image(self) -> None:
        result = _make_reranked(chunk_id="text-1", page_number=2, text="Text passage.")
        builder = PromptBuilder()
        prompt = builder.build("question?", [result], visual_ids={"vis-1"})

        assert "[Page 2]" in prompt
        assert "[Image" not in prompt


# ---------------------------------------------------------------------------
# Tests — mixed text + visual context
# ---------------------------------------------------------------------------


class TestMixedContext:

    def test_mixed_labels_appear_correctly(self) -> None:
        text_result = _make_reranked(
            chunk_id="t-1", page_number=1, text="Text about policy."
        )
        visual_result = _make_reranked(
            chunk_id="v-1", page_number=4, text="Flowchart of approval process."
        )

        builder = PromptBuilder()
        prompt = builder.build(
            "approval process?",
            [text_result, visual_result],
            visual_ids={"v-1"},
        )

        assert "[Page 1]" in prompt
        assert "Text about policy." in prompt

        assert "[Image, Page 4]" in prompt
        assert "Flowchart of approval process." in prompt

    def test_visual_ids_none_treated_as_all_text(self) -> None:
        result = _make_reranked(chunk_id="any-id", page_number=7, text="Content.")
        builder = PromptBuilder()
        prompt = builder.build("q?", [result], visual_ids=None)

        assert "[Page 7]" in prompt
        assert "[Image" not in prompt

    def test_visual_ids_empty_treated_as_all_text(self) -> None:
        result = _make_reranked(chunk_id="any-id", page_number=7, text="Content.")
        builder = PromptBuilder()
        prompt = builder.build("q?", [result], visual_ids=set())

        assert "[Page 7]" in prompt
        assert "[Image" not in prompt
