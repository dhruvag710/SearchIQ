from app.retrieval.reranker import RerankedResult


class PromptBuilder:
    """Build a retrieval-augmented generation prompt from reranked context chunks."""

    def build(self, question: str, results: list[RerankedResult]) -> str:
        """Return a complete RAG prompt for the given question and retrieved passages."""
        instructions = (
            "You are a helpful assistant. Answer the question using only the context "
            "provided below.\n"
            "- Do not use outside knowledge or invent facts.\n"
            "- If the context does not contain enough information to answer, say so clearly.\n"
            "- When relevant, cite the page number(s) from the context that support your answer."
        )
        context = self._format_context(results)
        trimmed_question = question.strip()

        return (
            f"{instructions}\n\n"
            f"Context:\n"
            f"{context}\n\n"
            f"Question:\n"
            f"{trimmed_question}\n\n"
            f"Answer:"
        )

    def _format_context(self, results: list[RerankedResult]) -> str:
        """Format retrieved passages with preserved page numbers."""
        if not results:
            return "No relevant context was retrieved."

        passages = [
            f"[Page {result.page_number}]\n{result.text.strip()}"
            for result in results
        ]
        return "\n\n".join(passages)
