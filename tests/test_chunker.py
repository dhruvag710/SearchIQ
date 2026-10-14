import unittest

from app.chunking.chunker import (
    TARGET_CHUNK_SIZE,
    _is_markdown_table,
    _split_markdown_table,
    chunk_document,
)
from app.ingestion.models import Document, Page


class TestChunker(unittest.TestCase):
    """Test recursive character chunker with table-aware enhancements."""

    def test_is_markdown_table(self) -> None:
        """Verify markdown table detection for valid and invalid table strings."""
        valid_table = (
            "| Model | Recall@5 | MRR@5 |\n"
            "|---|---|---|\n"
            "| Dense BGE | 0.70 | 0.29 |"
        )
        self.assertTrue(_is_markdown_table(valid_table))

        valid_table_colons = (
            "| Model | Score |\n"
            "|:---|:---:|\n"
            "| Dense BGE | 0.70 |"
        )
        self.assertTrue(_is_markdown_table(valid_table_colons))

        # Plain prose
        prose = "This is a regular paragraph of prose text."
        self.assertFalse(_is_markdown_table(prose))

        # Single line starting with pipe
        single_pipe = "| Just a line with pipes |"
        self.assertFalse(_is_markdown_table(single_pipe))

        # Text with pipes but invalid second separator line
        invalid_separator = (
            "| Col1 | Col2 |\n"
            "| not | dashes |"
        )
        self.assertFalse(_is_markdown_table(invalid_separator))

    def test_small_markdown_table_remains_intact(self) -> None:
        """Verify that a small markdown table stays within a single chunk."""
        table = (
            "| Model | Recall@5 | MRR@5 |\n"
            "|---|---|---|\n"
            "| Dense BGE | 0.70 | 0.29 |\n"
            "| Hybrid RRF | 0.55 | 0.14 |"
        )
        doc = Document(
            document_title="Test Doc",
            total_pages=1,
            pages=[Page(page_number=1, text=table)],
        )
        chunks = chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Dense BGE", chunks[0].text)
        self.assertIn("Hybrid RRF", chunks[0].text)
        self.assertIn("| Model | Recall@5 | MRR@5 |", chunks[0].text)

    def test_oversized_markdown_table_splits_by_rows_and_repeats_headers(self) -> None:
        """Verify that an oversized table splits by rows and duplicates header block in every chunk."""
        header_block = "| Model | Recall@5 | MRR@5 |\n|---|---|---|"
        rows = [f"| Model-{i:02d} | 0.{i:02d} | 0.{i:02d} |" for i in range(40)]
        oversized_table = header_block + "\n" + "\n".join(rows)

        doc = Document(
            document_title="Large Table Doc",
            total_pages=1,
            pages=[Page(page_number=1, text=oversized_table)],
        )
        chunks = chunk_document(doc)

        self.assertGreater(len(chunks), 1)

        # Every chunk must contain the header row and separator line
        for idx, chunk in enumerate(chunks):
            self.assertIn(
                "| Model | Recall@5 | MRR@5 |",
                chunk.text,
                f"Chunk {idx} missing header row",
            )
            self.assertIn(
                "|---|---|---|",
                chunk.text,
                f"Chunk {idx} missing separator line",
            )
            # Verify no row is cut in half (each line must start and end with pipe)
            for line in chunk.text.splitlines():
                stripped = line.strip()
                if stripped:
                    self.assertTrue(
                        stripped.startswith("|") and stripped.endswith("|"),
                        f"Line not properly bounded: {stripped}",
                    )

        # Verify all 40 models appear across the chunks
        combined_text = "\n".join(c.text for c in chunks)
        for i in range(40):
            self.assertIn(f"Model-{i:02d}", combined_text)

    def test_mixed_prose_and_table_chunk_boundaries(self) -> None:
        """Verify that mixed pages with prose and tables chunk cleanly without corrupted overlap."""
        header = "| Metric | Value |\n|---|---|"
        table_rows = [f"| Row-{i:02d} | Val-{i:02d} |" for i in range(35)]
        table = header + "\n" + "\n".join(table_rows)

        prose_intro = "Introductory prose section before the tabular benchmark data. " * 8
        prose_outro = "Concluding analytical remarks following the experimental results. " * 8

        page_text = f"{prose_intro}\n\n{table}\n\n{prose_outro}"
        doc = Document(
            document_title="Mixed Page Doc",
            total_pages=1,
            pages=[Page(page_number=1, text=page_text)],
        )
        chunks = chunk_document(doc)

        # First chunk should have intro prose
        self.assertIn("Introductory prose section", chunks[0].text)

        # Middle chunk(s) should have table with header
        table_chunks = [c for c in chunks if "| Metric | Value |" in c.text]
        self.assertGreater(len(table_chunks), 0)
        for tc in table_chunks:
            self.assertTrue(tc.text.startswith("| Metric | Value |"))

        # Last chunk should have conclusion prose
        self.assertIn("Concluding analytical remarks", chunks[-1].text)

    def test_ordinary_prose_chunking_regression(self) -> None:
        """Verify that ordinary prose documents preserve chunk size and overlap behavior."""
        paragraph_1 = "This is the first paragraph. " * 15
        paragraph_2 = "This is the second paragraph with more content. " * 15
        full_text = f"{paragraph_1}\n\n{paragraph_2}"

        doc = Document(
            document_title="Prose Doc",
            total_pages=1,
            pages=[Page(page_number=1, text=full_text)],
        )
        chunks = chunk_document(doc)

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.char_count, TARGET_CHUNK_SIZE)

        # Chunks must retain sequential chunk_index and page_number
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, idx)
            self.assertEqual(chunk.page_number, 1)
            self.assertEqual(chunk.document_title, "Prose Doc")
