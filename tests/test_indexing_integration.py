from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import uuid

import fitz

from app.chunking.chunker import chunk_document
from app.ingestion.parser import parse_pdf
from app.services.indexing_service import index_document


def _save_pdf_to_temp(doc: fitz.Document) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    doc.save(str(tmp_path))
    return tmp_path


class TestIndexingIntegration(unittest.TestCase):
    """End-to-end integration tests for table-aware document ingestion and indexing."""

    def test_parse_to_chunk_pipeline_with_table(self) -> None:
        """Verify the full pipeline from PDF parsing through chunking for a tabular document."""
        doc = fitz.open()
        page = doc.new_page()

        # Insert intro prose
        page.insert_text((50, 40), "Executive Summary of Model Performance")

        # Insert table
        page.draw_rect(fitz.Rect(40, 60, 320, 130))
        page.draw_line((40, 85), (320, 85))
        page.draw_line((140, 60), (140, 130))
        page.draw_line((230, 60), (230, 130))

        page.insert_text((50, 78), "Model")
        page.insert_text((150, 78), "Recall@5")
        page.insert_text((240, 78), "MRR@5")

        page.insert_text((50, 105), "Dense BGE")
        page.insert_text((150, 105), "0.70")
        page.insert_text((240, 105), "0.29")

        # Insert outro prose
        page.insert_text((50, 150), "Dense BGE establishes the strongest baseline.")

        tmp_path = _save_pdf_to_temp(doc)
        try:
            parsed = parse_pdf(tmp_path)
            self.assertEqual(parsed.total_pages, 1)

            chunks = chunk_document(parsed)
            self.assertGreaterEqual(len(chunks), 1)

            # Combined chunk text must contain intro, markdown table, and outro
            all_text = "\n".join(c.text for c in chunks)
            self.assertIn("Executive Summary of Model Performance", all_text)
            self.assertIn("Model", all_text)
            self.assertIn("Recall@5", all_text)
            self.assertIn("MRR@5", all_text)
            self.assertIn("Dense BGE", all_text)
            self.assertIn("0.70", all_text)
            self.assertIn("0.29", all_text)
            self.assertIn("Dense BGE establishes the strongest baseline.", all_text)

            # Check chunk metadata fields
            for c in chunks:
                self.assertEqual(c.page_number, 1)
                self.assertIsNotNone(c.chunk_id)
                self.assertIsNotNone(c.document_id)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_indexing_service_persists_table_chunks(self) -> None:
        """Verify indexing_service processes a table PDF, creates records, and commits."""
        doc = fitz.open()
        page = doc.new_page()

        page.draw_rect(fitz.Rect(40, 50, 300, 110))
        page.draw_line((40, 75), (300, 75))
        page.draw_line((130, 50), (130, 110))
        page.insert_text((50, 68), "Framework")
        page.insert_text((140, 68), "Score")
        page.insert_text((50, 95), "SearchIQ")
        page.insert_text((140, 95), "0.98")

        tmp_path = _save_pdf_to_temp(doc)
        try:
            mock_db = MagicMock()
            doc_id = uuid.uuid4()

            # Mock embed_chunks to decouple from model download/GPU during unit tests
            with patch("app.services.indexing_service.embed_chunks") as mock_embed:
                mock_embed.side_effect = lambda chunks: [(c, [0.05] * 384) for c in chunks]

                summary = index_document(tmp_path, mock_db, document_id=doc_id)

                self.assertEqual(summary.document_id, str(doc_id))
                self.assertEqual(summary.total_pages, 1)
                self.assertGreaterEqual(summary.total_chunks, 1)
                self.assertEqual(summary.status, "indexed")

                # Verify db calls
                self.assertTrue(mock_db.commit.called)
                self.assertGreaterEqual(mock_db.add.call_count, 2)  # 1 document + >=1 chunks

                # Verify chunk record content contains table text
                added_chunks = [
                    call[0][0]
                    for call in mock_db.add.call_args_list
                    if hasattr(call[0][0], "chunk_index")
                ]
                self.assertGreaterEqual(len(added_chunks), 1)
                chunk_texts = "\n".join(ch.text for ch in added_chunks)
                self.assertIn("Framework", chunk_texts)
                self.assertIn("SearchIQ", chunk_texts)
                self.assertIn("0.98", chunk_texts)
        finally:
            tmp_path.unlink(missing_ok=True)
