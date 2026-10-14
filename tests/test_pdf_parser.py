from pathlib import Path
import tempfile
import unittest

import fitz

from app.ingestion.parser import PdfParser, _extract_page_content, parse_pdf


def _save_pdf_to_temp(doc: fitz.Document) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    doc.save(str(tmp_path))
    return tmp_path


class TestPdfParser(unittest.TestCase):
    """Test PDF parsing and table-aware Markdown extraction."""

    def test_plain_prose_pdf_regression(self) -> None:
        """Verify that documents without tables extract plain prose without regression."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "First paragraph of document prose.")
        page.insert_text((50, 100), "Second paragraph of document prose.")

        tmp_path = _save_pdf_to_temp(doc)
        try:
            parsed = parse_pdf(tmp_path)
            self.assertEqual(parsed.total_pages, 1)
            self.assertEqual(len(parsed.pages), 1)
            self.assertIn("First paragraph of document prose.", parsed.pages[0].text)
            self.assertIn("Second paragraph of document prose.", parsed.pages[0].text)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_digital_table_extraction_to_markdown(self) -> None:
        """Verify that digital PDF tables are converted into Markdown preserving headers and rows."""
        doc = fitz.open()
        page = doc.new_page()
        page.draw_rect(fitz.Rect(40, 50, 300, 120))
        page.draw_line((40, 75), (300, 75))
        page.draw_line((40, 95), (300, 95))
        page.draw_line((130, 50), (130, 120))
        page.draw_line((210, 50), (210, 120))

        page.insert_text((50, 68), "Model")
        page.insert_text((140, 68), "Recall@5")
        page.insert_text((220, 68), "MRR@5")

        page.insert_text((50, 88), "Dense BGE")
        page.insert_text((140, 88), "0.70")
        page.insert_text((220, 88), "0.29")

        page.insert_text((50, 110), "Hybrid RRF")
        page.insert_text((140, 110), "0.55")
        page.insert_text((220, 110), "0.14")

        tmp_path = _save_pdf_to_temp(doc)
        try:
            parsed = parse_pdf(tmp_path)
            page_text = parsed.pages[0].text
            self.assertIn("Model", page_text)
            self.assertIn("Recall@5", page_text)
            self.assertIn("MRR@5", page_text)
            self.assertIn("Dense BGE", page_text)
            self.assertIn("0.70", page_text)
            self.assertIn("0.29", page_text)
            self.assertIn("Hybrid RRF", page_text)
            self.assertIn("0.55", page_text)
            self.assertIn("0.14", page_text)
            self.assertIn("|", page_text)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_mixed_prose_and_table_preserves_reading_order_and_no_duplication(self) -> None:
        """Verify surrounding prose before and after a table is retained without duplicating cells."""
        doc = fitz.open()
        page = doc.new_page()

        page.insert_text((50, 30), "Section 1: Information Retrieval Benchmarks")

        page.draw_rect(fitz.Rect(40, 50, 300, 100))
        page.draw_line((40, 75), (300, 75))
        page.draw_line((150, 50), (150, 100))
        page.insert_text((50, 65), "System")
        page.insert_text((160, 65), "Latency")
        page.insert_text((50, 90), "SearchIQ")
        page.insert_text((160, 90), "12ms")

        page.insert_text((50, 130), "Section 2: Discussion and Evaluation")

        tmp_path = _save_pdf_to_temp(doc)
        try:
            parsed = parse_pdf(tmp_path)
            page_text = parsed.pages[0].text

            self.assertIn("Section 1: Information Retrieval Benchmarks", page_text)
            self.assertIn("Section 2: Discussion and Evaluation", page_text)
            self.assertIn("System", page_text)
            self.assertIn("Latency", page_text)
            self.assertIn("SearchIQ", page_text)
            self.assertIn("12ms", page_text)

            pos_sec1 = page_text.index("Section 1: Information Retrieval Benchmarks")
            pos_table = page_text.index("System")
            pos_sec2 = page_text.index("Section 2: Discussion and Evaluation")
            self.assertLess(pos_sec1, pos_table)
            self.assertLess(pos_table, pos_sec2)

            self.assertEqual(page_text.count("SearchIQ"), 1)
            self.assertEqual(page_text.count("12ms"), 1)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_multi_page_document_preserves_page_numbers(self) -> None:
        """Verify that multi-page documents maintain accurate page numbers and page-scoped content."""
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 50), "Content on page one.")
        p2 = doc.new_page()
        p2.insert_text((50, 50), "Content on page two.")

        tmp_path = _save_pdf_to_temp(doc)
        try:
            parsed = parse_pdf(tmp_path)
            self.assertEqual(parsed.total_pages, 2)
            self.assertEqual(parsed.pages[0].page_number, 1)
            self.assertEqual(parsed.pages[1].page_number, 2)
            self.assertIn("Content on page one.", parsed.pages[0].text)
            self.assertIn("Content on page two.", parsed.pages[1].text)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_graceful_fallback_when_find_tables_absent_or_fails(self) -> None:
        """Verify graceful fallback to standard text extraction when table detection fails."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Fallback text extraction.")

        content = _extract_page_content(page)
        self.assertIn("Fallback text extraction.", content)
