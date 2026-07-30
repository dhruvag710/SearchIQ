import sys
from pathlib import Path

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ingestion.markdown_parser import MarkdownParser, parse_markdown
from app.ingestion.models import Document, Page


def test_markdown_parser(tmp_path: Path):
    sample_file = tmp_path / "README.md"
    sample_file.write_text("# Title\n\nHello,  Markdown!\n\n\nThis is a   test file.", encoding="utf-8")

    parser = MarkdownParser()
    doc = parser.parse(sample_file)

    assert isinstance(doc, Document)
    assert doc.document_title == "README"
    assert doc.total_pages == 1
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert doc.pages[0].text == "# Title\n\nHello, Markdown!\n\nThis is a test file."


def test_parse_markdown_helper(tmp_path: Path):
    sample_file = tmp_path / "doc.md"
    sample_file.write_text("Testing parse_markdown helper.", encoding="utf-8")

    doc = parse_markdown(sample_file)

    assert doc.document_title == "doc"
    assert doc.total_pages == 1
    assert doc.pages[0].text == "Testing parse_markdown helper."


if __name__ == "__main__":
    import tempfile
    import unittest

    class TestMarkdownParserUnittest(unittest.TestCase):
        def test_markdown_parser(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "README.md"
                path.write_text("# Title\n\nHello,  Markdown!\n\n\nThis is a   test file.", encoding="utf-8")

                parser = MarkdownParser()
                doc = parser.parse(path)
                self.assertEqual(doc.document_title, "README")
                self.assertEqual(doc.total_pages, 1)
                self.assertEqual(doc.pages[0].page_number, 1)
                self.assertEqual(doc.pages[0].text, "# Title\n\nHello, Markdown!\n\nThis is a test file.")

        def test_parse_markdown_helper(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "doc.md"
                path.write_text("Testing parse_markdown helper.", encoding="utf-8")

                doc = parse_markdown(path)
                self.assertEqual(doc.document_title, "doc")
                self.assertEqual(doc.total_pages, 1)
                self.assertEqual(doc.pages[0].text, "Testing parse_markdown helper.")

    unittest.main()
