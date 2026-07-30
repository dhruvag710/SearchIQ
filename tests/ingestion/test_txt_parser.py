import sys
from pathlib import Path

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ingestion.models import Document, Page
from app.ingestion.txt_parser import TxtParser, parse_txt


def test_txt_parser(tmp_path: Path):
    sample_file = tmp_path / "sample_doc.txt"
    sample_file.write_text("  Hello,  World!  \n\n\nThis is a   test file.  ", encoding="utf-8")

    parser = TxtParser()
    doc = parser.parse(sample_file)

    assert isinstance(doc, Document)
    assert doc.document_title == "sample_doc"
    assert doc.total_pages == 1
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert doc.pages[0].text == "Hello, World! \n\nThis is a test file."


def test_parse_txt_helper(tmp_path: Path):
    sample_file = tmp_path / "another_doc.txt"
    sample_file.write_text("Testing parse_txt helper.", encoding="utf-8")

    doc = parse_txt(sample_file)

    assert doc.document_title == "another_doc"
    assert doc.total_pages == 1
    assert doc.pages[0].text == "Testing parse_txt helper."


if __name__ == "__main__":
    import unittest
    import tempfile

    class TestTxtParserUnittest(unittest.TestCase):
        def test_txt_parser(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "sample_doc.txt"
                path.write_text("  Hello,  World!  \n\n\nThis is a   test file.  ", encoding="utf-8")
                
                parser = TxtParser()
                doc = parser.parse(path)
                self.assertEqual(doc.document_title, "sample_doc")
                self.assertEqual(doc.total_pages, 1)
                self.assertEqual(doc.pages[0].page_number, 1)
                self.assertEqual(doc.pages[0].text, "Hello, World! \n\nThis is a test file.")

        def test_parse_txt_helper(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "another_doc.txt"
                path.write_text("Testing parse_txt helper.", encoding="utf-8")
                
                doc = parse_txt(path)
                self.assertEqual(doc.document_title, "another_doc")
                self.assertEqual(doc.total_pages, 1)
                self.assertEqual(doc.pages[0].text, "Testing parse_txt helper.")

    unittest.main()
