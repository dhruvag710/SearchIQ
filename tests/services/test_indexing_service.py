import sys
from pathlib import Path

import pytest

# Add backend directory to sys.path if not present
backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.indexing_service import _parse_document


def test_parse_document_txt(tmp_path: Path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello text file", encoding="utf-8")

    doc, file_type = _parse_document(txt_file)
    assert file_type == "txt"
    assert doc.document_title == "test"
    assert doc.pages[0].text == "Hello text file"


def test_parse_document_markdown(tmp_path: Path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello Markdown", encoding="utf-8")

    doc, file_type = _parse_document(md_file)
    assert file_type == "md"
    assert doc.document_title == "test"
    assert doc.pages[0].text == "# Hello Markdown"


def test_parse_document_unsupported(tmp_path: Path):
    unsupported_file = tmp_path / "test.xyz"
    unsupported_file.write_text("Unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type for indexing"):
        _parse_document(unsupported_file)


if __name__ == "__main__":
    import tempfile
    import unittest

    class TestIndexingServiceUnittest(unittest.TestCase):
        def test_parse_txt(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test.txt"
                path.write_text("Hello text file", encoding="utf-8")
                doc, file_type = _parse_document(path)
                self.assertEqual(file_type, "txt")
                self.assertEqual(doc.document_title, "test")
                self.assertEqual(doc.pages[0].text, "Hello text file")

        def test_parse_markdown(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test.md"
                path.write_text("# Hello Markdown", encoding="utf-8")
                doc, file_type = _parse_document(path)
                self.assertEqual(file_type, "md")
                self.assertEqual(doc.document_title, "test")
                self.assertEqual(doc.pages[0].text, "# Hello Markdown")

        def test_unsupported_extension(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test.xyz"
                path.write_text("Unsupported", encoding="utf-8")
                with self.assertRaises(ValueError):
                    _parse_document(path)

    unittest.main()
