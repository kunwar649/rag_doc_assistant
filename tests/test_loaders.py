import pytest
from docx import Document

from rag.loaders import load_directory, load_file


def test_load_txt(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Line   one\n\n\nLine two   ")
    pages = load_file(f)
    assert len(pages) == 1
    assert pages[0].text == "Line one\nLine two"
    assert pages[0].page is None


def test_load_docx_includes_tables(tmp_path):
    f = tmp_path / "spec.docx"
    doc = Document()
    doc.add_paragraph("Intro paragraph")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Plan"
    table.rows[0].cells[1].text = "Free"
    doc.save(f)
    text = load_file(f)[0].text
    assert "Intro paragraph" in text
    assert "Plan | Free" in text


def test_unsupported_extension(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_file(f)


def test_directory_skips_unsupported_and_empty(tmp_path):
    (tmp_path / "a.txt").write_text("content")
    (tmp_path / "empty.txt").write_text("   ")
    (tmp_path / "skip.csv").write_text("a,b")
    pages = load_directory(tmp_path)
    assert [p.source for p in pages] == ["a.txt"]
