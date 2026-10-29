import pytest

from rag.chunking import chunk_pages
from rag.models import Page


def _page(text, page=1):
    return Page(text=text, source="doc.pdf", page=page)


def test_short_page_is_one_chunk():
    chunks = chunk_pages([_page("Hello world.")], chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].metadata == {"source": "doc.pdf", "chunk_index": 0, "page": 1}


def test_long_page_splits_within_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    chunks = chunk_pages([_page(text)], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_ids_are_deterministic_and_unique():
    pages = [_page("alpha " * 200, 1), _page("beta " * 200, 2)]
    a = chunk_pages(pages, 200, 20)
    b = chunk_pages(pages, 200, 20)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_page_none_is_omitted_from_metadata():
    chunks = chunk_pages([Page(text="no pages here", source="notes.txt")], 100, 10)
    assert "page" not in chunks[0].metadata


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages([_page("x")], chunk_size=100, chunk_overlap=100)
