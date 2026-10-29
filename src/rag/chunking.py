"""Split pages into overlapping chunks with stable IDs and metadata."""
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Chunk, Page


def _chunk_id(source: str, page: int | None, index: int, text: str) -> str:
    # Deterministic: re-ingesting the same file produces the same IDs,
    # so upserts overwrite instead of duplicating.
    raw = f"{source}|{page}|{index}|{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def chunk_pages(pages: list[Page], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Tries paragraph -> line -> sentence -> word boundaries, in that order,
    # so chunks break at natural points instead of mid-sentence.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(splitter.split_text(page.text)):
            metadata = {"source": page.source, "chunk_index": i}
            if page.page is not None:  # Chroma metadata can't hold None
                metadata["page"] = page.page
            chunks.append(Chunk(id=_chunk_id(page.source, page.page, i, text), text=text, metadata=metadata))
    return chunks
