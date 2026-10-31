"""Plain data types shared across the pipeline."""
from dataclasses import dataclass, field


@dataclass
class Page:
    """One unit of extracted text (a PDF page, or a whole DOCX/TXT file)."""

    text: str
    source: str  # file name
    page: int | None = None  # 1-based page number; None for formats without pages


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    text: str
    metadata: dict
    # Meaning depends on the stage that produced it:
    #   vector  -> cosine similarity (higher = more similar)
    #   bm25    -> BM25 keyword score (unbounded, higher = more term overlap)
    #   hybrid  -> RRF fused score (small numbers, only the order matters)
    #   rerank  -> cross-encoder relevance logit (can be negative)
    score: float
    id: str = ""

    @property
    def citation(self) -> str:
        page = self.metadata.get("page")
        src = self.metadata.get("source", "unknown")
        return f"{src} (p. {page})" if page else src
