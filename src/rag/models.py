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
    score: float  # cosine similarity, higher = more relevant

    @property
    def citation(self) -> str:
        page = self.metadata.get("page")
        src = self.metadata.get("source", "unknown")
        return f"{src} (p. {page})" if page else src
