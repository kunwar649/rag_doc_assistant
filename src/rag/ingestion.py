"""Load -> chunk -> embed -> store, shared by the CLI (scripts/ingest.py) and the API (/upload)."""
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_pages
from .config import Settings
from .embeddings import get_embedder
from .loaders import SUPPORTED_EXTENSIONS, load_file
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    files: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    pages: int = 0
    chunks: int = 0


def find_documents(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def ingest_paths(paths: list[Path], settings: Settings, store: VectorStore | None = None) -> IngestReport:
    store = store or VectorStore(settings.chroma_dir, settings.collection_name)
    report = IngestReport()
    pages = []
    for path in paths:
        try:
            file_pages = load_file(path)
        except Exception:  # one bad file shouldn't stop the whole ingest
            logger.exception("Failed to load %s", path.name)
            report.failed.append(path.name)
            continue
        if not file_pages:
            logger.warning("No text extracted from %s", path.name)
            report.failed.append(path.name)
            continue
        logger.info("Loaded %s: %d page(s)", path.name, len(file_pages))
        report.files.append(path.name)
        pages.extend(file_pages)

    if not pages:
        return report

    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    # Clear each file's old chunks first so an edited document leaves nothing stale behind.
    for source in report.files:
        store.delete_source(source)
    embeddings = get_embedder(settings.embedding_model).embed_documents([c.text for c in chunks])
    store.upsert(chunks, embeddings)

    report.pages, report.chunks = len(pages), len(chunks)
    return report
