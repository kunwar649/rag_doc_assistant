"""Load -> chunk -> embed -> store.

Usage:
    python scripts/ingest.py               # ingest everything in DOCS_DIR
    python scripts/ingest.py --reset       # wipe the collection first
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings
from rag.ingestion import find_documents, ingest_paths
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, help="override DOCS_DIR")
    parser.add_argument("--reset", action="store_true", help="delete existing chunks first")
    args = parser.parse_args()

    s = get_settings()
    docs_dir = args.docs_dir or s.docs_dir
    t0 = time.perf_counter()

    paths = find_documents(docs_dir)
    if not paths:
        log.error("No documents found in %s. Add PDF/DOCX/TXT/MD files and retry.", docs_dir)
        sys.exit(1)

    store = VectorStore(s.chroma_dir, s.collection_name)
    if args.reset:
        store.client.delete_collection(s.collection_name)
        store = VectorStore(s.chroma_dir, s.collection_name)
        log.info("Collection reset")

    log.info("Embedding with %s (first run downloads the model)...", s.embedding_model)
    report = ingest_paths(paths, s, store)
    if report.failed:
        log.warning("Skipped: %s", ", ".join(report.failed))
    log.info(
        "%d file(s), %d pages -> %d chunks (size=%d, overlap=%d)",
        len(report.files), report.pages, report.chunks, s.chunk_size, s.chunk_overlap,
    )
    log.info("Done in %.1fs. Collection now holds %d chunks.", time.perf_counter() - t0, store.count())


if __name__ == "__main__":
    main()
