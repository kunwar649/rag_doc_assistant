"""Day 1-2: load -> chunk -> embed -> store.

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

from rag.chunking import chunk_pages
from rag.config import get_settings
from rag.embeddings import get_embedder
from rag.loaders import load_directory
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

    pages = load_directory(docs_dir)
    if not pages:
        log.error("No documents found in %s. Add PDF/DOCX/TXT/MD files and retry.", docs_dir)
        sys.exit(1)

    chunks = chunk_pages(pages, s.chunk_size, s.chunk_overlap)
    log.info("%d pages -> %d chunks (size=%d, overlap=%d)", len(pages), len(chunks), s.chunk_size, s.chunk_overlap)

    store = VectorStore(s.chroma_dir, s.collection_name)
    if args.reset:
        store.client.delete_collection(s.collection_name)
        store = VectorStore(s.chroma_dir, s.collection_name)
        log.info("Collection reset")
    else:
        # Re-ingesting a file: clear its old chunks so edited docs don't leave stale text behind.
        for source in sorted({c.metadata["source"] for c in chunks}):
            store.delete_source(source)

    log.info("Embedding with %s (first run downloads the model)...", s.embedding_model)
    embedder = get_embedder(s.embedding_model)
    embeddings = embedder.embed_documents([c.text for c in chunks])

    store.upsert(chunks, embeddings)
    log.info("Done in %.1fs. Collection now holds %d chunks.", time.perf_counter() - t0, store.count())


if __name__ == "__main__":
    main()
