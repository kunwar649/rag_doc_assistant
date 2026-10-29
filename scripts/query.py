"""Day 2: test retrieval from the command line (no LLM yet).

Usage:
    python scripts/query.py "How do I reset my password?"
    python scripts/query.py "pricing" --top-k 3 --source handbook.pdf
    python scripts/query.py            # interactive mode
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings
from rag.embeddings import get_embedder
from rag.vector_store import VectorStore


def show(results) -> None:
    if not results:
        print("No results.")
    for rank, r in enumerate(results, start=1):
        preview = r.text[:300].replace("\n", " ")
        print(f"\n[{rank}] score={r.score:.3f}  {r.citation}\n    {preview}{'...' if len(r.text) > 300 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--source", help="only search this file (metadata filter)")
    args = parser.parse_args()

    s = get_settings()
    store = VectorStore(s.chroma_dir, s.collection_name)
    if store.count() == 0:
        sys.exit("Collection is empty. Run: python scripts/ingest.py")

    embedder = get_embedder(s.embedding_model)
    where = {"source": args.source} if args.source else None
    top_k = args.top_k or s.top_k

    def ask(q: str) -> None:
        show(store.search(embedder.embed_query(q), top_k=top_k, where=where))

    if args.question:
        ask(args.question)
        return

    print(f"{store.count()} chunks from {len(store.list_sources())} file(s). Ctrl+C to quit.")
    try:
        while q := input("\nQuestion> ").strip():
            ask(q)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
