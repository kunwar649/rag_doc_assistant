"""Test retrieval from the command line (no LLM yet).

Usage:
    python scripts/query.py "What does exit code 2 mean?"                 # default: rerank
    python scripts/query.py "utf-8-sig" --mode bm25
    python scripts/query.py "ctl00_PageContent_dlstLicensure" --compare   # all modes side by side
    python scripts/query.py "pricing" --top-k 3 --source handbook.pdf
    python scripts/query.py                                               # interactive mode

Modes: vector | bm25 | hybrid | rerank
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings
from rag.retriever import MODES, Retriever


def show(results, preview_len: int = 300) -> None:
    if not results:
        print("No results.")
    for rank, r in enumerate(results, start=1):
        preview = r.text[:preview_len].replace("\n", " ")
        more = "..." if len(r.text) > preview_len else ""
        print(f"\n[{rank}] score={r.score:.3f}  {r.citation}\n    {preview}{more}")


def compare(retriever: Retriever, question: str, top_k: int, source: str | None) -> None:
    """Show where each mode ranks things, one line per result."""
    for mode in MODES:
        print(f"\n=== {mode} ===")
        results = retriever.retrieve(question, mode=mode, top_k=top_k, source=source)
        if not results:
            print("  (no results)")
        for rank, r in enumerate(results, start=1):
            print(f"  [{rank}] {r.score:8.3f}  {r.citation}  | {r.text[:70].replace(chr(10), ' ')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--mode", choices=MODES, default="rerank")
    parser.add_argument("--compare", action="store_true", help="run every mode and compare")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--source", help="only search this file (metadata filter)")
    args = parser.parse_args()

    s = get_settings()
    retriever = Retriever(s)
    if retriever.store.count() == 0:
        sys.exit("Collection is empty. Run: python scripts/ingest.py")

    top_k = args.top_k or s.top_k

    def ask(q: str) -> None:
        if args.compare:
            compare(retriever, q, top_k, args.source)
        else:
            show(retriever.retrieve(q, mode=args.mode, top_k=top_k, source=args.source))

    if args.question:
        ask(args.question)
        return

    print(
        f"{retriever.store.count()} chunks from {len(retriever.store.list_sources())} file(s). "
        f"Mode: {'compare' if args.compare else args.mode}. Ctrl+C to quit."
    )
    try:
        while q := input("\nQuestion> ").strip():
            ask(q)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
