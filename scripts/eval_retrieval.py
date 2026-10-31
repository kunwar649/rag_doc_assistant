"""Score every retrieval mode against your golden questions.

Usage:
    python scripts/eval_retrieval.py                       # uses data/eval_questions.md
    python scripts/eval_retrieval.py --questions my.md --top-k 5
    python scripts/eval_retrieval.py --show-misses         # list questions each mode got wrong

Metrics:
    hit@1  share of questions where the correct chunk was ranked first
    hit@k  share where it appeared anywhere in the top k
    MRR    mean reciprocal rank: 1.0 if always first, 0.5 if always second, 0 if never found
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings
from rag.eval_retrieval import first_hit_rank, parse_questions, summarize
from rag.retriever import MODES, Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/eval_questions.md"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    if not args.questions.exists():
        sys.exit(f"Question file not found: {args.questions}")
    questions = parse_questions(args.questions)
    if not questions:
        sys.exit("No questions parsed. Check the Q:/A:/Source:/Type: format.")

    retriever = Retriever(get_settings())
    if retriever.store.count() == 0:
        sys.exit("Collection is empty. Run: python scripts/ingest.py")

    known = set(retriever.store.list_sources())
    missing = sorted({q.source for q in questions} - known)
    if missing:
        print(f"WARNING: these sources aren't ingested, so their questions will always miss: {missing}\n")

    k = args.top_k
    qtypes = sorted({q.qtype for q in questions})
    print(f"{len(questions)} questions ({', '.join(f'{t}: {sum(q.qtype == t for q in questions)}' for t in qtypes)}), top_k={k}\n")

    header = f"{'mode':<8} {'subset':<8} {'hit@1':>6} {'hit@' + str(k):>6} {'MRR':>6} {'ms/q':>6}"
    print(header)
    print("-" * len(header))

    misses: dict[str, list[str]] = {}
    for mode in args.modes:
        ranks, start = [], time.perf_counter()
        for q in questions:
            ranks.append(first_hit_rank(retriever.retrieve(q.question, mode=mode, top_k=k), q))
        ms = (time.perf_counter() - start) * 1000 / len(questions)

        subsets = [("all", ranks)] + [
            (t, [r for r, q in zip(ranks, questions) if q.qtype == t]) for t in qtypes
        ]
        for name, sub in subsets:
            m = summarize(sub, k)
            timing = f"{ms:6.0f}" if name == "all" else ""
            print(f"{mode:<8} {name:<8} {m['hit@1']:6.2f} {m[f'hit@{k}']:6.2f} {m['mrr']:6.2f} {timing:>6}")
        print()
        misses[mode] = [f"{q.qid} (rank {r or '-'}) {q.question}" for r, q in zip(ranks, questions) if r != 1]

    if args.show_misses:
        for mode, items in misses.items():
            print(f"\n--- {mode}: not ranked first ({len(items)}) ---")
            for item in items:
                print("  " + item)


if __name__ == "__main__":
    main()
