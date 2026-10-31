"""Chat with your documents in the terminal: streamed answers with citations.

Usage:
    python scripts/ask.py "What does exit code 2 mean?"
    python scripts/ask.py                       # interactive chat; follow-up questions work
    python scripts/ask.py --mode vector         # compare retrieval modes end to end
    python scripts/ask.py --show-sources        # print the full text of each cited source
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings
from rag.llm import LLMUnavailableError
from rag.pipeline import RAGPipeline
from rag.retriever import MODES

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")


def run_turn(pipeline: RAGPipeline, question: str, history: list[dict], args) -> str:
    sources, parts = [], []
    for event in pipeline.stream(question, history=history, mode=args.mode, source=args.source):
        kind = event["type"]
        if kind == "sources":
            sources = event["sources"]
            if event["query"] != question:
                print(f"  (searched for: {event['query']})")
            print()
        elif kind == "token":
            parts.append(event["text"])
            print(event["text"], end="", flush=True)
        elif kind == "error":
            print(f"\n\nERROR: {event['message']}")
            return ""
        elif kind == "done":
            print("\n")
            cited = set(event["cited"])
            for s in sources:
                mark = "*" if s["n"] in cited else " "
                print(f" {mark}[{s['n']}] {s['citation']}  (score {s['score']:.2f})")
                if args.show_sources and s["n"] in cited:
                    print("      " + s["text"][:600].replace("\n", "\n      "))
            if sources:
                print("  * = cited in the answer")
            print(f"  {event['provider'] or 'no LLM call'} | {event['latency_ms']} ms")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--mode", choices=MODES, default="rerank")
    parser.add_argument("--source", help="only search this file")
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args()

    try:
        pipeline = RAGPipeline(get_settings())
    except LLMUnavailableError as e:
        sys.exit(str(e))
    if pipeline.retriever.store.count() == 0:
        sys.exit("Collection is empty. Run: python scripts/ingest.py")

    if args.question:
        run_turn(pipeline, args.question, [], args)
        return

    print("Ask about your documents. Follow-up questions use the conversation. Ctrl+C to quit, 'reset' to clear history.")
    history: list[dict] = []
    try:
        while True:
            question = input("\nYou> ").strip()
            if not question:
                continue
            if question.lower() == "reset":
                history = []
                print("History cleared.")
                continue
            answer = run_turn(pipeline, question, history, args)
            if answer:
                history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
