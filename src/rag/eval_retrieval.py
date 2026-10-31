"""Retrieval evaluation: did the right chunk come back, and how high?

Reads a golden question file in this format:

    Q1: What is the default output path?
    A: output/tennessee_license_results.csv
    Source: Tennessee_Adapter_Detailed_Explanation.pdf, p. 7
    Type: meaning

Questions with `Type: negative` (answer not in the documents) are skipped here;
they test the LLM's "I don't know" behaviour on Day 4.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import SearchResult


@dataclass
class EvalQuestion:
    qid: str
    question: str
    answer: str
    source: str  # file name
    pages: set[int] = field(default_factory=set)  # empty = any page in that file counts
    qtype: str = "meaning"


_Q = re.compile(r"^Q(\d+):\s*(.+)$")


def parse_questions(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    current: dict | None = None

    def flush() -> None:
        if current and current.get("source") and current.get("qtype") != "negative":
            questions.append(EvalQuestion(**current))

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if m := _Q.match(line):
            flush()
            current = {"qid": f"Q{m.group(1)}", "question": m.group(2), "answer": "", "source": "", "pages": set()}
        elif current is None:
            continue
        elif line.startswith("A:"):
            current["answer"] = line[2:].strip()
        elif line.startswith("Source:"):
            value = line[len("Source:") :].strip()
            current["source"] = value.split(",")[0].strip()
            current["pages"] = {int(p) for p in re.findall(r"p\.\s*(\d+)", value)}
        elif line.startswith("Type:"):
            current["qtype"] = line[len("Type:") :].strip().lower()
    flush()
    return questions


def is_hit(result: SearchResult, q: EvalQuestion) -> bool:
    if result.metadata.get("source") != q.source:
        return False
    return not q.pages or result.metadata.get("page") in q.pages


def first_hit_rank(results: list[SearchResult], q: EvalQuestion) -> int | None:
    """1-based rank of the first correct chunk, or None if it wasn't retrieved."""
    for rank, r in enumerate(results, start=1):
        if is_hit(r, q):
            return rank
    return None


def summarize(ranks: list[int | None], k: int) -> dict[str, float]:
    """hit@1, hit@k and MRR (mean reciprocal rank) for a list of first-hit ranks."""
    n = len(ranks) or 1
    return {
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }
