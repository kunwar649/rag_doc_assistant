"""Retrieval-augmented generation: retrieve chunks, then have the LLM answer from them."""
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .config import Settings
from .llm import LLMClient
from .models import SearchResult
from .prompts import (
    NO_ANSWER,
    build_answer_messages,
    build_condense_messages,
    extract_citations,
)

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    answer: str
    query: str  # the standalone query used for retrieval (differs from the question for follow-ups)
    sources: list[dict] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    provider: str | None = None
    latency_ms: int = 0


def source_dicts(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "n": i,
            "source": r.metadata.get("source"),
            "page": r.metadata.get("page"),
            "citation": r.citation,
            "score": round(r.score, 4),
            "text": r.text,
        }
        for i, r in enumerate(results, start=1)
    ]


class RAGPipeline:
    def __init__(self, settings: Settings, retriever=None, llm: LLMClient | None = None):
        self.s = settings
        if retriever is None:
            from .retriever import (
                Retriever,  # heavy import (models); skipped when a test injects one
            )

            retriever = Retriever(settings)
        self.retriever = retriever
        self.llm = llm or LLMClient.from_settings(settings)

    def _trim_history(self, history: list[dict] | None) -> list[dict]:
        if not history:
            return []
        keep = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
        return keep[-2 * self.s.history_turns :]

    def rewrite_query(self, question: str, history: list[dict]) -> str:
        """Turn a follow-up ("what about its timeout?") into a standalone search query."""
        if not history:
            return question
        try:
            rewritten = self.llm.complete(build_condense_messages(question, history)).strip().strip('"')
            return rewritten or question
        except Exception:  # retrieval with the raw question beats failing the whole request
            logger.warning("Query rewrite failed; using the original question", exc_info=True)
            return question

    def _prepare(self, question, history, mode, top_k, source):
        history = self._trim_history(history)
        query = self.rewrite_query(question, history)
        results = self.retriever.retrieve(query, mode=mode, top_k=top_k or self.s.top_k, source=source)
        messages = build_answer_messages(question, results, history) if results else None
        return query, results, messages

    def answer(self, question: str, history: list[dict] | None = None, mode: str = "rerank",
               top_k: int | None = None, source: str | None = None) -> Answer:
        t0 = time.perf_counter()
        query, results, messages = self._prepare(question, history, mode, top_k, source)
        if not results:  # nothing retrieved: don't spend an LLM call to say so
            return Answer(answer=NO_ANSWER, query=query, latency_ms=int((time.perf_counter() - t0) * 1000))

        text = self.llm.complete(messages)
        return Answer(
            answer=text,
            query=query,
            sources=source_dicts(results),
            cited=extract_citations(text, len(results)),
            provider=self.llm.last_provider,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    def stream(self, question: str, history: list[dict] | None = None, mode: str = "rerank",
               top_k: int | None = None, source: str | None = None) -> Iterator[dict]:
        """Yields events: one "sources", many "token", one "done" (or "error")."""
        t0 = time.perf_counter()
        query, results, messages = self._prepare(question, history, mode, top_k, source)
        yield {"type": "sources", "query": query, "sources": source_dicts(results)}

        if not results:
            yield {"type": "token", "text": NO_ANSWER}
            yield {"type": "done", "cited": [], "provider": None, "latency_ms": int((time.perf_counter() - t0) * 1000)}
            return

        parts: list[str] = []
        try:
            for token in self.llm.stream(messages):
                parts.append(token)
                yield {"type": "token", "text": token}
        except Exception as e:
            logger.exception("Answer generation failed")
            yield {"type": "error", "message": str(e)}
            return
        yield {
            "type": "done",
            "cited": extract_citations("".join(parts), len(results)),
            "provider": self.llm.last_provider,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
