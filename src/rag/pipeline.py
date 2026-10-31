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
from .tracing import NOOP, Tracer, create_tracer

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    answer: str
    query: str  # the standalone query used for retrieval (differs from the question for follow-ups)
    sources: list[dict] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    provider: str | None = None
    latency_ms: int = 0
    usage: dict | None = None  # tokens for the answer call: {"input": n, "output": n}


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


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


class RAGPipeline:
    def __init__(self, settings: Settings, retriever=None, llm: LLMClient | None = None, tracer: Tracer | None = None):
        self.s = settings
        if retriever is None:
            from .retriever import (
                Retriever,  # heavy import (models); skipped when a test injects one
            )

            retriever = Retriever(settings)
        self.retriever = retriever
        self.llm = llm or LLMClient.from_settings(settings)
        self.tracer = tracer or create_tracer(settings)

    # --- tracing helpers ---------------------------------------------------

    def _end_generation(self, gen, output: str) -> None:
        gen.update(output=output)
        if self.llm.last_usage:
            gen.update(usage_details=self.llm.last_usage)
        if self.llm.last_provider:
            # "groq:openai/gpt-oss-20b" -> model name Langfuse can price
            gen.update(model=self.llm.last_provider.split(":", 1)[-1], metadata={"provider": self.llm.last_provider})
        gen.end()

    def _start_trace(self, question: str, history: list[dict], mode: str, top_k, source):
        return self.tracer.start(
            "rag-query",
            input={"question": question, "previous_turns": len(history) // 2},
            metadata={"mode": mode, "top_k": top_k or self.s.top_k, "source_filter": source},
        )

    # --- steps -------------------------------------------------------------

    def _trim_history(self, history: list[dict] | None) -> list[dict]:
        if not history:
            return []
        keep = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
        return keep[-2 * self.s.history_turns :]

    def rewrite_query(self, question: str, history: list[dict], trace=NOOP) -> str:
        """Turn a follow-up ("what about its timeout?") into a standalone search query."""
        if not history:
            return question
        messages = build_condense_messages(question, history)
        gen = trace.child("rewrite", as_type="generation", input=messages)
        try:
            rewritten = self.llm.complete(messages).strip().strip('"') or question
            self._end_generation(gen, rewritten)
            return rewritten
        except Exception as e:  # retrieval with the raw question beats failing the whole request
            logger.warning("Query rewrite failed; using the original question", exc_info=True)
            gen.end(level="WARNING", status_message=f"rewrite failed, used original question: {e}")
            return question

    def _prepare(self, question, history, mode, top_k, source, trace=NOOP):
        history = self._trim_history(history)
        query = self.rewrite_query(question, history, trace)

        span = trace.child("retrieve", input={"query": query, "mode": mode, "top_k": top_k or self.s.top_k, "source": source})
        results = self.retriever.retrieve(query, mode=mode, top_k=top_k or self.s.top_k, source=source)
        span.end(output=[{"citation": r.citation, "score": round(r.score, 3)} for r in results])

        messages = build_answer_messages(question, results, history) if results else None
        return query, results, messages, history

    # --- public API --------------------------------------------------------

    def answer(self, question: str, history: list[dict] | None = None, mode: str = "rerank",
               top_k: int | None = None, source: str | None = None) -> Answer:
        t0 = time.perf_counter()
        trace = self._start_trace(question, self._trim_history(history), mode, top_k, source)
        try:
            query, results, messages, _ = self._prepare(question, history, mode, top_k, source, trace)
            if not results:  # nothing retrieved: don't spend an LLM call to say so
                trace.end(output={"answer": NO_ANSWER, "reason": "no chunks retrieved"})
                return Answer(answer=NO_ANSWER, query=query, latency_ms=_ms(t0))

            gen = trace.child("generate", as_type="generation", input=messages)
            text = self.llm.complete(messages)
            self._end_generation(gen, text)
            cited = extract_citations(text, len(results))
            trace.end(output={"answer": text, "cited": cited})
            return Answer(
                answer=text,
                query=query,
                sources=source_dicts(results),
                cited=cited,
                provider=self.llm.last_provider,
                latency_ms=_ms(t0),
                usage=self.llm.last_usage,
            )
        except Exception as e:
            trace.end(level="ERROR", status_message=str(e))
            raise

    def stream(self, question: str, history: list[dict] | None = None, mode: str = "rerank",
               top_k: int | None = None, source: str | None = None) -> Iterator[dict]:
        """Yields events: one "sources", many "token", one "done" (or "error")."""
        t0 = time.perf_counter()
        trace = self._start_trace(question, self._trim_history(history), mode, top_k, source)
        ended = False
        try:
            query, results, messages, _ = self._prepare(question, history, mode, top_k, source, trace)
            yield {"type": "sources", "query": query, "sources": source_dicts(results)}

            if not results:
                yield {"type": "token", "text": NO_ANSWER}
                trace.end(output={"answer": NO_ANSWER, "reason": "no chunks retrieved"})
                ended = True
                yield {"type": "done", "cited": [], "provider": None, "usage": None, "latency_ms": _ms(t0)}
                return

            gen = trace.child("generate", as_type="generation", input=messages)
            parts: list[str] = []
            try:
                for token in self.llm.stream(messages):
                    parts.append(token)
                    yield {"type": "token", "text": token}
            except Exception as e:
                logger.exception("Answer generation failed")
                gen.end(level="ERROR", status_message=str(e))
                trace.end(level="ERROR", status_message=str(e))
                ended = True
                yield {"type": "error", "message": str(e)}
                return

            text = "".join(parts)
            self._end_generation(gen, text)
            cited = extract_citations(text, len(results))
            trace.end(output={"answer": text, "cited": cited})
            ended = True
            yield {
                "type": "done",
                "cited": cited,
                "provider": self.llm.last_provider,
                "usage": self.llm.last_usage,
                "latency_ms": _ms(t0),
            }
        finally:
            if not ended:  # client disconnected mid-stream, or an unexpected error
                trace.end(level="WARNING", status_message="stream ended early")
