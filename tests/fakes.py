"""Test doubles: let pipeline/API tests run without models, API keys or network."""
from types import SimpleNamespace

import httpx
from openai import APIConnectionError

from rag.models import SearchResult


def connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))


class FakeCompletions:
    """Mimics client.chat.completions. `reply` is a string, or an exception to raise."""

    def __init__(self, reply, tokens=None):
        self.reply = reply
        self.tokens = tokens  # for streaming; defaults to splitting reply into words
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        if kwargs.get("stream"):
            parts = self.tokens or [w + " " for w in self.reply.split()]
            return iter(
                [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])]  # reasoning/role chunk
                + [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))]) for p in parts]
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def fake_client(reply, tokens=None):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(reply, tokens)))


class FakeStore:
    def __init__(self, sources=("a.pdf",)):
        self._sources = list(sources)

    def list_sources(self):
        return self._sources

    def count(self):
        return 3


class FakeRetriever:
    def __init__(self, results=None):
        self.results = results if results is not None else [
            SearchResult(text="Exit code 2 means the result table did not appear.", metadata={"source": "a.pdf", "page": 7}, score=2.0, id="c1"),
            SearchResult(text="Output path is output/results.csv.", metadata={"source": "a.pdf", "page": 7}, score=0.5, id="c2"),
        ]
        self.queries = []
        self.store = FakeStore()
        self.refreshed = 0

    def retrieve(self, query, mode="rerank", top_k=5, source=None):
        self.queries.append(query)
        return self.results[:top_k]

    def refresh(self):
        self.refreshed += 1
