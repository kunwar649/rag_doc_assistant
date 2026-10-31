"""Cross-encoder re-ranking (local, free).

A bi-encoder (our embedding model) embeds query and chunk *separately* and
compares the vectors: fast, so it can scan every chunk, but coarse.
A cross-encoder reads query and chunk *together* in one pass: much more
accurate, but too slow to run on every chunk. So we retrieve ~20 candidates
cheaply, then let the cross-encoder pick the best few.
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from .models import SearchResult


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not results:
            return []
        scores = self.model.predict([(query, r.text) for r in results])
        ranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        # ms-marco cross-encoders output raw logits: higher = more relevant,
        # negative values are normal for weak matches.
        return [SearchResult(text=r.text, metadata=r.metadata, score=float(s), id=r.id) for r, s in ranked]


@lru_cache
def get_reranker(model_name: str) -> Reranker:
    return Reranker(model_name)
