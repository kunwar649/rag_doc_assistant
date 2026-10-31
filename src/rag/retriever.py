"""One entry point for every retrieval strategy, so they can be compared fairly."""
from .bm25 import BM25Index
from .config import Settings
from .embeddings import get_embedder
from .fusion import reciprocal_rank_fusion
from .models import SearchResult
from .vector_store import VectorStore

MODES = ("vector", "bm25", "hybrid", "rerank")


class Retriever:
    """
    vector -> embeddings only (Day 2 behaviour)
    bm25   -> keywords only
    hybrid -> vector + bm25 merged with RRF
    rerank -> hybrid candidates re-scored by a cross-encoder (the default)
    """

    def __init__(self, settings: Settings, store: VectorStore | None = None):
        self.s = settings
        self.store = store or VectorStore(settings.chroma_dir, settings.collection_name)
        self.embedder = get_embedder(settings.embedding_model)
        self.bm25 = BM25Index(self.store.all_chunks())
        self._reranker = None  # loaded only when rerank mode is used

    def refresh(self) -> None:
        """Rebuild the keyword index after documents are added or removed."""
        self.bm25 = BM25Index(self.store.all_chunks())

    def _vector(self, query: str, k: int, source: str | None) -> list[SearchResult]:
        where = {"source": source} if source else None
        return self.store.search(self.embedder.embed_query(query), top_k=k, where=where)

    def _hybrid(self, query: str, k: int, source: str | None) -> list[SearchResult]:
        n = self.s.candidate_k  # retrieve broadly from both, then fuse
        return reciprocal_rank_fusion(
            [self._vector(query, n, source), self.bm25.search(query, n, source)],
            k=self.s.rrf_k,
            top_k=k,
        )

    def retrieve(self, query: str, mode: str = "rerank", top_k: int | None = None, source: str | None = None) -> list[SearchResult]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        top_k = top_k or self.s.top_k

        if mode == "vector":
            return self._vector(query, top_k, source)
        if mode == "bm25":
            return self.bm25.search(query, top_k, source)
        if mode == "hybrid":
            return self._hybrid(query, top_k, source)

        candidates = self._hybrid(query, self.s.candidate_k, source)
        if self._reranker is None:
            from .reranker import get_reranker

            self._reranker = get_reranker(self.s.reranker_model)
        return self._reranker.rerank(query, candidates, top_k)
