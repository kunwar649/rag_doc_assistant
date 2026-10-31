"""Reciprocal Rank Fusion: merge ranked lists from different retrievers."""
from .models import SearchResult


def reciprocal_rank_fusion(result_lists: list[list[SearchResult]], k: int = 60, top_k: int | None = None) -> list[SearchResult]:
    """Combine rankings using only positions, never raw scores.

    Cosine similarities (~0.4-0.9) and BM25 scores (0-30+) aren't on the same
    scale, so adding them directly would let BM25 dominate. RRF sidesteps that:
    each list contributes 1 / (k + rank) for every chunk it contains.

    k=60 is the value from the original RRF paper (Cormack et al., 2009); it
    dampens the gap between rank 1 and rank 2 so one retriever can't dominate.
    """
    fused: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            fused[r.id] = fused.get(r.id, 0.0) + 1.0 / (k + rank)
            best.setdefault(r.id, r)

    order = sorted(fused, key=fused.get, reverse=True)
    if top_k is not None:
        order = order[:top_k]
    return [SearchResult(text=best[i].text, metadata=best[i].metadata, score=fused[i], id=i) for i in order]
