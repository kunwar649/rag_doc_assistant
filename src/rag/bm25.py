"""BM25 keyword search: catches exact terms (IDs, codes, names) that embeddings miss."""
import re

from rank_bm25 import BM25Okapi

from .models import Chunk, SearchResult

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens that keep identifiers searchable.

    "ctl00_PageContent_txtLicense" yields the whole identifier *and* its parts
    (ctl00, pagecontent, txtlicense), so both an exact paste and a partial
    term match.
    """
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        tokens.append(tok)
        if "_" in tok:
            tokens.extend(part for part in tok.split("_") if part)
    return tokens


class BM25Index:
    """In-memory index over all chunks.

    Rebuilt from the vector store on startup. That takes well under a second for
    thousands of chunks; for millions you'd use a search engine (Elasticsearch,
    OpenSearch) or a vector DB with built-in sparse search instead.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        tokenized = [tokenize(c.text) for c in chunks]
        self._token_sets = [set(t) for t in tokenized]
        self._bm25 = BM25Okapi(tokenized) if chunks else None

    def search(self, query: str, top_k: int = 5, source: str | None = None) -> list[SearchResult]:
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        query_set = set(tokens)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked:
            # Filter on actual term overlap, not score > 0: BM25Okapi gives terms
            # found in over half the chunks a negative IDF, so a real match on a
            # common word ("license") can still score <= 0.
            if not query_set & self._token_sets[i]:
                continue
            chunk = self.chunks[i]
            if source and chunk.metadata.get("source") != source:
                continue
            results.append(SearchResult(text=chunk.text, metadata=chunk.metadata, score=float(scores[i]), id=chunk.id))
            if len(results) == top_k:
                break
        return results
