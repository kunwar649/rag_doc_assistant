"""Local, free embeddings via sentence-transformers (runs on CPU)."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

# BGE models are trained to embed *queries* with this instruction prefix;
# documents are embedded as-is. Skipping it noticeably hurts retrieval.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # unit vectors -> dot product == cosine
            show_progress_bar=len(texts) > 100,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(self._query_prefix + query, normalize_embeddings=True)
        return vector.tolist()


@lru_cache
def get_embedder(model_name: str) -> Embedder:
    # Loading the model takes seconds; do it once per process.
    return Embedder(model_name)
