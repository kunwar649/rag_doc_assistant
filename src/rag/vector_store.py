"""ChromaDB wrapper (embedded, persisted to disk, free)."""
from pathlib import Path

import chromadb

from .models import Chunk, SearchResult


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # match normalized embeddings
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 500) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=embeddings[start : start + batch_size],
            )

    def delete_source(self, source: str) -> None:
        """Remove every chunk of one file, e.g. before re-ingesting an edited version."""
        self.collection.delete(where={"source": source})

    def search(self, query_embedding: list[float], top_k: int = 5, where: dict | None = None) -> list[SearchResult]:
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            # Chroma returns cosine *distance*; convert to similarity.
            SearchResult(text=doc, metadata=meta, score=1 - dist, id=cid)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk; used to build the BM25 keyword index."""
        res = self.collection.get(include=["documents", "metadatas"])
        return [
            Chunk(id=cid, text=doc, metadata=meta)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})
