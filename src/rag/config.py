"""Central configuration, loaded from environment variables / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    docs_dir: Path = Path("data/docs")
    chroma_dir: Path = Path("data/chroma")
    collection_name: str = "documents"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 5

    # Day 3: hybrid search + re-ranking
    candidate_k: int = 20  # candidates fetched from each retriever before fusing/re-ranking
    rrf_k: int = 60  # RRF damping constant
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    groq_api_key: str = ""
    google_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
