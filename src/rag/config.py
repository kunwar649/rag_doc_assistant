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

    # Day 4: answer generation. Providers are tried in order; any with an empty key is skipped.
    # Free-tier model lineups change often: if you get "model not found / decommissioned",
    # check console.groq.com/docs/models or ai.google.dev and update the model name in .env.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    google_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    llm_temperature: float = 0.1  # low = stick closely to the sources
    llm_max_tokens: int = 2048  # reasoning models count their thinking tokens here too
    llm_timeout: float = 60.0
    history_turns: int = 4  # previous question/answer pairs used for follow-ups
    max_upload_mb: int = 20
    # Browser apps allowed to call the API (comma-separated). The Next.js dev server runs on :3000.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Day 5: optional Langfuse tracing (leave keys empty to disable).
    # EU cloud: https://cloud.langfuse.com   US cloud: https://us.cloud.langfuse.com
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
