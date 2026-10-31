"""FastAPI backend.

Run from the project folder:
    uvicorn rag.api:app --app-dir src --reload
Then open http://127.0.0.1:8000/docs for interactive API docs.

Endpoints
    GET  /health      liveness check (doesn't load models)
    GET  /documents   ingested files and chunk count
    POST /ask         answer a question (JSON, or Server-Sent Events with "stream": true)
    POST /upload      add a PDF/DOCX/TXT/MD file and index it
"""
import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .llm import LLMUnavailableError
from .loaders import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)
    mode: Literal["vector", "bm25", "hybrid", "rerank"] = "rerank"
    top_k: int | None = Field(default=None, ge=1, le=20)
    source: str | None = None  # restrict to one file
    stream: bool = False


class Source(BaseModel):
    n: int
    source: str | None
    page: int | None
    citation: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    query: str
    sources: list[Source]
    cited: list[int]
    provider: str | None
    latency_ms: int


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def create_app(settings: Settings | None = None, pipeline_factory: Callable | None = None,
               ingest_fn: Callable | None = None) -> FastAPI:
    """Factories are injectable so tests can run without models or API keys."""
    settings = settings or get_settings()
    app = FastAPI(title="RAG Document Assistant", version="0.4.0")
    lock = threading.Lock()
    state: dict = {}

    def get_pipeline():
        # Built on first use: loading embedding/re-ranking models takes several seconds,
        # and /health should answer instantly.
        if "pipeline" not in state:
            with lock:
                if "pipeline" not in state:
                    if pipeline_factory:
                        state["pipeline"] = pipeline_factory()
                    else:
                        from .pipeline import RAGPipeline

                        state["pipeline"] = RAGPipeline(settings)
        return state["pipeline"]

    def default_ingest(paths: list[Path]) -> int:
        from .ingestion import ingest_paths

        pipeline = get_pipeline()
        report = ingest_paths(paths, settings, pipeline.retriever.store)
        pipeline.retriever.refresh()  # make new chunks searchable by BM25 too
        return report.chunks

    ingest = ingest_fn or default_ingest

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/documents")
    def documents():
        store = get_pipeline().retriever.store
        return {"documents": store.list_sources(), "chunks": store.count()}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        pipeline = get_pipeline()
        history = [m.model_dump() for m in req.history]
        kwargs = {"history": history, "mode": req.mode, "top_k": req.top_k, "source": req.source}

        if req.stream:
            events = (_sse(e) for e in pipeline.stream(req.question, **kwargs))
            return StreamingResponse(events, media_type="text/event-stream")

        try:
            result = pipeline.answer(req.question, **kwargs)
        except LLMUnavailableError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return AskResponse(**result.__dict__)

    @app.post("/upload")
    def upload(file: UploadFile = File(...)):  # noqa: B008 (standard FastAPI dependency pattern)
        name = Path(file.filename or "").name  # drop any directory parts a client sends
        ext = Path(name).suffix.lower()
        if not name or ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {sorted(SUPPORTED_EXTENSIONS)}")

        limit = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"File larger than {settings.max_upload_mb} MB")
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        path = settings.docs_dir / name
        path.write_bytes(data)

        chunks = ingest([path])
        if not chunks:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="No text could be extracted (scanned PDF or empty document?)")
        return {"file": name, "chunks": chunks}

    return app


app = create_app()
