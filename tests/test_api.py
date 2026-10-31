import json

import pytest
from fakes import FakeRetriever, connection_error, fake_client
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.config import Settings
from rag.llm import LLMClient, Provider
from rag.pipeline import RAGPipeline


@pytest.fixture
def setup(tmp_path):
    settings = Settings(docs_dir=tmp_path / "docs", max_upload_mb=1, _env_file=None)
    retriever = FakeRetriever()
    llm = LLMClient([Provider("fake", "m", fake_client("Exit code 2 means no table [1]."))])
    pipeline = RAGPipeline(settings, retriever=retriever, llm=llm)
    ingested = []

    def ingest(paths):
        ingested.extend(paths)
        return 0 if paths[0].name.startswith("empty") else 4

    client = TestClient(create_app(settings, pipeline_factory=lambda: pipeline, ingest_fn=ingest))
    return client, pipeline, ingested, settings


def test_health(setup):
    client, *_ = setup
    assert client.get("/health").json() == {"status": "ok"}


def test_documents(setup):
    client, *_ = setup
    assert client.get("/documents").json() == {"documents": ["a.pdf"], "chunks": 3}


def test_ask_json(setup):
    client, *_ = setup
    body = client.post("/ask", json={"question": "What does exit code 2 mean?"}).json()
    assert body["answer"] == "Exit code 2 means no table [1]."
    assert body["cited"] == [1]
    assert body["sources"][0]["citation"] == "a.pdf (p. 7)"


def test_ask_validation(setup):
    client, *_ = setup
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "q", "mode": "magic"}).status_code == 422
    assert client.post("/ask", json={"question": "q", "top_k": 99}).status_code == 422


def test_ask_stream_sse(setup):
    client, *_ = setup
    with client.stream("POST", "/ask", json={"question": "q", "stream": True}) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "token").strip() == "Exit code 2 means no table [1]."


def test_ask_returns_503_when_llm_down(setup):
    client, pipeline, *_ = setup
    pipeline.llm = LLMClient([Provider("bad", "m", fake_client(connection_error()))])
    assert client.post("/ask", json={"question": "q"}).status_code == 503


def test_upload_saves_and_ingests(setup):
    client, _, ingested, settings = setup
    resp = client.post("/upload", files={"file": ("../../evil/notes.txt", b"hello world", "text/plain")})
    assert resp.status_code == 200 and resp.json() == {"file": "notes.txt", "chunks": 4}
    assert (settings.docs_dir / "notes.txt").read_bytes() == b"hello world"  # path traversal stripped
    assert ingested == [settings.docs_dir / "notes.txt"]


def test_upload_rejects_bad_type_size_and_empty_text(setup):
    client, _, _, settings = setup
    assert client.post("/upload", files={"file": ("x.exe", b"x", "application/octet-stream")}).status_code == 400
    assert client.post("/upload", files={"file": ("big.txt", b"x" * (1024 * 1024 + 1), "text/plain")}).status_code == 413
    resp = client.post("/upload", files={"file": ("empty.txt", b"   ", "text/plain")})
    assert resp.status_code == 422
    assert not (settings.docs_dir / "empty.txt").exists()  # removed when nothing was extracted
