# RAG Document Assistant

Chat with your documents (PDF, DOCX, TXT, MD) and get answers with citations.
Built entirely on free tools: local embeddings, embedded vector DB, free-tier LLMs.

> **Status:** Day 1–2 complete — ingestion pipeline and vector retrieval.
> Next: hybrid search + re-ranking (Day 3), LLM answers + API (Day 4).

## Architecture (Day 1–2)

```
data/docs/*.pdf|docx|txt
        │
        ▼
  loaders.py ── extract text, keep {source, page}
        │
        ▼
  chunking.py ── recursive split (1000 chars, 150 overlap), deterministic IDs
        │
        ▼
  embeddings.py ── bge-small-en-v1.5 (local CPU), normalized vectors
        │
        ▼
  vector_store.py ── ChromaDB (persistent, cosine / HNSW)
        │
        ▼
  query.py ── embed question → top-k chunks with citations
```

## Project structure

```
rag-doc-assistant/
├── src/rag/
│   ├── config.py        # settings from .env (pydantic-settings)
│   ├── models.py        # Page, Chunk, SearchResult
│   ├── loaders.py       # PDF / DOCX (incl. tables) / TXT loaders
│   ├── chunking.py      # recursive splitter + stable chunk IDs
│   ├── embeddings.py    # sentence-transformers wrapper
│   └── vector_store.py  # ChromaDB wrapper: upsert, search, delete
├── scripts/
│   ├── ingest.py        # load → chunk → embed → store
│   └── query.py         # CLI retrieval test
├── tests/               # pytest unit tests
├── data/docs/           # put your documents here
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

`sentence-transformers` installs PyTorch (~a few hundred MB on CPU). To keep it smaller:
`pip install torch --index-url https://download.pytorch.org/whl/cpu` before the requirements.

## Run

```bash
# 1. Add some documents
cp ~/Downloads/*.pdf data/docs/

# 2. Ingest (first run downloads the embedding model, ~130 MB)
python scripts/ingest.py

# 3. Test retrieval
python scripts/query.py "What is the refund policy?"
python scripts/query.py "pricing" --top-k 3 --source handbook.pdf
python scripts/query.py            # interactive mode

# Tests and lint
pytest -q
ruff check .
```

Re-running `ingest.py` replaces the old chunks of each file it finds, so edited
documents don't leave stale text behind. Use `--reset` to wipe everything.

## Design decisions (interview talking points)

| Decision | Why |
|---|---|
| Recursive splitter, 1000 chars / 150 overlap | Breaks at paragraph → sentence → word boundaries; overlap keeps context that spans a cut. Tune later with eval data. |
| Page-level metadata on every chunk | Enables citations ("handbook.pdf, p. 4") and per-file filtering. |
| Deterministic chunk IDs (SHA-256) | Re-ingesting is idempotent — upserts overwrite instead of duplicating. |
| `bge-small-en-v1.5` | Strong quality for its size, runs on CPU, free. Uses BGE's query instruction prefix for better retrieval. |
| Normalized embeddings + cosine space | Scores are comparable similarities in [−1, 1]. |
| ChromaDB embedded | Zero setup, persists to disk. Swappable for Qdrant later behind the same `VectorStore` interface. |
| One bad file doesn't stop ingest | Errors are logged per file; real document sets always contain a few broken files. |

## Known limitations (to address later)
- Scanned/image-only PDFs have no extractable text (needs OCR).
- Pure vector search misses exact terms like IDs and codes → fixed by hybrid BM25 search on Day 3.
- No answer generation yet → LLM integration on Day 4.

## Roadmap
- [x] Day 1 – Loaders, chunking, metadata
- [x] Day 2 – Embeddings, Chroma, retrieval CLI
- [ ] Day 3 – BM25 hybrid search (RRF) + cross-encoder re-ranking
- [ ] Day 4 – Groq/Gemini answers with citations, FastAPI `/upload` `/ask`
- [ ] Day 5 – Streamlit UI + Langfuse tracing
- [ ] Day 6 – RAGAS evaluation: vector-only vs hybrid + re-rank
- [ ] Day 7 – Docker, Hugging Face Spaces deploy, GitHub Actions
