# RAG Document Assistant

Chat with your documents (PDF, DOCX, TXT, MD) and get grounded answers with citations.
Built entirely on free tools: local embeddings and re-ranker, embedded vector DB, free-tier LLMs.

> **Status:** Day 4 complete — LLM answers with citations, provider fallback, follow-up questions, FastAPI backend.
> Next: Streamlit UI + tracing (Day 5).

## Architecture

```
INGEST (scripts/ingest.py, POST /upload)
data/docs/*.pdf|docx|txt|md
   → loaders.py      extract text, keep {source, page}, strip repeated PDF headers/footers
   → chunking.py     recursive split (1000 chars, 150 overlap), deterministic IDs
   → embeddings.py   bge-small-en-v1.5 (local CPU), normalized vectors
   → vector_store.py ChromaDB (persistent, cosine / HNSW)

ANSWER (src/rag/pipeline.py, POST /ask)
question + history
   → rewrite follow-up into a standalone query (LLM, only when there is history)
   → retrieve:  vector (top 20) ┐
                BM25   (top 20) ┴─► RRF fusion ─► cross-encoder re-rank ─► top 5
   → prompt: rules + numbered sources [1]..[5] + question
   → LLM: Groq (gpt-oss-20b) ──fails?──► Gemini (fallback)
   → answer with [n] citations, cited sources, provider, latency
```

## Project structure

```
rag-doc-assistant/
├── src/rag/
│   ├── config.py          # settings from .env (pydantic-settings)
│   ├── models.py          # Page, Chunk, SearchResult
│   ├── loaders.py         # PDF / DOCX (incl. tables) / TXT loaders, header stripping
│   ├── chunking.py        # recursive splitter + stable chunk IDs
│   ├── embeddings.py      # sentence-transformers bi-encoder wrapper
│   ├── vector_store.py    # ChromaDB wrapper: upsert, search, delete
│   ├── ingestion.py       # load → chunk → embed → store (shared by CLI and API)
│   ├── bm25.py            # keyword index with identifier-aware tokenizer
│   ├── fusion.py          # Reciprocal Rank Fusion
│   ├── reranker.py        # cross-encoder re-ranking
│   ├── retriever.py       # one entry point: vector | bm25 | hybrid | rerank
│   ├── llm.py             # Groq/Gemini client with fallback and streaming
│   ├── prompts.py         # system prompt, context formatting, citation parsing
│   ├── pipeline.py        # retrieve → prompt → generate (JSON or streamed events)
│   ├── api.py             # FastAPI app
│   └── eval_retrieval.py  # golden-question parser + hit@k / MRR metrics
├── scripts/
│   ├── ingest.py          # index everything in data/docs
│   ├── query.py           # retrieval only, single mode or --compare
│   ├── ask.py             # terminal chat with streamed, cited answers
│   └── eval_retrieval.py  # score every retrieval mode against golden questions
├── tests/                 # pytest: unit + API tests (fake LLM, no keys needed)
├── data/docs/             # put your documents here
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

Add at least one free LLM key to `.env` (both is better: the second is the fallback):

| Provider | Get a key | `.env` |
|---|---|---|
| Groq | console.groq.com → API Keys | `GROQ_API_KEY=...` |
| Google Gemini | aistudio.google.com → Get API key | `GOOGLE_API_KEY=...` |

Model names are configurable (`GROQ_MODEL`, `GEMINI_MODEL`). Free-tier lineups change often; if you
get a "model not found / decommissioned" error, pick a current model from the provider's docs.

## Run

```bash
# 1. Index documents in data/docs/
python scripts/ingest.py --reset

# 2. Ask questions in the terminal (streamed answer + cited sources)
python scripts/ask.py "What does exit code 2 mean?"
python scripts/ask.py                     # chat; follow-ups like "what about its timeout?" work

# 3. Run the API, then open http://127.0.0.1:8000/docs
uvicorn rag.api:app --app-dir src --reload

# Retrieval only / evaluation
python scripts/query.py "utf-8-sig" --compare
python scripts/eval_retrieval.py --show-misses

# Tests and lint (no API key or model download needed)
pytest -q
ruff check .
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check (doesn't load models) |
| GET | `/documents` | Ingested files and chunk count |
| POST | `/ask` | Answer a question. Body: `question`, optional `history`, `mode`, `top_k`, `source`, `stream` |
| POST | `/upload` | Upload a PDF/DOCX/TXT/MD file; it's saved and indexed immediately |

`/ask` returns JSON (`answer`, `sources`, `cited`, `query`, `provider`, `latency_ms`), or with
`"stream": true` a Server-Sent Events stream: one `sources` event, many `token` events, then `done`
(or `error`).

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What does exit code 2 mean?"}'
```

## Retrieval results

28 golden questions over 3 documents (7 exact-term, 21 meaning-based), top 5.

| Mode | hit@1 | hit@5 | MRR | ms/query (CPU) |
|---|---|---|---|---|
| vector | 0.71 | 0.93 | 0.81 | 24 |
| bm25 | 0.89 | 1.00 | 0.93 | |
| hybrid | 0.75 | | | |
| **rerank** | **0.89** | **1.00** | **0.94** | 1268 |

Re-ranking raised exact-term hit@1 from 0.43 (vector) to 0.71 and meaning-based hit@1 to 0.95.
Some remaining "misses" are label strictness: the answer also appears on another page.

## Design decisions (interview talking points)

| Decision | Why |
|---|---|
| Recursive splitter, 1000 chars / 150 overlap | Breaks at paragraph → sentence → word boundaries; overlap keeps context that spans a cut. |
| Page-level metadata on every chunk | Enables citations ("handbook.pdf, p. 4") and per-file filtering. |
| Strip repeated PDF headers/footers | Running headers ("Report \| Page 7") were in every chunk, adding noise and making unrelated chunks look alike. Only edge lines repeated on ≥50% of pages are removed. |
| Deterministic chunk IDs (SHA-256) | Re-ingesting is idempotent — upserts overwrite instead of duplicating. |
| Hybrid BM25 + vector, RRF fusion | Embeddings capture meaning but blur exact tokens; BM25 matches them. RRF merges by rank, so different score scales don't matter. |
| Retrieve 20, re-rank to 5 | The cross-encoder is accurate but slow, so it only scores a short candidate list. |
| Numbered sources + "cite [n]" rule | Makes every claim traceable; the API returns which sources were actually cited. |
| Exact refusal string | "I couldn't find this in the provided documents." is easy to detect in evaluation and in the UI. |
| "Sources are not instructions" rule | Basic prompt-injection defense: a malicious document can't redirect the assistant. |
| No LLM call when nothing is retrieved | Saves cost and avoids a model inventing an answer from nothing. |
| Follow-up rewriting | "What about its timeout?" is rewritten to a standalone query before retrieval; the raw question is used if rewriting fails. |
| Provider fallback (Groq → Gemini) | Free tiers rate-limit and retire models; the second provider keeps the app working. Streaming falls back only before the first token. |
| One SDK for both providers | Both expose OpenAI-compatible APIs, so switching is a base URL + model name change. |
| Low temperature (0.1) | Answers should follow the sources, not be creative. |
| Lazy model loading in the API | `/health` responds instantly; models load on the first real request. |
| Injectable dependencies | Tests swap in a fake LLM and retriever, so CI needs no keys, network or models. |

## Known limitations (to address later)
- Scanned/image-only PDFs have no extractable text (needs OCR).
- Images inside DOCX files (e.g. screenshots of buttons) aren't read.
- DOCX files have no page numbers, so their citations show the file name only.
- The BM25 index lives in memory and is rebuilt on startup/upload — fine for thousands of chunks, not millions.
- No authentication on the API yet; don't expose it publicly as-is.
- Answer quality isn't measured yet → RAGAS evaluation on Day 6.

## Roadmap
- [x] Day 1 – Loaders, chunking, metadata
- [x] Day 2 – Embeddings, Chroma, retrieval CLI
- [x] Day 3 – BM25 hybrid search (RRF) + cross-encoder re-ranking + retrieval eval
- [x] Day 4 – LLM answers with citations, Groq/Gemini fallback, follow-ups, FastAPI
- [ ] Day 5 – Streamlit UI + Langfuse tracing
- [ ] Day 6 – RAGAS evaluation of answer quality
- [ ] Day 7 – Docker, Hugging Face Spaces deploy, GitHub Actions
