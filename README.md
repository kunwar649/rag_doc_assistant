# RAG Document Assistant

Chat with your documents (PDF, DOCX, TXT, MD) and get grounded answers with citations.
Built entirely on free tools: local embeddings and re-ranker, embedded vector DB, free-tier LLMs.

> **Status:** Day 5 complete — Next.js frontend with an evidence pane, Streamlit UI, Langfuse tracing, token usage.
> Next: answer-quality evaluation with RAGAS (Day 6).

## Architecture

```
INGEST (scripts/ingest.py, POST /upload)
data/docs/*.pdf|docx|txt|md
   → loaders.py      extract text, keep {source, page}, strip repeated PDF headers/footers
   → chunking.py     recursive split (500 chars, 80 overlap), deterministic IDs
   → embeddings.py   bge-small-en-v1.5 (local CPU), normalized vectors
   → vector_store.py ChromaDB (persistent, cosine / HNSW)

ANSWER (src/rag/pipeline.py, POST /ask)
question + history
   → rewrite follow-up into a standalone query (LLM, only when there is history)
   → retrieve:  vector (top 20) ┐
                BM25   (top 20) ┴─► RRF fusion ─► cross-encoder re-rank ─► top 5
   → prompt: rules + numbered sources [1]..[5] + question
   → LLM: Groq (gpt-oss-20b) ──fails?──► Gemini (fallback)
   → answer with [n] citations, cited sources, provider, latency, token usage

WEB (frontend/)    Next.js + React → POST /ask (SSE) → streamed answer, citations open the evidence pane
UI (app.py)        Streamlit chat → pipeline.stream() → streamed answer + expandable sources
TRACING (tracing.py)  rag-query ─┬─ rewrite (generation, follow-ups only)
                                 ├─ retrieve (span: query → pages + scores)
                                 └─ generate (generation: prompt → answer, model, tokens)
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
│   ├── tracing.py         # optional Langfuse tracing that can't break the app
│   ├── ui.py              # display helpers for the Streamlit app
│   └── eval_retrieval.py  # golden-question parser + hit@k / MRR metrics
├── scripts/
│   ├── ingest.py          # index everything in data/docs
│   ├── query.py           # retrieval only, single mode or --compare
│   ├── ask.py             # terminal chat with streamed, cited answers
│   └── eval_retrieval.py  # score every retrieval mode against golden questions
├── frontend/              # Next.js + React + TypeScript web app (see frontend/README.md)
├── app.py                 # Streamlit chat UI (quick alternative to the web app)
├── .streamlit/config.toml # upload limit, hidden dev toolbar
├── tests/                 # pytest: unit, API and Streamlit app tests (fake LLM, no keys needed)
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

Optional tracing: create a free project at cloud.langfuse.com → Settings → API Keys, then add
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (and `LANGFUSE_HOST=https://us.cloud.langfuse.com` for the US region).

Model names are configurable (`GROQ_MODEL`, `GEMINI_MODEL`). Free-tier lineups change often; if you
get a "model not found / decommissioned" error, pick a current model from the provider's docs.

## Run

```bash
# 1. Index documents in data/docs/
python scripts/ingest.py --reset

# 2. Web app: start the API, then the Next.js frontend (needs Node.js 20.9+)
uvicorn rag.api:app --app-dir src --reload --reload-dir src   # terminal 1
cd frontend && npm install && npm run dev           # terminal 2 → http://localhost:3000

# Or the Streamlit UI in one command (http://localhost:8501)
streamlit run app.py

# Or ask in the terminal (streamed answer + cited sources)
python scripts/ask.py "What does exit code 2 mean?"
python scripts/ask.py                     # chat; follow-ups like "what about its timeout?" work

# 3. Run the API, then open http://127.0.0.1:8000/docs
uvicorn rag.api:app --app-dir src --reload --reload-dir src

# Retrieval only / evaluation
python scripts/query.py "utf-8-sig" --compare
python scripts/eval_retrieval.py --show-misses

# Tests and lint (no API key or model download needed)
pytest -q
ruff check .
cd frontend && npm test && npm run lint
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check (doesn't load models) |
| GET | `/documents` | Ingested files and chunk count |
| GET | `/config` | Defaults the frontend needs: `top_k`, upload limit, file types, tracing on/off |
| POST | `/ask` | Answer a question. Body: `question`, optional `history`, `mode`, `top_k`, `source`, `stream` |
| POST | `/upload` | Upload a PDF/DOCX/TXT/MD file; it's saved and indexed immediately |

CORS is enabled for the origins in `CORS_ORIGINS` (default: the Next.js dev server on port 3000).

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
| vector | 0.64 | 0.89 | 0.74 | 24 |
| bm25 | 0.93 | 1.00 | 0.95 | <1 |
| hybrid | 0.79 | 0.96 | 0.85 | 24 |
| **rerank** | **0.93** | **1.00** | **0.96** | 982 |

Re-ranking raised exact-term hit@1 from 0.43 (vector only) to 0.86 and meaning-based hit@1 from 0.71 to 0.95,
with the correct chunk in the top 5 for every question. BM25 scores as well as re-ranking here partly because
the questions reuse the documents' wording; re-ranking is the safer default for real users' phrasing.

## Design decisions (interview talking points)

| Decision | Why |
|---|---|
| Recursive splitter, 500 chars / 80 overlap | Breaks at paragraph → sentence → word boundaries; overlap keeps context that spans a cut. Chosen over 1000/150 after comparing retrieval on test questions. |
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
| Streamlit UI over the same pipeline | The UI, CLI and API all call `RAGPipeline.stream()`, so behaviour is identical everywhere. |
| Cited sources first, expandable | Users can verify any claim in one click; uncited sources stay available but out of the way. |
| Spinner until the first token | Retrieval + re-ranking take ~1 s before generation starts; the spinner shows the app is working. |
| Tracing per step (Langfuse) | Shows where time and tokens go (rewrite vs retrieval vs generation) for each question. |
| Tracing can't break answers | All SDK calls are wrapped; if Langfuse is down, misconfigured or changes its API, the app keeps answering. |
| Explicit spans, not context managers | FastAPI streams from worker threads, where thread-local "current span" state is lost. |
| Next.js frontend over the same API | The browser streams `/ask` with fetch + ReadableStream (EventSource can't POST) and parses SSE itself; the Python pipeline is unchanged. |
| Evidence pane | The product's job is trustworthy answers, so checking a claim is one click: citations open the passage, highlighted, beside the answer. |
| Browser-only render for the chat | Saved history lives in localStorage; rendering it only in the browser avoids server/browser mismatches. |
| CORS allow-list, not `*` | Only the configured frontend origins can call the API from a browser. |
| Fonts bundled from npm | No runtime or build-time requests to Google Fonts; works offline and behind strict networks. |
| Token usage captured from streams | Groq reports usage on the last chunk (`x_groq.usage`); it's shown in the UI and sent to Langfuse for cost tracking. |

## Known limitations (to address later)
- Scanned/image-only PDFs have no extractable text (needs OCR).
- Images inside DOCX files (e.g. screenshots of buttons) aren't read.
- DOCX files have no page numbers, so their citations show the file name only.
- The BM25 index lives in memory and is rebuilt on startup/upload — fine for thousands of chunks, not millions.
- No authentication on the API or UI yet; don't expose them publicly as-is.
- Chat history lives in the browser session only; refreshing the page clears it.
- Answer quality isn't measured yet → RAGAS evaluation on Day 6.

## Roadmap
- [x] Day 1 – Loaders, chunking, metadata
- [x] Day 2 – Embeddings, Chroma, retrieval CLI
- [x] Day 3 – BM25 hybrid search (RRF) + cross-encoder re-ranking + retrieval eval
- [x] Day 4 – LLM answers with citations, Groq/Gemini fallback, follow-ups, FastAPI
- [x] Day 5 – Streamlit UI + Langfuse tracing
- [ ] Day 6 – RAGAS evaluation of answer quality
- [ ] Day 7 – Docker, Hugging Face Spaces deploy, GitHub Actions
