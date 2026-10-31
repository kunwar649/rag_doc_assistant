# RAG Document Assistant

Chat with your documents (PDF, DOCX, TXT, MD) and get answers with citations.
Built entirely on free tools: local embeddings, embedded vector DB, free-tier LLMs.

> **Status:** Day 3 complete — hybrid search (BM25 + vector) with re-ranking, plus a retrieval evaluation script.
> Next: LLM answers with citations + FastAPI (Day 4).

## Architecture

```
INGEST (scripts/ingest.py)
data/docs/*.pdf|docx|txt
   → loaders.py      extract text, keep {source, page}
   → chunking.py     recursive split (1000 chars, 150 overlap), deterministic IDs
   → embeddings.py   bge-small-en-v1.5 (local CPU), normalized vectors
   → vector_store.py ChromaDB (persistent, cosine / HNSW)

RETRIEVE (src/rag/retriever.py)
question ─┬─► vector search (top 20) ─┐
          │                           ├─► RRF fusion ─► cross-encoder re-rank ─► top 5 with citations
          └─► BM25 keyword (top 20) ──┘
```

## Project structure

```
rag-doc-assistant/
├── src/rag/
│   ├── config.py          # settings from .env (pydantic-settings)
│   ├── models.py          # Page, Chunk, SearchResult
│   ├── loaders.py         # PDF / DOCX (incl. tables) / TXT loaders
│   ├── chunking.py        # recursive splitter + stable chunk IDs
│   ├── embeddings.py      # sentence-transformers bi-encoder wrapper
│   ├── vector_store.py    # ChromaDB wrapper: upsert, search, delete
│   ├── bm25.py            # keyword index with identifier-aware tokenizer
│   ├── fusion.py          # Reciprocal Rank Fusion
│   ├── reranker.py        # cross-encoder re-ranking
│   ├── retriever.py       # one entry point: vector | bm25 | hybrid | rerank
│   └── eval_retrieval.py  # golden-question parser + hit@k / MRR metrics
├── scripts/
│   ├── ingest.py          # load → chunk → embed → store
│   ├── query.py           # CLI retrieval, single mode or --compare
│   └── eval_retrieval.py  # score every mode against golden questions
├── tests/                 # pytest unit tests
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

`sentence-transformers` installs PyTorch (~a few hundred MB on CPU). To keep it smaller:
`pip install torch --index-url https://download.pytorch.org/whl/cpu` before the requirements.

## Run

```bash
# 1. Add documents to data/docs/, then ingest (first run downloads the embedding model, ~130 MB)
python scripts/ingest.py

# 2. Retrieve (default mode is rerank; first rerank run downloads the cross-encoder, ~90 MB)
python scripts/query.py "What does exit code 2 mean?"
python scripts/query.py "utf-8-sig" --mode bm25
python scripts/query.py "ctl00_PageContent_dlstLicensure" --compare   # every mode side by side
python scripts/query.py                                               # interactive mode

# 3. Evaluate retrieval against golden questions (data/eval_questions.md)
python scripts/eval_retrieval.py
python scripts/eval_retrieval.py --show-misses

# Tests and lint
pytest -q
ruff check .
```

### Retrieval modes

| Mode | What it does | Score shown |
|---|---|---|
| `vector` | Embedding similarity only | Cosine similarity |
| `bm25` | Keyword matching only | BM25 score (unbounded) |
| `hybrid` | Vector + BM25 merged with RRF | RRF score (only the order matters) |
| `rerank` | Hybrid top 20 re-scored by a cross-encoder | Relevance logit (can be negative) |

### Golden question format

```
Q1: What is the default output path for the results file?
A: output/tennessee_license_results.csv
Source: Tennessee_Adapter_Detailed_Explanation.pdf, p. 7
Type: meaning          # meaning | exact | negative (negative = not in docs, skipped here)
```
A retrieved chunk counts as correct if it comes from the listed file and page.

## Results

Final Day 3 results
Mode	hit@1	hit@5	MRR	ms/query
vector	0.71	0.93	0.81	24
bm25	0.89	1.00	0.93*	–
hybrid	0.75	(from your full run)	(from your full run)	–
rerank	0.89	1.00	0.94	1268

## Design decisions (interview talking points)

| Decision | Why |
|---|---|
| Recursive splitter, 1000 chars / 150 overlap | Breaks at paragraph → sentence → word boundaries; overlap keeps context that spans a cut. Tuned with eval data. |
| Page-level metadata on every chunk | Enables citations ("handbook.pdf, p. 4") and per-file filtering. |
| Deterministic chunk IDs (SHA-256) | Re-ingesting is idempotent — upserts overwrite instead of duplicating. |
| `bge-small-en-v1.5` | Strong quality for its size, runs on CPU, free. Uses BGE's query instruction prefix. |
| Hybrid BM25 + vector | Embeddings capture meaning but blur exact tokens (IDs, codes); BM25 matches them exactly. |
| Identifier-aware tokenizer | `ctl00_PageContent_txtLicense` is indexed whole *and* split into parts, so full and partial lookups both match. |
| RRF instead of adding scores | Cosine (~0.4–0.9) and BM25 (0–30+) are on different scales; RRF uses ranks only, so neither dominates. |
| Retrieve 20, re-rank to 5 | The cross-encoder reads query + chunk together (accurate but slow), so it only sees a short candidate list. |
| BM25 index rebuilt in memory from Chroma | Single source of truth, no second store to sync. Fine for thousands of chunks; use a search engine at millions. |
| Retrieval eval separate from answer eval | hit@k / MRR show whether the right context is found before any LLM is involved, so failures can be localized. |
| One bad file doesn't stop ingest | Errors are logged per file; real document sets always contain a few broken files. |

## Known limitations (to address later)
- Scanned/image-only PDFs have no extractable text (needs OCR).
- Images inside DOCX files (e.g. screenshots of buttons) aren't read.
- DOCX files have no page numbers, so their citations show the file name only.
- Golden questions written from the documents' own wording favour keyword search; add paraphrased questions for a fair comparison.
- No answer generation yet → LLM integration on Day 4.

## Roadmap
- [x] Day 1 – Loaders, chunking, metadata
- [x] Day 2 – Embeddings, Chroma, retrieval CLI
- [x] Day 3 – BM25 hybrid search (RRF) + cross-encoder re-ranking + retrieval eval
- [ ] Day 4 – Groq/Gemini answers with citations, FastAPI `/upload` `/ask`
- [ ] Day 5 – Streamlit UI + Langfuse tracing
- [ ] Day 6 – RAGAS evaluation of answer quality
- [ ] Day 7 – Docker, Hugging Face Spaces deploy, GitHub Actions
