"""Streamlit chat UI for the RAG document assistant.

Run from the project folder:
    streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from rag.config import get_settings
from rag.llm import LLMUnavailableError
from rag.loaders import SUPPORTED_EXTENSIONS
from rag.retriever import MODES
from rag.ui import escape_markdown, format_meta, order_sources, source_label

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")

MODE_HELP = {
    "rerank": "Hybrid search, then a cross-encoder re-orders the top 20. Most accurate, ~1 s slower.",
    "hybrid": "Vector + keyword results merged with Reciprocal Rank Fusion.",
    "vector": "Embedding similarity only: good for meaning, weak on exact terms.",
    "bm25": "Keyword matching only: good for IDs and exact phrases, weak on rewording.",
}


@st.cache_resource(show_spinner="Loading models (the first run downloads them)...")
def get_pipeline():
    from rag.pipeline import RAGPipeline

    return RAGPipeline(get_settings())


def notify(level: str, text: str) -> None:
    """Queue a message to show after st.rerun() (messages drawn before a rerun are lost)."""
    st.session_state.setdefault("notices", []).append((level, text))


def index_uploads(pipeline, files) -> None:
    from rag.ingestion import ingest_paths

    s = get_settings()
    s.docs_dir.mkdir(parents=True, exist_ok=True)
    limit = s.max_upload_mb * 1024 * 1024
    saved = []
    for f in files:
        name = Path(f.name).name  # never trust directory parts in an uploaded name
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            notify("warning", f"Skipped {name}: unsupported file type.")
            continue
        if f.size > limit:
            notify("warning", f"Skipped {name}: larger than {s.max_upload_mb} MB.")
            continue
        path = s.docs_dir / name
        path.write_bytes(f.getvalue())
        saved.append(path)
    if not saved:
        return

    with st.spinner(f"Indexing {len(saved)} file(s)..."):
        report = ingest_paths(saved, s, pipeline.retriever.store)
        pipeline.retriever.refresh()  # make new chunks searchable by keyword too
    for name in report.failed:
        (s.docs_dir / name).unlink(missing_ok=True)
        notify("warning", f"No text could be extracted from {name} (scanned PDF or empty file?).")
    if report.files:
        notify("success", f"Indexed {', '.join(report.files)}: {report.chunks} chunks.")


def render_sources(sources: list[dict], cited: list[int]) -> None:
    if not sources:
        return
    st.caption("Sources: click one to see the text the answer was based on.")
    for s in order_sources(sources, cited):
        with st.expander(source_label(s, cited)):
            st.text(s["text"])


def render_footer(msg: dict) -> None:
    if msg.get("query") and msg["query"] != msg.get("question"):
        st.caption(f"Searched for: {msg['query']}")
    st.caption(format_meta(msg.get("provider"), msg.get("latency_ms"), msg.get("usage")))


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(escape_markdown(msg["content"]))
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []), msg.get("cited", []))
            render_footer(msg)


# --- startup -----------------------------------------------------------------

try:
    pipeline = get_pipeline()
except LLMUnavailableError as e:
    st.error(f"{e}\n\nAdd a key to the .env file in the project folder, then restart the app.")
    st.stop()

settings = get_settings()
store = pipeline.retriever.store
sources_available = store.list_sources()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("📄 Documents")
    for level, text in st.session_state.pop("notices", []):
        getattr(st, level)(text)
    st.caption(f"{len(sources_available)} file(s) · {store.count()} chunks")
    for name in sources_available:
        st.markdown(f"- {escape_markdown(name)}")

    with st.form("upload", clear_on_submit=True):
        uploads = st.file_uploader(
            "Add documents",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
        )
        if st.form_submit_button("Upload and index") and uploads:
            index_uploads(pipeline, uploads)
            st.rerun()

    st.header("⚙️ Settings")
    mode = st.selectbox("Retrieval mode", MODES, index=MODES.index("rerank"), help=MODE_HELP["rerank"])
    st.caption(MODE_HELP[mode])
    top_k = st.slider("Sources per answer", min_value=1, max_value=10, value=settings.top_k)
    scope = st.selectbox("Search in", ["All documents", *sources_available])
    source_filter = None if scope == "All documents" else scope

    st.caption("Tracing: " + ("Langfuse on" if pipeline.tracer.enabled else "off (no Langfuse keys)"))
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- chat ----------------------------------------------------------------------

st.title("Chat with your documents")
if not sources_available:
    st.info("No documents indexed yet. Upload a PDF, DOCX, TXT or MD file in the sidebar.")

for msg in st.session_state.messages:
    render_message(msg)

question = st.chat_input("Ask a question about your documents", disabled=not sources_available)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    render_message({"role": "user", "content": question})

    reply = {"role": "assistant", "question": question, "content": "", "sources": [], "cited": []}
    with st.chat_message("assistant"):
        parts: list[str] = []

        def tokens():
            for event in pipeline.stream(question, history=history, mode=mode, top_k=top_k, source=source_filter):
                kind = event["type"]
                if kind == "sources":
                    reply["sources"], reply["query"] = event["sources"], event["query"]
                elif kind == "token":
                    parts.append(event["text"])
                    yield escape_markdown(event["text"])
                elif kind == "done":
                    reply.update(cited=event["cited"], provider=event["provider"],
                                 latency_ms=event["latency_ms"], usage=event.get("usage"))
                elif kind == "error":
                    reply["error"] = event["message"]

        def with_first(first, rest):
            if first is not None:
                yield first
            yield from rest

        # Retrieval and re-ranking happen before the first token arrives, so show a spinner
        # until then, then stream the rest of the answer as it's generated.
        with st.spinner("Searching your documents..."):
            stream = tokens()
            first = next(stream, None)
        st.write_stream(with_first(first, stream))
        reply["content"] = "".join(parts)

        if reply.get("error"):
            st.error(f"Answer generation failed: {reply['error']}")
        render_sources(reply["sources"], reply["cited"])
        render_footer(reply)

    if reply["content"]:
        st.session_state.messages.append(reply)
