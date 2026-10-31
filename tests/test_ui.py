from pathlib import Path

import pytest

from rag.ui import escape_markdown, format_meta, order_sources, source_label

SOURCES = [
    {"n": 1, "citation": "a.pdf (p. 4)", "score": 4.3, "text": "x"},
    {"n": 2, "citation": "a.pdf (p. 7)", "score": 1.54, "text": "y"},
    {"n": 3, "citation": "b.docx", "score": -2.98, "text": "z"},
]


def test_escape_markdown_dollars():
    assert escape_markdown("costs $5 and $10") == "costs \\$5 and \\$10"


def test_order_sources_puts_cited_first():
    assert [s["n"] for s in order_sources(SOURCES, cited=[3, 1])] == [3, 1, 2]
    assert [s["n"] for s in order_sources(SOURCES, cited=[])] == [1, 2, 3]


def test_source_label():
    assert source_label(SOURCES[0], cited=[1]) == "✅ cited · [1] a.pdf (p. 4) · score 4.30"
    assert source_label(SOURCES[2], cited=[1]) == "[3] b.docx · score -2.98"


def test_format_meta():
    assert format_meta("groq:m", 4975, {"input": 1234, "output": 56}) == "groq:m · 5.0 s · 1,234 in / 56 out tokens"
    assert format_meta(None, 120, None) == "no LLM call · 0.1 s"


# --- the Streamlit app itself, with a fake pipeline -----------------------------

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app(monkeypatch):
    pytest.importorskip("streamlit")
    import streamlit as st
    from fakes import FakeRetriever, fake_client
    from streamlit.testing.v1 import AppTest

    import rag.pipeline
    from rag.config import Settings
    from rag.llm import LLMClient, Provider
    from rag.tracing import Tracer

    real_pipeline = rag.pipeline.RAGPipeline

    def fake_pipeline(_settings):
        # app.py does `from rag.pipeline import RAGPipeline` at call time, so it picks this up.
        llm = LLMClient([Provider("fake", "m", fake_client("Exit code 2 means no table [1]."))])
        return real_pipeline(Settings(_env_file=None), retriever=FakeRetriever(), llm=llm, tracer=Tracer())

    monkeypatch.setattr(rag.pipeline, "RAGPipeline", fake_pipeline)
    st.cache_resource.clear()
    at = AppTest.from_file(str(APP), default_timeout=30)
    yield at
    st.cache_resource.clear()


def test_app_loads_and_answers(app):
    app.run()
    assert not app.exception
    assert "1 file(s) · 3 chunks" in [c.value for c in app.sidebar.caption]

    app.chat_input[0].set_value("What does exit code 2 mean?").run()
    assert not app.exception
    assert any("Exit code 2 means no table [1]." in m.value for m in app.markdown)
    labels = [e.label for e in app.expander]
    assert labels[0].startswith("✅ cited · [1]") and len(labels) == 2
    assert any("fake:m" in c.value for c in app.caption)

    # follow-up keeps the conversation and is rendered from history on rerun
    app.chat_input[0].set_value("And exit code 0?").run()
    assert not app.exception
    assert len(app.chat_message) == 4
