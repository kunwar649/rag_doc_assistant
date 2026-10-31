import pytest
from fakes import FakeRetriever, connection_error, fake_client

from rag.config import Settings
from rag.llm import LLMClient, LLMUnavailableError, Provider
from rag.models import SearchResult
from rag.pipeline import RAGPipeline
from rag.prompts import NO_ANSWER, build_answer_messages, extract_citations


def _llm(*replies):
    return LLMClient([Provider(f"p{i}", f"m{i}", fake_client(r)) for i, r in enumerate(replies)])


# --- prompts -----------------------------------------------------------------

def test_answer_messages_number_sources_and_keep_history():
    results = [SearchResult("alpha", {"source": "a.pdf", "page": 2}, 1.0, "x")]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    msgs = build_answer_messages("Q?", results, history)
    assert msgs[0]["role"] == "system" and "ONLY" in msgs[0]["content"]
    assert msgs[1:3] == history
    assert "[1] (a.pdf (p. 2))\nalpha" in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("Question: Q?")


def test_extract_citations_dedupes_and_drops_out_of_range():
    assert extract_citations("A [2]. B [1][2]. C [9].", n_sources=3) == [2, 1]
    assert extract_citations("no citations", n_sources=3) == []


# --- LLM fallback ------------------------------------------------------------

def test_complete_falls_back_to_next_provider():
    llm = _llm(connection_error(), "from backup")
    assert llm.complete([{"role": "user", "content": "q"}]) == "from backup"
    assert llm.last_provider == "p1:m1"


def test_complete_raises_when_all_fail():
    with pytest.raises(LLMUnavailableError):
        _llm(connection_error(), connection_error()).complete([])


def test_stream_skips_empty_chunks_and_falls_back():
    llm = _llm(connection_error(), "hello world")
    assert "".join(llm.stream([])) == "hello world "
    assert llm.last_provider == "p1:m1"


def test_no_providers_configured():
    with pytest.raises(LLMUnavailableError):
        LLMClient.from_settings(Settings(groq_api_key="", google_api_key="", _env_file=None))


def test_gpt_oss_gets_low_reasoning_effort():
    llm = LLMClient.from_settings(Settings(groq_api_key="k", groq_model="openai/gpt-oss-20b", google_api_key="g", _env_file=None))
    assert [p.name for p in llm.providers] == ["groq", "gemini"]
    assert llm.providers[0].extra == {"reasoning_effort": "low"}
    assert llm.providers[1].extra == {}


# --- pipeline ----------------------------------------------------------------

def _pipeline(retriever=None, llm=None):
    return RAGPipeline(Settings(_env_file=None), retriever=retriever or FakeRetriever(), llm=llm or _llm("Exit code 2 means no table [1]."))


def test_answer_returns_text_sources_and_citations():
    result = _pipeline().answer("What does exit code 2 mean?")
    assert result.answer == "Exit code 2 means no table [1]."
    assert result.cited == [1]
    assert [s["n"] for s in result.sources] == [1, 2]
    assert result.sources[0]["citation"] == "a.pdf (p. 7)"
    assert result.provider == "p0:m0"


def test_no_results_skips_llm():
    llm = _llm("should not be called")
    result = _pipeline(retriever=FakeRetriever(results=[]), llm=llm).answer("anything")
    assert result.answer == NO_ANSWER and result.sources == []
    assert llm.providers[0].client.chat.completions.calls == []


def test_follow_up_is_rewritten_before_retrieval():
    retriever = FakeRetriever()
    # first call = query rewrite, second = answer (same fake reply for both, fine here)
    llm = LLMClient([Provider("p", "m", fake_client("What is the default output path of the script?"))])
    history = [{"role": "user", "content": "What does the script write?"}, {"role": "assistant", "content": "A CSV [1]."}]
    result = _pipeline(retriever, llm).answer("Where is it saved?", history=history)
    assert retriever.queries == ["What is the default output path of the script?"]
    assert result.query == "What is the default output path of the script?"


def test_rewrite_failure_falls_back_to_original_question():
    retriever = FakeRetriever()
    p = _pipeline(retriever, _llm("answer [1]"))
    p.llm = LLMClient([Provider("bad", "m", fake_client(connection_error()))])
    assert p.rewrite_query("Where is it saved?", [{"role": "user", "content": "x"}]) == "Where is it saved?"


def test_history_is_trimmed():
    p = _pipeline()
    p.s.history_turns = 1
    history = [{"role": "user", "content": str(i)} for i in range(6)] + [{"role": "system", "content": "ignore"}]
    assert [m["content"] for m in p._trim_history(history)] == ["4", "5"]


def test_stream_event_order():
    events = list(_pipeline().stream("What does exit code 2 mean?"))
    assert events[0]["type"] == "sources" and len(events[0]["sources"]) == 2
    assert all(e["type"] == "token" for e in events[1:-1])
    assert events[-1]["type"] == "done" and events[-1]["cited"] == [1]


def test_stream_reports_error_event():
    events = list(_pipeline(llm=LLMClient([Provider("bad", "m", fake_client(connection_error()))])).stream("q"))
    assert [e["type"] for e in events] == ["sources", "error"]


def test_special_spaces_and_hyphens_are_normalized():
    from rag.llm import clean_text

    assert clean_text("exit code 2 and re‑rank now") == "exit code 2 and re-rank now"
    llm = LLMClient([Provider("p", "m", fake_client("x", tokens=["Exit code ", "2 means [1]."]))])
    assert "".join(llm.stream([])) == "Exit code 2 means [1]."
    assert _llm("code 0").complete([]) == "code 0"
