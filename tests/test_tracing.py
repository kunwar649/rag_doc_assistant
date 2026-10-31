from types import SimpleNamespace

from fakes import FakeRetriever, connection_error, fake_client

from rag.config import Settings
from rag.llm import LLMClient, Provider, _chunk_usage
from rag.pipeline import RAGPipeline
from rag.tracing import NOOP, Tracer, create_tracer


class FakeObs:
    """Records what the pipeline sends to Langfuse."""

    def __init__(self, log, name, as_type="span", **kwargs):
        self.log, self.name, self.as_type = log, name, as_type
        self.data = dict(kwargs)
        self.ended = False
        log.append(self)

    def start_observation(self, name, as_type="span", **kwargs):
        return FakeObs(self.log, name, as_type, **kwargs)

    def update(self, **kwargs):
        self.data.update(kwargs)

    def end(self):
        self.ended = True


class FakeLangfuse:
    def __init__(self):
        self.log = []

    def start_observation(self, name, as_type="span", **kwargs):
        return FakeObs(self.log, name, as_type, **kwargs)

    def flush(self):
        pass


class BrokenLangfuse:
    def start_observation(self, **kwargs):
        raise RuntimeError("SDK changed")

    def flush(self):
        raise RuntimeError("network down")


def _pipeline(tracer, llm=None):
    llm = llm or LLMClient([Provider("groq", "gpt-oss", fake_client("Exit code 2 means no table [1]."))])
    return RAGPipeline(Settings(_env_file=None), retriever=FakeRetriever(), llm=llm, tracer=tracer)


def test_tracing_disabled_without_keys():
    assert not create_tracer(Settings(langfuse_public_key="", langfuse_secret_key="", _env_file=None)).enabled
    assert Tracer().start("x") is NOOP


def test_trace_has_retrieve_and_generate_with_model():
    lf = FakeLangfuse()
    _pipeline(Tracer(lf)).answer("What does exit code 2 mean?")
    names = [o.name for o in lf.log]
    assert names == ["rag-query", "retrieve", "generate"]
    root, retrieve, gen = lf.log
    assert root.data["input"]["question"] == "What does exit code 2 mean?"
    assert root.data["output"]["cited"] == [1]
    assert retrieve.data["output"][0]["citation"] == "a.pdf (p. 7)"
    assert gen.as_type == "generation" and gen.data["model"] == "gpt-oss"
    assert all(o.ended for o in lf.log)


def test_follow_up_adds_rewrite_generation():
    lf = FakeLangfuse()
    history = [{"role": "user", "content": "What does it write?"}, {"role": "assistant", "content": "A CSV [1]."}]
    _pipeline(Tracer(lf)).answer("Where is it saved?", history=history)
    assert [o.name for o in lf.log] == ["rag-query", "rewrite", "retrieve", "generate"]


def test_stream_trace_ends_and_records_errors():
    lf = FakeLangfuse()
    list(_pipeline(Tracer(lf)).stream("q"))
    assert [o.name for o in lf.log] == ["rag-query", "retrieve", "generate"] and all(o.ended for o in lf.log)

    lf = FakeLangfuse()
    bad = LLMClient([Provider("bad", "m", fake_client(connection_error()))])
    list(_pipeline(Tracer(lf), llm=bad).stream("q"))
    assert lf.log[0].data["level"] == "ERROR" and all(o.ended for o in lf.log)


def test_stream_closed_early_still_ends_trace():
    lf = FakeLangfuse()
    events = _pipeline(Tracer(lf)).stream("q")
    next(events)  # sources only, then the client disconnects
    events.close()
    assert lf.log[0].ended and lf.log[0].data["level"] == "WARNING"


def test_broken_sdk_never_breaks_answers():
    result = _pipeline(Tracer(BrokenLangfuse())).answer("q")
    assert result.answer == "Exit code 2 means no table [1]."
    Tracer(BrokenLangfuse()).flush()  # must not raise


# --- token usage ---------------------------------------------------------------

def test_usage_from_openai_style_and_groq_chunks():
    openai_style = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30), choices=[])
    groq_style = SimpleNamespace(usage=None, x_groq={"usage": {"prompt_tokens": 50, "completion_tokens": 7}}, choices=[])
    assert _chunk_usage(openai_style) == {"input": 120, "output": 30}
    assert _chunk_usage(groq_style) == {"input": 50, "output": 7}
    assert _chunk_usage(SimpleNamespace(usage=None, choices=[])) is None


def test_complete_records_usage():
    client = fake_client("hi")
    original = client.chat.completions.create

    def create(**kwargs):
        resp = original(**kwargs)
        resp.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
        return resp

    client.chat.completions.create = create
    llm = LLMClient([Provider("p", "m", client)])
    llm.complete([])
    assert llm.last_usage == {"input": 10, "output": 2}
