"""Pure display helpers for the Streamlit app (kept separate so they can be unit-tested)."""


def escape_markdown(text: str) -> str:
    """Streamlit renders $...$ as LaTeX math; documents with dollar signs would turn into formulas."""
    return text.replace("$", "\\$")


def order_sources(sources: list[dict], cited: list[int]) -> list[dict]:
    """Cited sources first, in the order the answer cites them, then the rest by rank."""
    by_n = {s["n"]: s for s in sources}
    first = [by_n[n] for n in cited if n in by_n]
    return first + [s for s in sources if s["n"] not in cited]


def source_label(source: dict, cited: list[int]) -> str:
    mark = "✅ cited · " if source["n"] in cited else ""
    return f"{mark}[{source['n']}] {source['citation']} · score {source['score']:.2f}"


def format_meta(provider: str | None, latency_ms: int | None, usage: dict | None) -> str:
    parts = [provider or "no LLM call"]
    if latency_ms is not None:
        parts.append(f"{latency_ms / 1000:.1f} s")
    if usage:
        parts.append(f"{usage['input']:,} in / {usage['output']:,} out tokens")
    return " · ".join(parts)
