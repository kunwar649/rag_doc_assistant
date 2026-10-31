from rag.bm25 import BM25Index, tokenize
from rag.eval_retrieval import EvalQuestion, first_hit_rank, parse_questions, summarize
from rag.fusion import reciprocal_rank_fusion
from rag.models import Chunk, SearchResult


def _r(cid, source="a.pdf", page=1):
    return SearchResult(text=cid, metadata={"source": source, "page": page}, score=0.0, id=cid)


# --- tokenizer / BM25 -------------------------------------------------------

def test_tokenize_keeps_identifier_and_parts():
    tokens = tokenize("Wait for ctl00_PageContent_dlstLicensure!")
    assert "ctl00_pagecontent_dlstlicensure" in tokens
    assert {"ctl00", "pagecontent", "dlstlicensure"} <= set(tokens)


def test_bm25_finds_exact_identifier():
    chunks = [
        Chunk("1", "General text about browsers and waiting.", {"source": "a.pdf"}),
        Chunk("2", "The result container is ctl00_PageContent_dlstLicensure.", {"source": "a.pdf"}),
        Chunk("3", "CSV output uses utf-8-sig encoding.", {"source": "b.pdf"}),
    ]
    index = BM25Index(chunks)
    assert index.search("dlstLicensure")[0].id == "2"
    assert index.search("utf-8-sig")[0].id == "3"


def test_bm25_source_filter_and_no_match():
    chunks = [Chunk("1", "alpha beta", {"source": "a.pdf"}), Chunk("2", "alpha gamma", {"source": "b.pdf"})]
    index = BM25Index(chunks)
    assert [r.id for r in index.search("alpha", source="b.pdf")] == ["2"]
    assert index.search("nothing here") == []


def test_bm25_empty_index():
    assert BM25Index([]).search("anything") == []


# --- RRF ---------------------------------------------------------------------

def test_rrf_rewards_agreement():
    vector = [_r("a"), _r("b"), _r("c")]
    keyword = [_r("c"), _r("b"), _r("d")]
    fused = reciprocal_rank_fusion([vector, keyword], k=60)
    # b and c appear in both lists, so they beat a and d which appear once
    assert {fused[0].id, fused[1].id} == {"b", "c"}
    assert {r.id for r in fused} == {"a", "b", "c", "d"}


def test_rrf_top_k():
    assert len(reciprocal_rank_fusion([[_r("a"), _r("b"), _r("c")]], top_k=2)) == 2


# --- evaluation helpers ------------------------------------------------------

def test_parse_questions(tmp_path):
    f = tmp_path / "q.md"
    f.write_text(
        "# title\n\n"
        "Q1: First?\nA: one\nSource: doc.pdf, p. 4 and p. 8\nType: exact\n\n"
        "Q2: Second?\nA: two\nSource: notes.docx\nType: meaning\n\n"
        "Q3: Unknown?\nA: Not stated.\nType: negative\n",
        encoding="utf-8",
    )
    qs = parse_questions(f)
    assert [q.qid for q in qs] == ["Q1", "Q2"]  # negative skipped
    assert qs[0].source == "doc.pdf" and qs[0].pages == {4, 8} and qs[0].qtype == "exact"
    assert qs[1].source == "notes.docx" and qs[1].pages == set()


def test_first_hit_rank_and_summary():
    q = EvalQuestion("Q1", "?", "", "a.pdf", {2})
    results = [_r("x", "a.pdf", 1), _r("y", "b.pdf", 2), _r("z", "a.pdf", 2)]
    assert first_hit_rank(results, q) == 3
    assert first_hit_rank(results[:2], q) is None
    m = summarize([1, 2, None, 1], k=5)
    assert m["hit@1"] == 0.5 and m["hit@5"] == 0.75
    assert abs(m["mrr"] - (1 + 0.5 + 0 + 1) / 4) < 1e-9
