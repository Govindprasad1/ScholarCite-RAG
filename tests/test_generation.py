"""
Stage 3 tests — basic retrieve-and-generate chain.
These tests make a REAL call to the Groq API, so they need GROQ_API_KEY
set and will fail/skip without internet access. Kept separate from
test_retrieval.py since they're slower and depend on an external service.
"""
import os
import shutil
import pytest

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval import vector_store as vs
from src.llm.generate import answer_question

SAMPLE_PDF = "tests/fixtures/sample.pdf"

requires_groq_key = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live LLM tests",
)


@pytest.fixture
def indexed_sample_doc(tmp_path, monkeypatch):
    """Ingest and index the sample PDF into an isolated temp vector store."""
    import src.utils.config as config_module

    original_load_config = config_module.load_config

    def patched_load_config():
        cfg = original_load_config()
        cfg["vector_store"] = dict(cfg["vector_store"])
        cfg["vector_store"]["persist_directory"] = str(tmp_path / "chroma_test")
        cfg["vector_store"]["collection_name"] = "test_generation"
        return cfg

    monkeypatch.setattr(vs, "load_config", patched_load_config)

    pages = extract_pages(SAMPLE_PDF)
    chunks = chunk_pages(pages)
    vs.add_chunks(chunks, doc_id="gen_test_doc")

    yield "gen_test_doc"
    shutil.rmtree(tmp_path / "chroma_test", ignore_errors=True)


@requires_groq_key
def test_answer_question_returns_answer_and_sources(indexed_sample_doc):
    result = answer_question("What method does the paper propose?", doc_id=indexed_sample_doc)
    assert "answer" in result
    assert "source_chunks" in result
    assert len(result["source_chunks"]) > 0
    assert result["answer"].strip() != ""


@requires_groq_key
def test_answer_includes_citation(indexed_sample_doc):
    result = answer_question("What method does the paper propose?", doc_id=indexed_sample_doc)
    assert "Page" in result["answer"]  # should cite at least one page


@requires_groq_key
def test_unanswerable_question_is_refused(indexed_sample_doc):
    result = answer_question(
        "What is the capital of France?", doc_id=indexed_sample_doc, top_k=3
    )
    # The paper has nothing about France's capital — the LLM should refuse,
    # not hallucinate an answer. This is a soft check since LLM phrasing
    # can vary slightly.
    assert "cannot answer" in result["answer"].lower() or "context" in result["answer"].lower()


def test_no_chunks_returns_refusal_without_llm_call(monkeypatch):
    """If retrieval returns nothing, we should refuse WITHOUT even calling the LLM."""
    monkeypatch.setattr("src.llm.generate.query_similar", lambda *a, **kw: [])
    result = answer_question("anything", doc_id="nonexistent_doc")
    assert result["answer"] == "I cannot answer this from the given context."
    assert result["source_chunks"] == []