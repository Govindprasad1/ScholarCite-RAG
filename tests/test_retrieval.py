"""
Stage 2 tests — embeddings + vector store.
Uses the same synthetic sample.pdf fixture from Stage 1 so we already
know exactly what content should map to what page/section.
"""

import shutil
import pytest

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval import vector_store as vs
from src.retrieval.embeddings import embed_texts, get_embedding_model

SAMPLE_PDF = "tests/fixtures/sample.pdf"


@pytest.fixture
def sample_chunks():
    pages = extract_pages(SAMPLE_PDF)
    return chunk_pages(pages)


@pytest.fixture
def isolated_vector_store(tmp_path, monkeypatch):
    """
    Point the vector store at a throwaway temp directory with a unique
    collection name for each test, so tests never touch the real
    data/chroma_db and never collide with each other.
    """
    import src.utils.config as config_module

    original_load_config = config_module.load_config

    def patched_load_config():
        cfg = original_load_config()
        cfg["vector_store"] = dict(cfg["vector_store"])
        cfg["vector_store"]["persist_directory"] = str(tmp_path / "chroma_test")
        cfg["vector_store"]["collection_name"] = "test_collection"
        return cfg

    monkeypatch.setattr(vs, "load_config", patched_load_config)
    yield
    shutil.rmtree(tmp_path / "chroma_test", ignore_errors=True)


def test_embedding_model_loads():
    model = get_embedding_model()
    assert model is not None


def test_embed_texts_returns_correct_shape():
    vectors = embed_texts(["hello world", "another sentence"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384  # bge-small-en-v1.5's output dimension
    assert len(vectors[0]) == len(vectors[1])


def test_embed_empty_list_returns_empty():
    assert embed_texts([]) == []


def test_add_and_query_roundtrip(sample_chunks, isolated_vector_store):
    added = vs.add_chunks(sample_chunks, doc_id="sample_doc")
    assert added == len(sample_chunks)

    results = vs.query_similar("What method did the paper use?", top_k=3, doc_id="sample_doc")
    assert len(results) > 0
    assert all("page_number" in r["metadata"] for r in results)


def test_query_finds_abstract_section(sample_chunks, isolated_vector_store):
    vs.add_chunks(sample_chunks, doc_id="sample_doc")
    results = vs.query_similar("What is this paper about, in summary?", top_k=5, doc_id="sample_doc")
    sections = [r["metadata"]["section"] for r in results]
    assert "Abstract" in sections


def test_delete_document_removes_chunks(sample_chunks, isolated_vector_store):
    vs.add_chunks(sample_chunks, doc_id="doc_to_delete")
    vs.delete_document("doc_to_delete")
    results = vs.query_similar("anything", top_k=5, doc_id="doc_to_delete")
    assert len(results) == 0
from src.retrieval.bm25_search import bm25_query
from src.retrieval.reranker import rerank
from src.retrieval.hybrid import hybrid_query, _reciprocal_rank_fusion


def test_bm25_finds_exact_term_match(sample_chunks, isolated_vector_store):
    vs.add_chunks(sample_chunks, doc_id="sample_doc")
    # "0.91" is an exact figure only in the Experiments section — a good
    # test of BM25's strength (exact numeric/token match) vs. pure semantic search.
    results = bm25_query("0.91", doc_id="sample_doc", top_k=3)
    assert len(results) > 0
    assert any("0.91" in r["text"] for r in results)


def test_reranker_reorders_by_relevance(sample_chunks, isolated_vector_store):
    vs.add_chunks(sample_chunks, doc_id="sample_doc")
    candidates = vs.query_similar("What method was used?", top_k=5, doc_id="sample_doc")
    reranked = rerank("What method was used?", candidates, top_k=3)
    assert len(reranked) <= 3
    assert all("rerank_score" in r for r in reranked)


def test_reciprocal_rank_fusion_favors_chunks_in_both_lists():
    chunk_a = {"text": "chunk A", "metadata": {"page_number": 1, "section": "X"}}
    chunk_b = {"text": "chunk B", "metadata": {"page_number": 2, "section": "Y"}}
    chunk_c = {"text": "chunk C", "metadata": {"page_number": 3, "section": "Z"}}

    vector_results = [chunk_a, chunk_b]  # A ranked 1st, B ranked 2nd
    bm25_results = [chunk_b, chunk_c]    # B ranked 1st, C ranked 2nd

    merged = _reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    # B appears highly ranked in BOTH lists, so it should come out on top
    assert merged[0]["text"] == "chunk B"


def test_hybrid_query_returns_final_top_k(sample_chunks, isolated_vector_store):
    vs.add_chunks(sample_chunks, doc_id="sample_doc")
    results = hybrid_query("What did the experiments show?", doc_id="sample_doc")
    assert len(results) > 0
    assert len(results) <= 5  # final_top_k from config.yaml