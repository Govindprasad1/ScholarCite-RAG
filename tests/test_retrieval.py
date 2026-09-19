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